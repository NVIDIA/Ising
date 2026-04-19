"""
End-to-end round-trip latency for the FP8 TRT engine:
  host pinned buffer  ->  H2D  ->  engine replay  ->  D2H  ->  host pinned buffer.

This emulates the I/O path a realtime controller would see: one syndrome arrives,
we push it to the GPU, the pre-decoder runs, and the correction decision lands
back in host memory. Syndromes are drawn from stim ahead of time so the timed
section covers only the transfer + compute path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

try:
    import stim  # noqa: F401
    _HAS_STIM = True
except Exception:
    _HAS_STIM = False


_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.FP8: torch.uint8,
}


def _load_engine(path: str, logger: trt.Logger) -> trt.ICudaEngine:
    runtime = trt.Runtime(logger)
    blob = Path(path).read_bytes()
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise SystemExit(f"could not deserialize engine {path}")
    return engine


def _make_buffers(engine, context, device):
    """Allocate one persistent device I/O buffer pair and matching pinned host buffers."""
    buffers = {"in": None, "out": None}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        shape = tuple(1 if s == -1 else s for s in engine.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)
        dev = torch.empty(*shape, dtype=dtype, device=device)
        host = torch.empty(*shape, dtype=dtype, pin_memory=True)
        context.set_tensor_address(name, int(dev.data_ptr()))
        key = "in" if mode == trt.TensorIOMode.INPUT else "out"
        buffers[key] = {"name": name, "dev": dev, "host": host, "shape": shape, "dtype": dtype}
    return buffers


def _sample_syndromes(count: int, shape: tuple[int, ...], distance: int, rounds: int, rng) -> np.ndarray:
    """
    Produce `count` syndrome vectors matching the engine input shape.

    Uses stim if available and the detector count matches. Falls back to
    uniform-random bytes (same shape) when stim produces a different count.
    """
    flat = int(np.prod(shape[1:]))  # drop leading batch dim
    out = np.empty((count, *shape[1:]), dtype=np.uint8)
    if _HAS_STIM:
        try:
            circuit = stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=distance,
                rounds=rounds,
                after_clifford_depolarization=0.006,
                before_round_data_depolarization=0.006,
                after_reset_flip_probability=0.006,
                before_measure_flip_probability=0.006,
            )
            sampler = circuit.compile_detector_sampler()
            syn = sampler.sample(count).astype(np.uint8)
            if syn.shape[1] == flat:
                out[:] = syn.reshape(count, *shape[1:])
                return out
            print(f"[roundtrip] stim produced {syn.shape[1]} detectors; engine expects {flat}. Using random fill.")
        except Exception as exc:
            print(f"[roundtrip] stim sampling failed ({exc!r}); using random fill.")
    out[:] = rng.integers(0, 2, size=out.shape, dtype=np.uint8)
    return out


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return sum(xs) / n, xs[n // 2], xs[min(n - 1, int(0.99 * n))], xs[0], xs[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda:0")
    logger = trt.Logger(trt.Logger.WARNING)

    engine = _load_engine(args.engine, logger)
    context = engine.create_execution_context()
    bufs = _make_buffers(engine, context, device)
    assert bufs["in"] is not None and bufs["out"] is not None

    print(f"[roundtrip] engine={Path(args.engine).name}")
    print(f"  input  {bufs['in']['name']:<22} shape={bufs['in']['shape']}  dtype={bufs['in']['dtype']}")
    print(f"  output {bufs['out']['name']:<22} shape={bufs['out']['shape']}  dtype={bufs['out']['dtype']}")

    rng = np.random.default_rng(42)
    total = args.warmup + args.iters
    syn = _sample_syndromes(total, bufs["in"]["shape"], args.distance, args.rounds, rng)
    syn_t = torch.from_numpy(syn)  # CPU

    stream = torch.cuda.Stream()

    host_in = bufs["in"]["host"]
    host_out = bufs["out"]["host"]
    dev_in = bufs["in"]["dev"]
    dev_out = bufs["out"]["dev"]

    # Warm up the full H2D -> engine -> D2H chain on a side stream before capture.
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        for _ in range(3):
            dev_in.copy_(host_in, non_blocking=True)
            context.execute_async_v3(side.cuda_stream)
            host_out.copy_(dev_out, non_blocking=True)
    torch.cuda.synchronize()

    # Full-chain graph: H2D + engine + D2H all captured into a single replay.
    # Only the pinned buffer address is fixed; callers update its contents
    # from the CPU side between replays.
    graph_full = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_full, stream=stream):
        dev_in.copy_(host_in, non_blocking=True)
        context.execute_async_v3(stream.cuda_stream)
        host_out.copy_(dev_out, non_blocking=True)

    # Engine-only graph, kept for the diagnostic breakdown.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        context.execute_async_v3(stream.cuda_stream)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup: full chain as one graph replay.
    for i in range(args.warmup):
        host_in.copy_(syn_t[i])
        torch.cuda.synchronize()
        graph_full.replay()
        torch.cuda.synchronize()

    # Timed loop: single captured graph replay does H2D + engine + D2H.
    times_us = []
    for i in range(args.iters):
        host_in.copy_(syn_t[args.warmup + i])
        torch.cuda.synchronize()
        start.record(stream)
        graph_full.replay()
        end.record(stream)
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)

    mean, p50, p99, mn, mx = _stats(times_us)
    print(f"[roundtrip-graph]  mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}  (iters={args.iters})")

    # Legacy eager 3-launch chain for comparison.
    eager_times = []
    for i in range(args.iters):
        host_in.copy_(syn_t[args.warmup + i])
        torch.cuda.synchronize()
        start.record(stream)
        with torch.cuda.stream(stream):
            dev_in.copy_(host_in, non_blocking=True)
            graph.replay()
            host_out.copy_(dev_out, non_blocking=True)
        end.record(stream)
        torch.cuda.synchronize()
        eager_times.append(start.elapsed_time(end) * 1000.0)
    mean, p50, p99, mn, mx = _stats(eager_times)
    print(f"[roundtrip-eager]  mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}")

    # Break down the fixed pieces for diagnostic purposes.
    # H2D only
    h2d = []
    for i in range(args.iters):
        host_in.copy_(syn_t[i])
        torch.cuda.synchronize()
        start.record(stream)
        with torch.cuda.stream(stream):
            dev_in.copy_(host_in, non_blocking=True)
        end.record(stream)
        torch.cuda.synchronize()
        h2d.append(start.elapsed_time(end) * 1000.0)
    mean, p50, p99, mn, mx = _stats(h2d)
    print(f"[H2D only]   mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}")

    # D2H only
    d2h = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        start.record(stream)
        with torch.cuda.stream(stream):
            host_out.copy_(dev_out, non_blocking=True)
        end.record(stream)
        torch.cuda.synchronize()
        d2h.append(start.elapsed_time(end) * 1000.0)
    mean, p50, p99, mn, mx = _stats(d2h)
    print(f"[D2H only]   mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}")

    # Engine replay only (graph)
    eng = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        start.record(stream)
        with torch.cuda.stream(stream):
            graph.replay()
        end.record(stream)
        torch.cuda.synchronize()
        eng.append(start.elapsed_time(end) * 1000.0)
    mean, p50, p99, mn, mx = _stats(eng)
    print(f"[engine]     mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}")


if __name__ == "__main__":
    main()
