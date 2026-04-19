"""
Compute-only latency benchmark for the FP8 TensorRT engine.

Loads one of the *_fp8.engine files, allocates static input/output tensors on
the GPU, and times execute_async_v3 at batch=1 using CUDA events. Also times a
CUDA-graph-captured replay when the runtime supports it.
"""
from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch


_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.FP8: torch.uint8,  # FP8 shares an 8-bit byte layout; torch has no native FP8 dtype.
}


def _load_engine(path: str, logger: trt.Logger) -> trt.ICudaEngine:
    runtime = trt.Runtime(logger)
    with open(path, "rb") as fh:
        blob = fh.read()
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise SystemExit(f"could not deserialize engine {path}")
    return engine


def _alloc_tensors(engine: trt.ICudaEngine, context: trt.IExecutionContext, device: torch.device):
    bindings = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        shape = list(engine.get_tensor_shape(name))
        # Handle dynamic dims by assuming batch=1 where the shape is -1.
        shape = [1 if s == -1 else s for s in shape]
        context.set_input_shape(name, tuple(shape)) if mode == trt.TensorIOMode.INPUT else None
        t = torch.empty(*shape, dtype=dtype, device=device)
        context.set_tensor_address(name, int(t.data_ptr()))
        bindings[name] = {"tensor": t, "mode": mode, "shape": tuple(shape), "dtype": dtype}
    return bindings


def _fill_input(bindings):
    for name, info in bindings.items():
        if info["mode"] == trt.TensorIOMode.INPUT:
            if info["dtype"].is_floating_point:
                info["tensor"].uniform_(-1.0, 1.0)
            else:
                info["tensor"].random_(0, 2)


def _time_eager(context, stream, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record(stream)
        context.execute_async_v3(stream.cuda_stream)
        end.record(stream)
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return times_us


def _time_cuda_graph(context, stream, iters, warmup=3):
    # Warm up in a side stream so TRT can pick its tactics, then capture.
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        for _ in range(warmup):
            context.execute_async_v3(side.cuda_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        context.execute_async_v3(stream.cuda_stream)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record(stream)
        graph.replay()
        end.record(stream)
        torch.cuda.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)
    return times_us


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    return sum(xs) / n, xs[n // 2], xs[min(n - 1, int(0.99 * n))], xs[0], xs[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda:0")
    logger = trt.Logger(trt.Logger.WARNING)

    engine = _load_engine(args.engine, logger)
    context = engine.create_execution_context()
    bindings = _alloc_tensors(engine, context, device)

    print(f"[trt-lat] engine={Path(args.engine).name}")
    for name, info in bindings.items():
        print(f"  {info['mode']!s:>22}  {name:>20}  shape={info['shape']}  dtype={info['dtype']}")

    _fill_input(bindings)
    stream = torch.cuda.Stream()

    # Warmup (eager).
    _ = _time_eager(context, stream, args.warmup)

    eager = _time_eager(context, stream, args.iters)
    mean, p50, p99, mn, mx = _stats(eager)
    print(f"[eager]       mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}  (iters={args.iters})")

    try:
        g = _time_cuda_graph(context, stream, args.iters)
        mean, p50, p99, mn, mx = _stats(g)
        print(f"[cuda-graph]  mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}  (iters={args.iters})")
    except Exception as exc:
        print(f"[cuda-graph]  SKIPPED ({exc!r})")


if __name__ == "__main__":
    main()
