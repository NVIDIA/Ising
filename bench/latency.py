"""
Compute-only latency benchmark for the fine-tuned Ising pre-decoder.

Loads the d=5 biased-noise best_model checkpoint, times the neural forward pass
on [1, C_in, T, X, Y] input at batch=1 using CUDA events, with and without
CUDA-graph capture. Does not touch TensorRT or ModelOpt.

Run via the project's Singularity image; the Ising-Decoding repo's code tree
must be on PYTHONPATH so model/factory.py and workflows/config_validator.py
are importable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def _make_model(cfg: DictConfig) -> torch.nn.Module:
    from model.factory import ModelFactory

    return ModelFactory.create_model(cfg)


def _load_cfg(repo_root: Path, config_name: str) -> DictConfig:
    from workflows.config_validator import (
        apply_public_defaults_and_model,
        validate_public_config,
    )

    conf_dir = str(repo_root / "conf")
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        cfg = compose(config_name=config_name, overrides=["workflow.task=inference"])
    model_spec = validate_public_config(cfg)
    cfg = apply_public_defaults_and_model(cfg, model_spec)
    return cfg


def _find_checkpoint(experiment_dir: Path) -> Path:
    best = sorted(
        (experiment_dir / "models" / "best_model").glob("PreDecoderModelMemory_v1.0.*.pt"),
        key=lambda p: int(p.stem.rsplit(".", 1)[-1]),
    )
    if best:
        return best[-1]
    latest = sorted(
        (experiment_dir / "models").glob("PreDecoderModelMemory_v1.0.*.pt"),
        key=lambda p: int(p.stem.rsplit(".", 1)[-1]),
    )
    if not latest:
        raise SystemExit(f"no checkpoint found under {experiment_dir}/models")
    return latest[-1]


def _probe_input_channels(model: torch.nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            return int(m.in_channels)
    raise RuntimeError("could not find a Conv3d to probe input channel count")


def _time_eager(model, x, iters, device):
    # CUDA-event timing of eager forward passes.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_us = []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start.record()
        with torch.no_grad():
            _ = model(x)
        end.record()
        torch.cuda.synchronize(device)
        times_us.append(start.elapsed_time(end) * 1000.0)
    return times_us


def _time_cuda_graph(model, x, iters, device, warmup=3):
    # Capture the forward pass in a CUDA graph and replay iters times.
    with torch.no_grad():
        # Warm up in a side stream before capture.
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(warmup):
                _ = model(x)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        static_out = torch.empty_like(model(x))
        with torch.cuda.graph(graph):
            static_out.copy_(model(x))

    times_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        times_us.append(start.elapsed_time(end) * 1000.0)
    return times_us


def _stats(xs):
    xs = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    p50 = xs[n // 2]
    p99 = xs[min(n - 1, int(0.99 * n))]
    return mean, p50, p99, min(xs), max(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ising-dir", default="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC/Ising-Decoding")
    ap.add_argument("--config-name", default="config_ising_qec_d5_biased")
    ap.add_argument("--experiment-dir", default="/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC/outputs/outputs/qec-decoder-d5-biased")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = ap.parse_args()

    ising_dir = Path(args.ising_dir).resolve()
    sys.path.insert(0, str(ising_dir / "code"))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    cfg = _load_cfg(ising_dir, args.config_name)
    model = _make_model(cfg).to(device).eval()

    ckpt_path = _find_checkpoint(Path(args.experiment_dir))
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"[latency] loaded {ckpt_path}")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if dtype != torch.float32:
        model = model.to(dtype)

    in_ch = _probe_input_channels(model)
    d, r = int(cfg.distance), int(cfg.n_rounds)
    R = int(cfg.model.num_filters and max(cfg.model.kernel_size) or r)
    # PreDecoderModelMemory_v1 expects [B, C_in, T, X, Y] matching its receptive field.
    # For Model 1 (Fast), R=9 so T=X=Y=9 regardless of config d/n_rounds.
    shape = (1, in_ch, 9, 9, 9)
    x = torch.randn(*shape, device=device, dtype=dtype)
    print(f"[latency] input shape {tuple(x.shape)} dtype={dtype}")

    # Eager warmup + measurement.
    _ = _time_eager(model, x, args.warmup, device)
    eager = _time_eager(model, x, args.iters, device)
    mean, p50, p99, mn, mx = _stats(eager)
    print(f"[eager]       mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}  (iters={args.iters})")

    # CUDA graph capture + replay.
    try:
        g = _time_cuda_graph(model, x, args.iters, device)
        mean, p50, p99, mn, mx = _stats(g)
        print(f"[cuda-graph]  mean={mean:.2f}us  p50={p50:.2f}  p99={p99:.2f}  min={mn:.2f}  max={mx:.2f}  (iters={args.iters})")
    except Exception as exc:
        print(f"[cuda-graph]  SKIPPED ({exc!r})")


if __name__ == "__main__":
    main()
