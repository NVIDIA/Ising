# Ising-QEC

Real-time QEC decoding experiments on top of [NVIDIA/Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding), targeting d=5 surface-code decoding with biased (Z:X = 10:1) noise on Harvard Cannon H200 hardware. End-to-end round-trip meets the <10 µs target.

Headline numbers (H200, FP8 TRT engine, batch=1, full H2D → engine → D2H captured as one CUDA graph):

```
roundtrip  mean  6.45 µs   p99  7.68 µs   engine-only  3.87 µs
fine-tune LER (Avg X/Z)   0.0338 (FP32)   0.0355 (FP8 TRT)
PyMatching speedup with neural pre-decoder: 1.60×
```

See `SUMMARY.md` for the numbers table and `notes.md` for the original plan.

## What this adds to upstream

- **Singularity container** (`env/ising.def` → `env/ising.sif`): torch 2.10+cu128, stim 1.15, pymatching 2.3, **tensorrt 10.12 cu12-pinned** (bare `tensorrt` pulls cu13 libs that segfault on some kempner_eng nodes).
- **d=5 biased-noise config** (`conf/config_ising_qec_d5_biased.yaml`, symlinked into `Ising-Decoding/conf/`) — drops distance/n_rounds to 5 and rebalances the 25-parameter noise model for Z:X = 10:1 at total p ≈ 6e-3.
- **SLURM wrappers** for kempner_eng (H200): `smoke_eng`, `bootstrap_verify_eng`, `train_eng`, `pytest_eng`, `export_fp8_eng`, `latency_eng`, `latency_trt_eng`, `roundtrip_eng`.
- **Pretrained-weight seed path**: `train_eng.sbatch` copies `Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Fast.pt` into `outputs/outputs/<EXP>/models/PreDecoderModelMemory_v1.0.0.pt` so upstream's `load_checkpoint` picks it up as the initialization.
- **Benchmark tools**: `bench/latency.py` (PyTorch forward), `bench/latency_trt.py` (TRT engine), `bench/roundtrip.py` (host ↔ device round-trip as one captured CUDA graph).

## Layout

```
Ising-QEC/
├── env/                 ising.def + ising.sif + build.log
├── conf/                local Hydra configs (symlinked into Ising-Decoding/conf/)
├── slurm/               sbatch scripts (all on kempner_eng)
├── bench/               latency + round-trip benchmarks
├── logs/                SLURM stdout/stderr
├── outputs/             training outputs (nested: outputs/outputs/<EXP>/)
├── Ising-Decoding/      clone of NVIDIA/Ising-Decoding (Git LFS weights)
├── notes.md             original project notes (has known errors — see the plan file in ~/.claude/plans/)
├── README.md
└── SUMMARY.md           metrics + artifacts + gotchas for later review
```

## Prerequisites

- FAS-RC Cannon account with access to `kempner_eng` (partition this repo defaults to).
- `singularity-ce` 4.x on the submit host.
- `git-lfs` installed (needed for the `.pt` weights under `Ising-Decoding/models/`).
- `~/.bashrc` provides `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` under netscratch.

## First-time setup

```bash
git clone https://github.com/dmbala/Ising-QEC
cd Ising-QEC

# Upstream (not committed here — provides code, weights, and default configs).
git lfs install
git clone https://github.com/NVIDIA/Ising-Decoding

# Symlink local configs into the Hydra search path.
ln -sfn $PWD/conf/config_ising_qec_d5_biased.yaml \
        Ising-Decoding/conf/config_ising_qec_d5_biased.yaml

# Container (one-shot, ~15-20 min on netscratch).
singularity build --fakeroot env/ising.sif env/ising.def
```

## Running

All sbatch scripts assume you `cd` into the repo root first. Every job goes to **kempner_eng** (H200). Default experiment is `qec-decoder-d5-biased`, default config is `config_ising_qec_d5_biased`.

```bash
# 0. Env sanity: imports + upstream smoke train (~5 min).
sbatch slurm/smoke_eng.sbatch

# 1. Verify the Fast-weight bootstrap drops loss from 0.69 → 0.045 in 1 epoch.
sbatch slurm/bootstrap_verify_eng.sbatch

# 2. 50-epoch biased fine-tune. Recommend PREDECODER_TRAIN_SAMPLES=262144
#    so a full schedule fits in ~70 min; raw default is ~125 h.
sbatch --export=ALL,PREDECODER_TRAIN_SAMPLES=262144 slurm/train_eng.sbatch

# 3. FP8 ONNX export + TRT engine build + quantized LER (~10 min).
sbatch slurm/export_fp8_eng.sbatch

# 4. Compute-only latency (PyTorch FP32 reference).
sbatch slurm/latency_eng.sbatch

# 5. Compute-only latency on the FP8 TRT engine.
sbatch slurm/latency_trt_eng.sbatch

# 6. Round-trip wall time (host-pinned H2D + engine + D2H, one captured graph).
sbatch slurm/roundtrip_eng.sbatch
```

Monitor with `squeue -u $USER` and tail `logs/<job>_<id>.out`.

## Config: biased noise

`conf/config_ising_qec_d5_biased.yaml` extends the public surface with 25-parameter depolarizing overrides. Bias is applied to the **bulk** Pauli channels only (idle-during-CNOT, idle-during-SPAM, and the two-qubit CNOT pair distribution); state-prep and measurement errors stay symmetric. Z:X ratio is 10:1; total per-gate error rate is ~6e-3 (just below the depolarizing threshold). Adjust the `data.noise_model` block to sweep.

## Gotchas

Full list with reproduction steps is in `SUMMARY.md`. Short version:

- **Partition:** always use `kempner_eng` (H200). kempner_dev sometimes pends for hours; kempner_dev A100 nodes additionally segfault in the cuStabilizer path.
- **TensorRT:** pin `tensorrt-cu12>=10.8,<10.13`. The bare `tensorrt` package pulls cu13 libs that fail with `CUDA initialization failure with error: 35` on older drivers.
- **Per-epoch LER:** set `PREDECODER_LER_FINAL_ONLY=1` during training. The cuStabilizer basis=Z path segfaults after dozens of repeated calls — fine for the final once-off LER call, crashes during per-epoch validation.
- **Samples per epoch:** upstream default is ~8.4 M samples/epoch (~125 h for 50 epochs). Override with `PREDECODER_TRAIN_SAMPLES=262144` for iteration speed.
- **Receptive field:** the R=9 Fast model always operates on a 9×9×9 window. `distance=5` and `n_rounds=5` in the config affect evaluation only; training data still arrives at 9×9×9.
- **Output path:** `cluster_train.sh` appends `outputs/` to `$SHARED_OUTPUT_DIR`, giving the nested `outputs/outputs/<EXP>/` layout. Cosmetic.

## Reference

- Upstream: https://github.com/NVIDIA/Ising-Decoding
- Paper: https://research.nvidia.com/publication/2026-04_fast-ai-based-pre-decoders-surface-codes
- Fast model (R=9, 0.91M): `Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Fast.pt`
- Accurate model (R=13, 1.79M): `Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Accurate.pt`
- Plan summary: see `notes.md`
