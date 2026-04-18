# Ising-QEC

Real-time QEC decoding experiments on top of [NVIDIA/Ising-Decoding](https://github.com/NVIDIA/Ising-Decoding), targeting d=5 surface-code decoding with biased (Z:X = 10:1) noise on Harvard Cannon H100/H200 hardware.

## What this adds to upstream

- **Singularity container** (`env/ising.def` → `env/ising.sif`) built from the upstream CUDA-12.1 Dockerfile, pinned at torch 2.10+cu128, stim 1.15, pymatching 2.3, tensorrt 10.16.
- **d=5 biased-noise config** (`conf/config_ising_qec_d5_biased.yaml`, symlinked into `Ising-Decoding/conf/`) — drops distance/n_rounds to 5 and rebalances the 25-parameter noise model for Z:X = 10:1 at total p ≈ 6e-3.
- **SLURM wrappers** for Kempner partitions: `slurm/smoke_eng.sbatch` (env sanity), `slurm/bootstrap_verify_eng.sbatch` (1-epoch pretrained-weight load check), `slurm/train_dev.sbatch` (50-epoch fine-tune), `slurm/pytest_eng.sbatch` (upstream test suite).
- **Pretrained-weight seed path**: on the first run, `train_dev.sbatch` copies `Ising-Decoding/models/Ising-Decoder-SurfaceCode-1-Fast.pt` into `outputs/outputs/<EXP>/models/PreDecoderModelMemory_v1.0.0.pt` so upstream's `load_checkpoint` picks it up as the initialization.

## Layout

```
Ising-QEC/
├── env/                 ising.def + built ising.sif + build.log
├── conf/                local Hydra configs (symlinked into Ising-Decoding/conf/)
├── slurm/               sbatch scripts for smoke/bootstrap/train/pytest
├── logs/                SLURM stdout/stderr
├── outputs/             training outputs (nested: outputs/outputs/<EXP>/)
├── Ising-Decoding/      clone of NVIDIA/Ising-Decoding (Git LFS weights)
├── notes.md             original project notes (has known errors — see /n/home07/bdesinghu/.claude/plans/...md)
└── README.md
```

## Prerequisites

- FAS-RC Cannon account with access to GPU partitions. 
- `singularity-ce` 4.x on submit host.
- `git-lfs` installed (needed to pull the `.pt` weights).
- `~/.bashrc` provides `SINGULARITY_CACHEDIR` and `SINGULARITY_TMPDIR` under netscratch.

## First-time setup

```bash
git clone https://github.com/dmbala/Ising-QEC

cd Ising-QEC

# Container (one-shot, ~15-20 min on netscratch).
singularity build --fakeroot env/ising.sif env/ising.def
```

Weights and the upstream repo are already on disk under `Ising-Decoding/`.

## Running

All sbatch scripts assume you `cd` into the repo root first.

```bash
# 1. Sanity: imports + upstream smoke train (kempner_eng, ~5 min).
sbatch slurm/smoke_eng.sbatch

# 2. Verify the Fast-weight bootstrap drops loss from 0.69 → 0.045 in 1 epoch (kempner_eng, ~5 min).
sbatch slurm/bootstrap_verify_eng.sbatch

# 3. Full 50-epoch biased fine-tune (kempner_dev, H100/H200; override EPOCHS / FRESH_START via --export).
sbatch slurm/train_dev.sbatch

# 4. Upstream pytest inside our container (optional, kempner_eng).
sbatch slurm/pytest_eng.sbatch
```

Monitor with `squeue -u $USER` and tail `logs/<job>_<id>.out`.

## Config: biased noise

`conf/config_ising_qec_d5_biased.yaml` extends the public surface with 25-parameter depolarizing overrides. Bias is applied to the **bulk** Pauli channels only (idle-during-CNOT, idle-during-SPAM, and the two-qubit CNOT pair distribution); state-prep and measurement errors stay symmetric. Z:X ratio is 10:1; total per-gate error rate is ~6e-3 (at/just below the depolarizing threshold). Adjust the `data.noise_model` block to sweep.

## Current state

- Container built, smoke + bootstrap verified on 2026-04-18.
- 50-epoch fine-tune from Fast weights is the current experiment.
- Outputs land at `outputs/outputs/qec-decoder-d5-biased/` (the doubled `outputs/outputs/` comes from upstream's `cluster_train.sh` appending "outputs" to `$SHARED_OUTPUT_DIR`; cosmetic, not a bug).

## Known issues / open items

- Runtime warning `Int8 GEMM failed on cuda:0, permanently falling back to float32 for weight reduction` appears during DEM build. Benign for training; may need attention if we exercise the INT8 path at export time.
- Validation generator runs at d=9 (upstream receptive-field constraint), test generator targets d=5 — both using the biased noise config. If we want val on d=5 too, need a custom val_generator override.
- I/O path for the <10 µs real-time target is **not** designed yet; current latency numbers will be compute-only (CUDA graph, batch=1) until we wire a streaming simulator handoff.
- Noise parameter sweep schedule is ad hoc — pick a reference paper or range before generating many runs.

## Reference

- Repo + paper: https://github.com/NVIDIA/Ising-Decoding
