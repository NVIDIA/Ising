#!/usr/bin/env bash
#SBATCH --job-name=ising-latency
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/latency_%j.out
#SBATCH --error=logs/latency_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"

cd "$REPO_ROOT"
echo "[lat] node=$(hostname)"
nvidia-smi -L

singularity exec --nv \
    --bind "${ISING_DIR}:${ISING_DIR}" \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" python "${REPO_ROOT}/bench/latency.py" \
        --ising-dir "${ISING_DIR}" \
        --config-name config_ising_qec_d5_biased \
        --experiment-dir "${REPO_ROOT}/outputs/outputs/qec-decoder-d5-biased" \
        --warmup "${WARMUP:-50}" \
        --iters "${ITERS:-1000}" \
        --dtype "${DTYPE:-fp32}"

echo "[lat] done"
