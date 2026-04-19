#!/usr/bin/env bash
#SBATCH --job-name=ising-roundtrip
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/roundtrip_%j.out
#SBATCH --error=logs/roundtrip_%j.err

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
ENGINE_X="${ISING_DIR}/predecoder_memory_d5_T5_X_fp8.engine"
ENGINE_Z="${ISING_DIR}/predecoder_memory_d5_T5_Z_fp8.engine"

cd "$REPO_ROOT"
echo "[rtt] node=$(hostname)"
nvidia-smi -L

for engine in "$ENGINE_X" "$ENGINE_Z"; do
    echo "=== $engine ==="
    singularity exec --nv \
        --bind "${ISING_DIR}:${ISING_DIR}" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --env PYTHONUNBUFFERED=1 \
        "$SIF" python "${REPO_ROOT}/bench/roundtrip.py" \
            --engine "$engine" \
            --warmup "${WARMUP:-50}" \
            --iters "${ITERS:-1000}" \
            --distance 5 \
            --rounds 5
done

echo "[rtt] done"
