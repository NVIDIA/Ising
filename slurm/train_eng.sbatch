#!/usr/bin/env bash
#SBATCH --job-name=ising-train-r9
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train_eng_r9_%j.out
#SBATCH --error=logs/train_eng_r9_%j.err

# Biased-noise fine-tune on kempner_eng (H200). Use for dev/eng partition.
# Override CONFIG_NAME / EXPERIMENT_NAME / PREDECODER_TRAIN_SAMPLES / FRESH_START
# via --export at submit time.

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
SHARED_OUTPUT_DIR="${REPO_ROOT}/outputs"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qec-decoder-d5-biased}"
CONFIG_NAME="${CONFIG_NAME:-config_ising_qec_d5_biased}"
FRESH_START="${FRESH_START:-0}"
EPOCHS="${PREDECODER_TRAIN_EPOCHS:-50}"
FAST_PT="${ISING_DIR}/models/Ising-Decoder-SurfaceCode-1-Fast.pt"
MODEL_DIR="${SHARED_OUTPUT_DIR}/outputs/${EXPERIMENT_NAME}/models"
BOOTSTRAP_PT="${MODEL_DIR}/PreDecoderModelMemory_v1.0.0.pt"

mkdir -p "$MODEL_DIR"
cd "$REPO_ROOT"

if [ "${FRESH_START}" != "1" ] && [ ! -f "$BOOTSTRAP_PT" ] && ! ls "${MODEL_DIR}"/PreDecoderModelMemory_v1.0.*.pt >/dev/null 2>&1; then
    echo "[train] seeding ${BOOTSTRAP_PT} from Fast pretrained weights"
    cp "$FAST_PT" "$BOOTSTRAP_PT"
fi

echo "[train] node=$(hostname)"
echo "[train] experiment=$EXPERIMENT_NAME config=$CONFIG_NAME fresh=$FRESH_START epochs=$EPOCHS samples=${PREDECODER_TRAIN_SAMPLES:-default}"
nvidia-smi -L

optional_env_args=()
if [ -n "${PREDECODER_TRAIN_SAMPLES:-}" ]; then
    optional_env_args+=(--env "PREDECODER_TRAIN_SAMPLES=${PREDECODER_TRAIN_SAMPLES}")
fi

singularity exec --nv \
    --bind "${ISING_DIR}:${ISING_DIR}" \
    --bind "${SHARED_OUTPUT_DIR}:${SHARED_OUTPUT_DIR}" \
    --env SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}" \
    --env EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    --env CONFIG_NAME="${CONFIG_NAME}" \
    --env WORKFLOW=train \
    --env GPUS=1 \
    --env FRESH_START="${FRESH_START}" \
    --env PREDECODER_TRAIN_EPOCHS="${EPOCHS}" \
    --env PREDECODER_DISABLE_SDR=1 \
    --env PREDECODER_TORCH_COMPILE=0 \
    --env PREDECODER_VERBOSE=1 \
    --env PREDECODER_LER_FINAL_ONLY=1 \
    "${optional_env_args[@]}" \
    "$SIF" bash "${ISING_DIR}/code/scripts/cluster_train.sh"

echo "[train] done"
