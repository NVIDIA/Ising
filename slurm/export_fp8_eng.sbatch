#!/usr/bin/env bash
#SBATCH --job-name=ising-fp8-export
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/fp8_%j.out
#SBATCH --error=logs/fp8_%j.err

# FP8 ONNX export + TensorRT build + TRT inference on the fine-tuned
# best_model checkpoint for qec-decoder-d5-biased.
#
# Upstream inference loader picks the highest-indexed PreDecoderModelMemory_v1.0.*.pt
# under <resume_dir>, which equals <output>/models. The final epoch (49) save is the
# best model (LER-driven ckpt_best trigger at epoch 49), so the default is what we want.

set -euo pipefail

REPO_ROOT=/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/Ising-QEC
SIF="${REPO_ROOT}/env/ising.sif"
ISING_DIR="${REPO_ROOT}/Ising-Decoding"
SHARED_OUTPUT_DIR="${REPO_ROOT}/outputs"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qec-decoder-d5-biased}"
CONFIG_NAME="${CONFIG_NAME:-config_ising_qec_d5_biased}"

cd "$REPO_ROOT"
echo "[fp8] node=$(hostname)"
nvidia-smi -L

singularity exec --nv \
    --bind "${ISING_DIR}:${ISING_DIR}" \
    --bind "${SHARED_OUTPUT_DIR}:${SHARED_OUTPUT_DIR}" \
    --env SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR}" \
    --env EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    --env CONFIG_NAME="${CONFIG_NAME}" \
    --env WORKFLOW=inference \
    --env GPUS=1 \
    --env FRESH_START=0 \
    --env ONNX_WORKFLOW="${ONNX_WORKFLOW:-2}" \
    --env QUANT_FORMAT="${QUANT_FORMAT:-fp8}" \
    --env PREDECODER_DISABLE_SDR=1 \
    --env PREDECODER_TORCH_COMPILE=0 \
    --env PREDECODER_VERBOSE=1 \
    --env PREDECODER_INFERENCE_NUM_SAMPLES="${PREDECODER_INFERENCE_NUM_SAMPLES:-4096}" \
    --env PREDECODER_INFERENCE_MEAS_BASIS=both \
    --env PREDECODER_INFERENCE_NUM_WORKERS=0 \
    "$SIF" bash "${ISING_DIR}/code/scripts/cluster_train.sh"

echo "[fp8] done"
