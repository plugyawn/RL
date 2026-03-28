#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/dg_gsm8k_long_compare_${STAMP}}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-1}"

UV_RUN=(uv run)
if [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  UV_RUN+=(--no-sync)
fi
UV_RUN+=(python)

# Defaults here are tuned for a larger multi-GPU node and to reduce the
# truncation seen in the 8-step pilot, while staying close to grpo_math_1B.
CONFIG_PATH="${CONFIG_PATH:-examples/configs/grpo_math_1B.yaml}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
PROMPT_FILE="${PROMPT_FILE:-examples/prompts/cot.txt}"
SEED="${SEED:-42}"
DG_ETA_MAIN="${DG_ETA_MAIN:-0.5}"
MAX_STEPS="${MAX_STEPS:-128}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-32}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-16}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-512}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-4}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-4}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
VAL_PERIOD="${VAL_PERIOD:-16}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
SAVE_PERIOD="${SAVE_PERIOD:-1000000}"
MATH_NUM_WORKERS="${MATH_NUM_WORKERS:-8}"
CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE:-8}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-1}"
DTENSOR_V2="${DTENSOR_V2:-0}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-0}"
CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-1}"
MODEL_SAVE_FORMAT="${MODEL_SAVE_FORMAT:-null}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-dg-grpo}"
BASELINE_WANDB_NAME="${BASELINE_WANDB_NAME:-}"
DG_WANDB_NAME="${DG_WANDB_NAME:-}"

mkdir -p "$OUT_ROOT"

run_one() {
  local label="$1"
  local dg_enabled="$2"
  local dg_eta="$3"
  local run_dir="${OUT_ROOT}/${label}"
  local wandb_name="$4"

  echo "=== Running ${label} ==="
  CONFIG_PATH="${CONFIG_PATH}" \
  MODEL_NAME="${MODEL_NAME}" \
  TOKENIZER_NAME="${TOKENIZER_NAME}" \
  PROMPT_FILE="${PROMPT_FILE}" \
  SEED="${SEED}" \
  DG_ENABLED="${dg_enabled}" \
  DG_ETA="${dg_eta}" \
  MAX_STEPS="${MAX_STEPS}" \
  NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP}" \
  NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT}" \
  TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE}" \
  TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE}" \
  LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE}" \
  MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  VAL_PERIOD="${VAL_PERIOD}" \
  MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
  SAVE_PERIOD="${SAVE_PERIOD}" \
  MATH_NUM_WORKERS="${MATH_NUM_WORKERS}" \
  CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE}" \
  CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES}" \
  DTENSOR_V2="${DTENSOR_V2}" \
  CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED}" \
  CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K}" \
  MODEL_SAVE_FORMAT="${MODEL_SAVE_FORMAT}" \
  WANDB_ENABLED="${WANDB_ENABLED}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_NAME="${wandb_name}" \
  RUN_NAME="${label}" \
  RUN_DIR="${run_dir}" \
    bash "${ROOT_DIR}/tools/run_dg_gsm8k_full.sh"
}

run_one "baseline" "0" "${DG_ETA_MAIN}" "${BASELINE_WANDB_NAME}"
run_one "dg_eta${DG_ETA_MAIN}" "1" "${DG_ETA_MAIN}" "${DG_WANDB_NAME}"

PLOTS_DIR="${OUT_ROOT}/plots"
STABILITY_DIR="${OUT_ROOT}/stability"

(
  cd "$ROOT_DIR"
  "${UV_RUN[@]}" tools/plot_grpo_ablation.py \
    --run "baseline=${OUT_ROOT}/baseline" \
    --run "dg_eta${DG_ETA_MAIN}=${OUT_ROOT}/dg_eta${DG_ETA_MAIN}" \
    --metric "validation/accuracy" \
    --metric "train/reward" \
    --metric "train/reward_mean" \
    --metric "train/reward_std" \
    --metric "train/loss" \
    --metric "train/truncation_rate" \
    --metric "train/natural_termination_rate" \
    --metric "train/dg_gate_mean" \
    --metric "train/dg_gate_spread" \
    --metric "train/dg_surprisal_mean" \
    --metric "train/dg_surprisal_max" \
    --metric "timing/train/total_step_time" \
    --out-dir "${PLOTS_DIR}"

  "${UV_RUN[@]}" tools/analyze_grpo_rollouts.py \
    --run "baseline=${OUT_ROOT}/baseline" \
    --run "dg_eta${DG_ETA_MAIN}=${OUT_ROOT}/dg_eta${DG_ETA_MAIN}" \
    --out-dir "${STABILITY_DIR}"
)

cat <<EOF
Long DG GSM8K comparison complete.
Artifacts:
  runs:      ${OUT_ROOT}
  plots:     ${PLOTS_DIR}
  stability: ${STABILITY_DIR}
EOF
