#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/grpo_math_1B.yaml}"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-1}"

UV_RUN=(uv run)
if [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  UV_RUN+=(--no-sync)
fi
UV_RUN+=(python)

DG_ENABLED="${DG_ENABLED:-1}"
DG_ETA="${DG_ETA:-0.5}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
PROMPT_FILE="${PROMPT_FILE:-examples/prompts/cot.txt}"
CLUSTER_GPUS_PER_NODE="${CLUSTER_GPUS_PER_NODE:-1}"
CLUSTER_NUM_NODES="${CLUSTER_NUM_NODES:-1}"
DTENSOR_V2="${DTENSOR_V2:-0}"
CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-0}"
CHECKPOINT_KEEP_TOP_K="${CHECKPOINT_KEEP_TOP_K:-1}"
MODEL_SAVE_FORMAT="${MODEL_SAVE_FORMAT:-null}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-dg-grpo}"
WANDB_NAME="${WANDB_NAME:-}"

# "Full" here means "single A100 sized" rather than paper-scale.
MAX_STEPS="${MAX_STEPS:-128}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-4}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-8}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-32}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-1}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
VAL_PERIOD="${VAL_PERIOD:-16}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
SAVE_PERIOD="${SAVE_PERIOD:-1000000}"
MATH_NUM_WORKERS="${MATH_NUM_WORKERS:-4}"

if [[ "$DG_ENABLED" == "1" ]]; then
  RUN_FLAVOR="dg_eta${DG_ETA}"
else
  RUN_FLAVOR="baseline"
fi

RUN_NAME="${RUN_NAME:-gsm8k_${RUN_FLAVOR}_seed${SEED}}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/results/${RUN_NAME}}"
mkdir -p "$RUN_DIR"

CMD=(
  "${UV_RUN[@]}" examples/run_grpo.py
  "--config" "${CONFIG_PATH}"
  "logger.log_dir=${RUN_DIR}/logs"
  "checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints"
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}"
  "grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}"
  "grpo.val_period=${VAL_PERIOD}"
  "grpo.val_at_end=true"
  "grpo.max_val_samples=${MAX_VAL_SAMPLES}"
  "grpo.val_batch_size=${VAL_BATCH_SIZE}"
  "grpo.seed=${SEED}"
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
  "policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE}"
  "policy.logprob_batch_size=${LOGPROB_BATCH_SIZE}"
  "policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}"
  "policy.model_name=${MODEL_NAME}"
  "policy.tokenizer.name=${TOKENIZER_NAME}"
  "policy.generation.max_new_tokens=${MAX_NEW_TOKENS}"
  "policy.generation.vllm_cfg.enable_vllm_metrics_logger=false"
  "checkpointing.enabled=${CHECKPOINTING_ENABLED}"
  "checkpointing.keep_top_k=${CHECKPOINT_KEEP_TOP_K}"
  "checkpointing.save_period=${SAVE_PERIOD}"
  "data.train.split_validation_size=0.01"
  "data.num_workers=1"
  "env.math.num_workers=${MATH_NUM_WORKERS}"
  "logger.tensorboard_enabled=true"
  "logger.wandb_enabled=${WANDB_ENABLED}"
  "logger.mlflow_enabled=false"
  "logger.swanlab_enabled=false"
  "logger.monitor_gpus=true"
  "logger.num_val_samples_to_print=0"
  "data.default.prompt_file=${PROMPT_FILE}"
  "cluster.gpus_per_node=${CLUSTER_GPUS_PER_NODE}"
  "cluster.num_nodes=${CLUSTER_NUM_NODES}"
  "data.train.dataset_name=gsm8k"
  "+data.train.split=train"
  "++data.validation.dataset_name=gsm8k"
  "++data.validation.split=test"
  "loss_fn.dg_enabled=${DG_ENABLED}"
)

if [[ -n "$WANDB_NAME" ]]; then
  CMD+=("logger.wandb.name=${WANDB_NAME}")
fi
CMD+=("logger.wandb.project=${WANDB_PROJECT}")

if [[ "$DTENSOR_V2" == "0" ]]; then
  CMD+=("policy.dtensor_cfg._v2=false")
  CMD+=("checkpointing.model_save_format=${MODEL_SAVE_FORMAT}")
fi

if [[ "$DG_ENABLED" == "1" ]]; then
  CMD+=("loss_fn.dg_eta=${DG_ETA}")
fi

printf '%s\n' "${CMD[@]}" > "${RUN_DIR}/command.txt"

(
  cd "$ROOT_DIR"
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/stdout.log"
)
