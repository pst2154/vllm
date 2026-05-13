#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${PYTHON_BIN:=python3}"
: "${LLAMA_BENCHY_BIN:=/home/asteiner/Git_Repos/vllm/.venv/bin/llama-benchy}"
: "${QWEN36_PROGRESS_RUN_ID:=qwen36_progression_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${QWEN36_PROGRESS_CONCURRENCIES:=1 2 3 4 5 10}"
: "${QWEN36_PROGRESS_VARIANTS:=vanilla_nvfp4 mtp_nvfp4 dflash_nvfp4 optimized_nvfp4}"
: "${QWEN36_PROGRESS_MAX_NUM_SEQS:=10}"
: "${QWEN36_PROGRESS_RESULT_ROOT:=${EXPERIMENT_DIR}/metrics/serving/${QWEN36_PROGRESS_RUN_ID}}"
: "${QWEN36_PROGRESS_BASE_URL:=http://127.0.0.1:8000/v1}"
: "${QWEN36_PROGRESS_MODEL:=qwen3.6-35b}"
: "${QWEN36_PROGRESS_TOKENIZER:=/home/asteiner/models/Qwen3.6-35B-A3B-NVFP4}"
: "${QWEN36_PROGRESS_RUNS:=10}"
: "${QWEN36_PROGRESS_PREWARM_RUNS:=2}"

SERVE_SCRIPT="${SCRIPT_DIR}/serve_qwen36_nvfp4_container.sh"
EVAL_SCRIPT="${EXPERIMENT_DIR}/evals/050_serving_tg128_eval.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_qwen36_progression.py"
RAW_DIR="${QWEN36_PROGRESS_RESULT_ROOT}/raw"
LOG_DIR="${QWEN36_PROGRESS_RESULT_ROOT}/logs"

mkdir -p "${RAW_DIR}" "${LOG_DIR}"

current_container=""

cleanup() {
  if [[ -n "${current_container}" ]]; then
    QWEN36_CONTAINER_NAME="${current_container}" "${SERVE_SCRIPT}" stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

variant_label() {
  case "$1" in
    vanilla_nvfp4) echo "Vanilla NVFP4" ;;
    mtp_nvfp4) echo "Native MTP" ;;
    dflash_nvfp4) echo "DFlash k=15" ;;
    optimized_nvfp4) echo "DFlash k=15 + GDN T16" ;;
    *) echo "$1" ;;
  esac
}

variant_description() {
  case "$1" in
    vanilla_nvfp4)
      echo "Qwen3.6 NVFP4 target only, no speculative decoding"
      ;;
    mtp_nvfp4)
      echo "Qwen3.6 NVFP4 with native MTP speculative decoding, num_speculative_tokens=1"
      ;;
    dflash_nvfp4)
      echo "Qwen3.6 NVFP4 with DFlash draft model, num_speculative_tokens=15"
      ;;
    optimized_nvfp4)
      echo "Qwen3.6 NVFP4 with DFlash k=15, fp16 GDN SSM cache, and Qwen GDN T16 commit1 unpaired Triton path"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

start_variant() {
  local variant="$1"
  local container="qwen36-progress-${variant}"
  current_container="${container}"

  local -a envs=(
    "QWEN36_CONTAINER_NAME=${container}"
    "QWEN36_SERVED_MODEL_NAME=${QWEN36_PROGRESS_MODEL}"
    "QWEN36_MAX_NUM_SEQS=${QWEN36_PROGRESS_MAX_NUM_SEQS}"
    "QWEN36_COMPILATION_CONFIG_JSON={\"cudagraph_mode\":0}"
    "QWEN36_STOP_EXISTING_CONTAINERS=1"
    "QWEN36_STOP_EXISTING_VLLM=1"
  )

  case "${variant}" in
    vanilla_nvfp4)
      envs+=(
        "QWEN36_ENABLE_DFLASH=0"
        "QWEN36_SPECULATIVE_CONFIG_JSON="
      )
      ;;
    mtp_nvfp4)
      envs+=(
        "QWEN36_ENABLE_DFLASH=0"
        "QWEN36_SPECULATIVE_CONFIG_JSON={\"method\":\"mtp\",\"num_speculative_tokens\":1}"
      )
      ;;
    dflash_nvfp4)
      envs+=(
        "QWEN36_ENABLE_DFLASH=1"
        "QWEN36_SPECULATIVE_CONFIG_JSON="
        "QWEN36_NUM_SPECULATIVE_TOKENS=15"
      )
      ;;
    optimized_nvfp4)
      envs+=(
        "QWEN36_ENABLE_DFLASH=1"
        "QWEN36_SPECULATIVE_CONFIG_JSON="
        "QWEN36_NUM_SPECULATIVE_TOKENS=15"
        "QWEN36_MAMBA_SSM_CACHE_DTYPE=float16"
        "QWEN36_ENABLE_GDN_T16_COMMIT1_UNPAIRED=1"
      )
      ;;
    *)
      echo "unknown_variant=${variant}" >&2
      return 2
      ;;
  esac

  echo "start_variant=${variant}"
  echo "container=${container}"
  env "${envs[@]}" "${SERVE_SCRIPT}" start 2>&1 | tee "${LOG_DIR}/${variant}.serve.log"
}

run_eval() {
  local variant="$1"
  local concurrency="$2"
  local candidate="${QWEN36_PROGRESS_RUN_ID}_${variant}_c${concurrency}"
  local result_file="${RAW_DIR}/${variant}_c${concurrency}_tg128.json"
  local description
  description="$(variant_description "${variant}")"

  echo "run_eval=${candidate}"
  "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    --candidate-name "${candidate}" \
    --candidate-description "${description}" \
    --llama-benchy-bin "${LLAMA_BENCHY_BIN}" \
    --base-url "${QWEN36_PROGRESS_BASE_URL}" \
    --model "${QWEN36_PROGRESS_MODEL}" \
    --tokenizer "${QWEN36_PROGRESS_TOKENIZER}" \
    --prompt-tokens 2048 \
    --generate-tokens 128 \
    --depth 0 \
    --concurrency "${concurrency}" \
    --runs "${QWEN36_PROGRESS_RUNS}" \
    --prewarm-runs "${QWEN36_PROGRESS_PREWARM_RUNS}" \
    --target-tok-per-sec 0 \
    --save-result "${result_file}" \
    --no-internal-warmup \
    --skip-coherence \
    --no-adapt-prompt 2>&1 | tee "${LOG_DIR}/${variant}_c${concurrency}.eval.log"
}

write_manifest() {
  local manifest="${QWEN36_PROGRESS_RESULT_ROOT}/manifest.json"
  {
    printf '{\n'
    printf '  "run_id": "%s",\n' "${QWEN36_PROGRESS_RUN_ID}"
    printf '  "started_at_utc": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '  "prompt_tokens": 2048,\n'
    printf '  "generate_tokens": 128,\n'
    printf '  "depth": 0,\n'
    printf '  "prewarm_runs": %s,\n' "${QWEN36_PROGRESS_PREWARM_RUNS}"
    printf '  "measured_runs": %s,\n' "${QWEN36_PROGRESS_RUNS}"
    printf '  "concurrencies": "%s",\n' "${QWEN36_PROGRESS_CONCURRENCIES}"
    printf '  "variants": "%s",\n' "${QWEN36_PROGRESS_VARIANTS}"
    printf '  "max_num_seqs": %s,\n' "${QWEN36_PROGRESS_MAX_NUM_SEQS}"
    printf '  "llama_benchy_bin": "%s"\n' "${LLAMA_BENCHY_BIN}"
    printf '}\n'
  } > "${manifest}"
}

write_manifest

read -r -a variants <<< "${QWEN36_PROGRESS_VARIANTS}"
read -r -a concurrencies <<< "${QWEN36_PROGRESS_CONCURRENCIES}"

for variant in "${variants[@]}"; do
  start_variant "${variant}"
  for concurrency in "${concurrencies[@]}"; do
    run_eval "${variant}" "${concurrency}"
  done
  QWEN36_CONTAINER_NAME="${current_container}" "${SERVE_SCRIPT}" stop >/dev/null 2>&1 || true
  current_container=""
done

"${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
  --run-id "${QWEN36_PROGRESS_RUN_ID}" \
  --result-root "${QWEN36_PROGRESS_RESULT_ROOT}" \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${QWEN36_PROGRESS_RESULT_ROOT}"

echo "qwen36_progression_result_root=${QWEN36_PROGRESS_RESULT_ROOT}"
