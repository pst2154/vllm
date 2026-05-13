# Qwen3.6 NVFP4 + DFlash on DGX Spark

This branch adds the Qwen3.6 35B A3B NVFP4/DFlash fast path that reached the
best confirmed DGX Spark result.

## Simple

Use this section if you just want to run the fast path and verify that it is on.

### Result

| Item | Value |
| --- | --- |
| Model | `Qwen3.6-35B-A3B-NVFP4` |
| Draft model | `Qwen3.6-35B-A3B-DFlash` |
| Hardware | DGX Spark / GB10 |
| Active optimization | `VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1` |
| Runtime recipe | DFlash `num_speculative_tokens=15`, fp16 GDN SSM cache, CUDA graph disabled |
| Best 5-run TG128 mean | `97.20284519156971 tok/s` |
| Best 5-run values | `74.78382592759694`, `102.32446498702366`, `95.06421604875982`, `130.84421561667477`, `82.99750337779336` |
| Activation log | `Qwen GDN T16 commit1 unpaired Triton path active: rows=16 accepted=16 state_dtype=torch.float16` |

### Run

The tested setup used the Spark TF5 image and mounted this fork's edited GDN
files into the container. Set the paths for your machine first:

```bash
export VLLM_REPO=/home/asteiner/Git_Repos/vllm
export MODEL_ROOT=/home/asteiner/models
export MODS_DIR=/home/asteiner/Git_Repos/spark-vllm-docker/mods
```

Start the server:

```bash
docker run -d \
  --name qwen36-t16-commit1-unpaired \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MODEL_ROOT}:/models:ro" \
  -v "${MODS_DIR}:/workspace/mods:ro" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1 \
  -v "${VLLM_REPO}/vllm/model_executor/layers/mamba/gdn_linear_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py:ro" \
  -v "${VLLM_REPO}/vllm/v1/attention/backends/gdn_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py:ro" \
  ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest \
  bash -lc 'set -euo pipefail
cd /workspace/mods/fix-qwen3.5-autoround && bash run.sh
cd /workspace/mods/fix-qwen3-coder-next && bash run.sh
exec vllm serve /models/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b \
  --language-model-only \
  --max-model-len 262144 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8 \
  --dtype auto \
  --kv-cache-dtype auto \
  --load-format fastsafetensors \
  --attention-backend flash_attn \
  --enable-chunked-prefill \
  --trust-remote-code \
  --quantization compressed-tensors \
  --enable-prefix-caching \
  --moe-backend flashinfer_cutlass \
  --mamba-ssm-cache-dtype float16 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --optimization-level 3 \
  --performance-mode throughput \
  --default-chat-template-kwargs "{\"preserve_thinking\":true}" \
  --override-generation-config "{\"temperature\":0.5,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0}" \
  --disable-uvicorn-access-log \
  --no-enable-log-requests \
  --compilation-config "{\"cudagraph_mode\":0}" \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"/models/Qwen3.6-35B-A3B-DFlash\",\"num_speculative_tokens\":15}"'
```

Wait for health:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Check that the fast path activated:

```bash
docker logs qwen36-t16-commit1-unpaired 2>&1 \
  | grep 'Qwen GDN T16 commit1 unpaired Triton path active'
```

### Benchmark

Use the same TG128 gate: `pp=2048`, `tg=128`, `depth=0`, concurrency 1,
generation latency, no prompt cache.

Run two warmup passes, then five measured runs:

```bash
BENCH=/home/asteiner/Git_Repos/vllm/.venv/bin/llama-benchy
TOKENIZER=/home/asteiner/models/Qwen3.6-35B-A3B-NVFP4

"${BENCH}" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b \
  --served-model-name qwen3.6-35b \
  --tokenizer "${TOKENIZER}" \
  --pp 2048 \
  --tg 128 \
  --depth 0 \
  --runs 2 \
  --no-cache \
  --latency-mode generation \
  --save-result qwen36_t16_warmup_tg128.json \
  --format json

"${BENCH}" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b \
  --served-model-name qwen3.6-35b \
  --tokenizer "${TOKENIZER}" \
  --pp 2048 \
  --tg 128 \
  --depth 0 \
  --runs 5 \
  --no-cache \
  --latency-mode generation \
  --save-result qwen36_t16_live5_tg128.json \
  --format json
```

## Details

Use this section if you want to understand what changed and why it improves
throughput.

### Optimizations

| Optimization | What changed | Why it is faster |
| --- | --- | --- |
| Fixed-shape GDN verifier | Added `_qwen_gdn_t16_commit1_unpaired_kernel` for the common DFlash k=15 verifier shape: 16 rows, 128-dim heads, one speculative decode. | Avoids the generic indexed GDN replay path for the hot decode case and runs the verifier recurrence as one compact Triton launch. |
| Commit only the accepted state | The kernel still computes every verifier output token, but only writes the accepted recurrent state row back to the cache. | Rejected speculative rows do not need persistent GDN state, so this cuts state-cache write traffic. |
| Unpaired value-head layout | Launches one value head per program with `block_v=8` instead of pairing sibling value heads in one larger program. | Uses fewer registers per program on GB10, which was faster than sharing q/k work across sibling value heads. |
| fp16 GDN cache | Serve with `--mamba-ssm-cache-dtype float16`; the kernel keeps recurrence math in fp32 and writes the final cache row in fp16. | Reduces memory bandwidth for the 128x128 recurrent state cache without changing the target verification rule. |
| Cached decay constants | Caches `-exp(A_log)` once per layer and passes it to the Triton kernel. | Removes static per-head decay setup from the hot verifier path. |
| Accepted-token metadata | `gdn_attn.py` carries the accepted token count into GDN metadata. | Lets the fast path choose the right state row directly, without an extra sync or guesswork. |
| Strict fallback guards | The fast path only activates for the exact tested shape; all other requests use stock vLLM behavior. | Keeps the speedup narrow and safe instead of adding overhead or behavior changes to unrelated paths. |

### Files

| File | Purpose |
| --- | --- |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | Qwen GDN T16 verifier kernel and guarded runtime path. |
| `vllm/v1/attention/backends/gdn_attn.py` | Accepted-token metadata needed to commit the right GDN state row. |
