# Qwen3.6 NVFP4 + DFlash on DGX Spark

This branch adds a DGX Spark fast path for `Qwen3.6-35B-A3B-NVFP4` with the
`Qwen3.6-35B-A3B-DFlash` draft model. The best confirmed strict TG128 result is
`97.20 tok/s`, up from `70.59 tok/s` for the no-speculation NVFP4 baseline on
the same benchmark shape.

## Simple

Use this section if you just want to run the fast path, verify that it is on,
and understand the measured speedup.

### Result

| Item | Value |
| --- | --- |
| Model | `Qwen3.6-35B-A3B-NVFP4` |
| Draft model | `Qwen3.6-35B-A3B-DFlash` |
| Hardware | DGX Spark / GB10 |
| Active fast path | `VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1` |
| Runtime recipe | DFlash `num_speculative_tokens=15`, fp16 GDN SSM cache, CUDA graph disabled |
| Best strict TG128 mean | `97.20284519156971 tok/s` |
| Best strict TG128 values | `74.78382592759694`, `102.32446498702366`, `95.06421604875982`, `130.84421561667477`, `82.99750337779336` |
| Gain vs no-spec NVFP4 | `+26.61 tok/s` / `+37.7%` |
| Gain vs stable DFlash k=10 | `+12.42 tok/s` / `+14.6%` |
| Activation log | `Qwen GDN T16 commit1 unpaired Triton path active: rows=16 accepted=16 state_dtype=torch.float16` |

### Performance Progression

The table below is the cleanest local progression we have for strict TG128:
`pp=2048`, `tg=128`, `depth=0`, concurrency 1, generation latency, no prompt
cache. The first two older artifacts predate the explicit contract stamp, but
their JSON benchmark shape matches the same TG128 gate.

| Stage | What changed | Avg TG128 TPS | Gain vs previous | Gain vs no-spec |
| --- | --- | ---: | ---: | ---: |
| No-spec NVFP4 baseline | Target model only, no speculative draft | `70.59` | baseline | baseline |
| Add DFlash k=10 | DFlash proposes draft tokens and the NVFP4 target verifies them | `84.78` | `+14.19` / `+20.1%` | `+14.19` / `+20.1%` |
| Retune to DFlash k=15 | Wider draft budget gave more accepted tokens per target step on this workload | `91.84` | `+7.05` / `+8.3%` | `+21.24` / `+30.1%` |
| Add Qwen GDN T16 fast path | Fused fixed-shape verifier, fp16 GDN state cache, accepted-row commit | `97.20` | `+5.37` / `+5.8%` | `+26.61` / `+37.7%` |

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

Use this section if you want to understand what changed, which parts are
Qwen-specific, and which ideas should transfer to other model paths.

### Measurement Notes

The numeric gains above are cumulative measured steps, not isolated ablations
for every line of code. The fast verifier pieces are coupled: accepted-row
metadata, the fixed shape, the state-cache dtype, and the Triton kernel have to
agree before the path can activate correctly. Where a sub-change was only
measured as part of that bundle, the table says so directly.

| Artifact | Runs | Avg TG128 TPS | Notes |
| --- | ---: | ---: | --- |
| `qwen36_nvfp4_nospec_eager_tf5_tg128.json` | 10 | `70.59` | No speculative draft baseline. |
| `qwen36_nvfp4_dflash10_compile_cgmode0_clean_tf5_tg128.json` | 10 | `84.78` | Stable DFlash k=10 baseline used by the harness ratchet. |
| `qwen36_nvfp4_dflash15_eager_clean_current_tf5_tg128.json` | 5 | `91.84` | Clean DFlash k=15 recipe before the custom GDN fast path. |
| `qwen36_t16_commit1_unpaired_fp16_state_triton_live5_fixed_warm_tg128.json` | 5 | `97.20` | Final path after two warmup runs; activation log confirmed. |

### Optimization Impact

| Optimization | Measured TPS effect | What it does | Model-specific or extensible |
| --- | --- | --- | --- |
| DFlash speculative decoding | `70.59 -> 84.78` (`+14.19 tok/s`, `+20.1%`) | Adds the DFlash draft model so several candidate tokens can be proposed before the NVFP4 target verifies them. | Extensible to other models with a compatible draft model and speculative verifier; acceptance rate is model and workload specific. |
| DFlash k=15 budget | `84.78 -> 91.84` (`+7.05 tok/s`, `+8.3%`) | Raises the draft budget from the stable k=10 lane to k=15, which gave more useful accepted tokens per target pass on TG128. | Tuning pattern is extensible, but the winning width is model, prompt, sampling, and hardware dependent. |
| Fixed-shape Qwen GDN verifier bundle | `91.84 -> 97.20` (`+5.37 tok/s`, `+5.8%`) | Replaces the generic verifier path for the common DFlash shape with a guarded Triton path for 16 rows, 128-dim heads, and one speculative decode step. | Kernel is Qwen3.6/GDN/DFlash-specific as written; the idea transfers to other stable verifier shapes. |
| Accepted-row state commit | Included in the `+5.37 tok/s` fast-GDN bundle | Computes verifier outputs exactly, but only commits the accepted recurrent state row to the persistent cache. | Broad speculative-decoding idea for stateful layers; each model needs correct accepted-row metadata and cache layout. |
| Unpaired value-head layout | Included in the `+5.37 tok/s` fast-GDN bundle | Launches one value head per program with `block_v=8` instead of pairing sibling value heads in a larger program. | Mostly hardware and shape specific; retune for other head counts, state sizes, or GPUs. |
| fp16 GDN SSM cache | Included in the final `97.20 tok/s` recipe | Stores the recurrent GDN state cache in fp16 while keeping recurrence math in fp32. | Extensible to recurrent-state models that tolerate fp16 state cache precision; accuracy-test per model. |
| Cached decay constants | Included in the `+5.37 tok/s` fast-GDN bundle | Caches `-exp(A_log)` per layer and passes it into the Triton verifier instead of rebuilding static decay terms in the hot path. | Reusable for GDN/SSM-style layers with static decay parameters. |
| Accepted-token metadata plumbing | Required for the fast path to activate | Carries the accepted token count into GDN attention metadata so the kernel can commit the right state row directly. | General speculative-decoding plumbing for any stateful verifier. |
| Strict fallback guards | No TPS claim; protects unrelated paths | Activates the custom kernel only for the exact tested shape and falls back to stock vLLM otherwise. | General safety pattern for experimental kernels. |

### Why This Is Faster

The main cost was not draft generation by itself; it was target-side
verification of DFlash candidates through Qwen's GDN recurrent state. Wider
DFlash (`k=15`) helped because the target saw more accepted tokens per pass, but
branch verification was still expensive. The final improvement attacks that
target-side cost directly: a fixed-shape verifier avoids generic indexing,
writes only the accepted recurrent state row, uses a smaller fp16 state cache,
and removes repeated static setup.

The final path is intentionally narrow. If the request shape, dtype, row count,
or metadata do not match the tested DFlash verifier shape, vLLM falls back to
the stock implementation.

### Files

| File | Purpose |
| --- | --- |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | Qwen GDN T16 verifier kernel and guarded runtime path. |
| `vllm/v1/attention/backends/gdn_attn.py` | Accepted-token metadata needed to commit the right GDN state row. |

### Activation Checklist

| Check | Expected result |
| --- | --- |
| `VLLM_QWEN_GDN_T16_COMMIT1_UNPAIRED=1` | Environment flag is present in the container. |
| `--mamba-ssm-cache-dtype float16` | GDN state cache uses fp16. |
| `--speculative-config ... "num_speculative_tokens":15` | DFlash runs with the measured k=15 budget. |
| Docker log grep | `Qwen GDN T16 commit1 unpaired Triton path active: rows=16 accepted=16 state_dtype=torch.float16` |

If the activation log is missing, benchmark results should be treated as stock
fallback results, not as evidence for this fast path.
