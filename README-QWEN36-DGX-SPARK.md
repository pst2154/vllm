# Qwen3.6 NVFP4 + DFlash on DGX Spark

This branch adds a DGX Spark fast path for `Qwen3.6-35B-A3B-NVFP4` with the
`Qwen3.6-35B-A3B-DFlash` draft model. The headline single-user strict TG128
result is `97.20 tok/s`, up from `70.59 tok/s` for the no-speculation NVFP4
baseline on the same benchmark shape. The checked-in sweep reports each
concurrency separately; TPS is not averaged across c1/c2/c3/c4/c5/c10.

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
| Gain vs no-spec NVFP4 | `+26.61 tok/s (+37.7%)` on the strict c1 lineage |
| Gain vs stable DFlash k=10 | `+12.42 tok/s (+14.6%)` on the strict c1 lineage |
| Activation log | `Qwen GDN T16 commit1 unpaired Triton path active: rows=16 accepted=16 state_dtype=torch.float16` |

### Performance Progression

The table below is the cleanest local progression we have for strict TG128:
`pp=2048`, `tg=128`, `depth=0`, concurrency 1, generation latency, no prompt
cache. The first two older artifacts predate the explicit contract stamp, but
their JSON benchmark shape matches the same TG128 gate.

| Stage | What changed | Mean TG128 TPS | Gain vs previous | Gain vs no-spec |
| --- | --- | ---: | ---: | ---: |
| No-spec NVFP4 baseline | Target model only, no speculative draft | `70.59` | baseline | baseline |
| Add DFlash k=10 | DFlash proposes draft tokens and the NVFP4 target verifies them | `84.78` | `+14.19 tok/s (+20.1%)` | `+14.19 tok/s (+20.1%)` |
| Retune to DFlash k=15 | Wider draft budget gave more accepted tokens per target step on this workload | `91.84` | `+7.05 tok/s (+8.3%)` | `+21.24 tok/s (+30.1%)` |
| Add Qwen GDN T16 fast path | Fused fixed-shape verifier, fp16 GDN state cache, accepted-row commit | `97.20` | `+5.37 tok/s (+5.8%)` | `+26.61 tok/s (+37.7%)` |

### Concurrency Sweep

The requested 2-warmup/10-measured TG128 sweep across concurrencies
`1,2,3,4,5,10` is checked in at
[`benchmarks/qwen36-dgx-spark/progression/qwen36_progression_20260513T152918Z/qwen36_progression_tg128.md`](benchmarks/qwen36-dgx-spark/progression/qwen36_progression_20260513T152918Z/qwen36_progression_tg128.md).

That artifact includes the SVG chart, CSV, raw measured JSON files, prewarm
JSON files, eval logs, and optimized-path activation evidence. Each value below
is the mean TPS of 10 measured TG128 requests at that concurrency. We do not
average these columns together.

![TG128 throughput by concurrency](benchmarks/qwen36-dgx-spark/progression/qwen36_progression_20260513T152918Z/qwen36_progression_tg128.svg)

| Variant | c1 | c2 | c3 | c4 | c5 | c10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla NVFP4 | `30.99` | `63.31` | `93.84` | `122.26` | `149.28` | `240.34` |
| Native MTP | `44.52` | `86.74` | `125.12` | `159.37` | `182.23` | `277.26` |
| DFlash k=15 | `78.91` | `109.28` | `142.44` | `165.51` | `175.99` | `231.54` |
| DFlash k=15 + GDN T16 | `89.60` | `119.66` | `147.36` | `170.57` | `197.82` | `262.97` |

The c1 slice from that exact sweep shows the single-request comparisons:

| Variant | c1 mean TG128 TPS | Attributed comparison | Delta tok/s | Delta % |
| --- | ---: | --- | ---: | ---: |
| Vanilla NVFP4 | `30.99` | Baseline | baseline | baseline |
| Native MTP | `44.52` | MTP recipe vs vanilla | `+13.53 tok/s` | `+43.7%` |
| DFlash k=15 | `78.91` | DFlash recipe vs vanilla | `+47.92 tok/s` | `+154.6%` |
| DFlash k=15 + GDN T16 | `89.60` | GDN T16 bundle vs DFlash k=15 | `+10.69 tok/s` | `+13.5%` |
| DFlash k=15 + GDN T16 | `89.60` | Final recipe vs vanilla | `+58.60 tok/s` | `+189.1%` |

The columns tell different stories. The optimized GDN path wins c1-c5 in this
sweep and recovers much of DFlash's c10 loss, while native MTP remains the
highest c10 result.

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

The strict c1 lineage above is cumulative for the DFlash path. The concurrency
sweep compares recipes: vanilla NVFP4, native MTP, DFlash k=15, and DFlash k=15
with the GDN T16 verifier bundle. MTP and DFlash are alternative speculative
backends, so DFlash should be compared to vanilla, not treated as an incremental
change on top of MTP.

The cleanest attribution in the sweep is the optimized recipe versus DFlash k=15,
because both use the same DFlash draft width and differ by the GDN verifier
bundle. The pieces inside that bundle were not individually ablated; they should
be credited as required parts of the measured bundle, not as standalone TPS
claims.

| Artifact | Runs | Mean TG128 TPS | Notes |
| --- | ---: | ---: | --- |
| `qwen36_nvfp4_nospec_eager_tf5_tg128.json` | 10 | `70.59` | No speculative draft baseline. |
| `qwen36_nvfp4_dflash10_compile_cgmode0_clean_tf5_tg128.json` | 10 | `84.78` | Stable DFlash k=10 baseline used by the harness ratchet. |
| `qwen36_nvfp4_dflash15_eager_clean_current_tf5_tg128.json` | 5 | `91.84` | Clean DFlash k=15 recipe before the custom GDN fast path. |
| `qwen36_t16_commit1_unpaired_fp16_state_triton_live5_fixed_warm_tg128.json` | 5 | `97.20` | Final path after two warmup runs; activation log confirmed. |

### Optimization Impact

These are c1 comparisons from the exact sweep. MTP and DFlash are separate
recipes, not stacked changes.

| Comparison | Before TPS | After TPS | Change tok/s | Change % | What changed | Scope |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| MTP vs vanilla | `30.99` | `44.52` | `+13.53` | `+43.7%` | Turned on native MTP speculative decoding. | Works for models with supported MTP heads. |
| DFlash k=15 vs vanilla | `30.99` | `78.91` | `+47.92` | `+154.6%` | Used the DFlash draft model with `num_speculative_tokens=15`. | Needs a compatible draft model; tune k per model. |
| GDN T16 bundle vs DFlash k=15 | `78.91` | `89.60` | `+10.69` | `+13.5%` | Added the Qwen GDN T16 verifier bundle: fp16 state cache, accepted-row commit, cached decay constants, and the Triton verifier path. | Qwen/GDN-specific as written; the fixed-shape verifier idea can transfer. |
| Final optimized recipe vs vanilla | `30.99` | `89.60` | `+58.60` | `+189.1%` | DFlash k=15 plus the GDN T16 verifier bundle. | Final tested Qwen Spark recipe. |

The GDN T16 sub-pieces were not measured one by one. They should be credited as
one bundle because the fast path only works when the metadata, cache dtype,
state commit, shape guards, and Triton kernel agree.

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
