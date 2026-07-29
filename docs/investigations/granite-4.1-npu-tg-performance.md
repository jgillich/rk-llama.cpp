# Investigation: Granite 4.1 NPU text-generation performance

**Date:** 2026-07-29
**Status:** Concluded — no code fix warranted
**Conclusion:** The NPU is utilized correctly for Granite 4.1-8B. The ~1.5 tok/s TG rate and high CPU usage are inherent to running a dense full-attention 8B model on the RK3588 NPU, not a misutilization bug. TG on CPU-REPACK is ~1.6× faster than on the NPU at single-token decode; the NPU's only advantage for this model is at prompt processing (large batch).

---

## Problem statement

Granite 4.1-8B (Q6_K) inference on the RK3588 NPU runs at <2 tok/s with high CPU utilization, despite Granite 4.0 being listed as supported. The suspicion was that the NPU was not properly engaged.

## Methodology

Added debug logging (gated behind `RKNPU_DBG=1`) to four sites:
- `src/llama-model.cpp` — `make_cpu_buft_list` (buft ordering)
- `src/llama-model-loader.cpp` — `select_weight_buft` (per-tensor buft selection)
- `ggml/src/ggml-rknpu2/ggml-rknpu2.cpp` — `ggml_backend_rknpu_device_supports_op` (rejection reasons) and `ggml_backend_rknpu_graph_compute` (runtime matmul execution)

Three runs were captured:
- **run1** — default Q6_K (cyclic `W8A8_STANDARD` / `W4A4_HADAMARD`). 1.55 tok/s.
- **run2** — `RKNPU_HYBRID="W8A8_STANDARD"` (quotes mistakenly included in the value). ~2.5 tok/s.
- **run3** — `RKNPU_HYBRID=W8A8_STANDARD` (quotes fixed). 1.2 tok/s, degrading to ~1.0 over a long generation.

The instrumentation has since been reverted; the working tree is clean.

## Findings

### The NPU is engaged and working in the default configuration

Run1 logs confirm correct offload:
- `RKNPU model buffer size = 6232.00 MiB` — weights loaded onto the NPU.
- All 7 matmuls/layer × 40 layers accepted by the NPU's `supports_op` (e.g. `blk.0.attn_q.weight` → `ACCEPT pipeline=W4A4_HADAMARD`).
- 6368 matmuls executed on the NPU during generation (`graph_compute: RUN`), zero skips, zero errors.
- `graph splits = 403` (one split per CPU↔NPU boundary in the dense attention topology).

There is no offload failure. The original "NPU not utilized" hypothesis was wrong.

### The RKNPU backend only accelerates MUL_MAT

`ggml_backend_rknpu_device_supports_op` returns `true` only for `GGML_OP_MUL_MAT` and `GGML_OP_NONE`; every other op falls through to `default: return false`. Consequently all non-matmul ops — `rms_norm`, `rope_ext`, `flash_attn_ext`, `soft_max_ext`, `silu`, elementwise `mul`/`add`/`scale`, `reshape`, `get_rows` — run on the CPU. This is structural, not a bug.

### Run2's "improvement" was a quoting accident, not a real win

`RKNPU_HYBRID="W8A8_STANDARD"` (with literal `"` characters in the env value) produced:
```
resolve_op: name=blk.0.attn_q.weight ... pattern_idx=0/1 -> "W8A8_STANDARD" (MISSING)
supports_op:   REJECT no_pipeline
select_buft:   try dev=RKNPU buft=RKNPU -> reject
select_buft:   try dev=CPU buft=CPU_REPACK -> ACCEPT
```
The NPU rejected every matmul (pipeline name mismatch) and the whole model fell back to CPU-REPACK. The ~2.5 tok/s was pure-CPU performance — the NPU was bypassed entirely. `RKNPU model buffer` was absent from the load log, and `graph_compute: RUN` count was 0.

### Run3 (real W8A8 on the NPU) is slower than the default

With quotes fixed, the NPU accepted all matmuls (`W8A8_STANDARD (found)`, `RKNPU model buffer = 7992 MiB`, 12691 NPU matmul executions). Result: 1.2 tok/s — **slower** than run1's 1.55. Forcing W8A8 doubled the weight bytes vs the default's half-W4A4 layers and did not reduce the CPU-attention bottleneck, confirming that decode bandwidth on the weights is not the dominant limiter — CPU-attention contention with the NPU pipeline is.

## Root cause of the low TG rate

Granite 4.1-8B is a **dense full-attention** model (`LLM_ARCH_GRANITE`, `model_type: granite` in the GGUF). All 40 layers use full attention. At single-token decode (M=1):

1. **Every layer interleaves ~10 CPU ops with the 7 NPU matmuls**, forcing 403 graph splits. Each split is a CPU↔NPU sync boundary that serializes execution.
2. **The CPU-side `flash_attn_ext` cost grows linearly with context length** across all 40 layers. At S≈640 the CPU attention is ~1.9 GFlop/token; at S=8192 it is ~24 GFlop/token. This growing work runs on the same 4 CPU cores that feed the NPU, stalling the NPU pipeline and producing the observed degradation (1.5 → 1.0 tok/s over a long generation).
3. **The NPU's M=1 matmuls are themselves slower than CPU-REPACK GEMV** for this model, because the per-matmul launch + IOMMU sync overhead (~433 ms/token aggregate across 280 matmuls) exceeds the compute saved vs the A76's repacked Q6_K GEMV path.

## Why Qwen3.5 9B Q8_0 reaches 3.2 tok/s on the same NPU

Qwen3.5 (`LLM_ARCH_QWEN35`) is a **hybrid linear-attention** model: `full_attn_interval = 4` means 3 of every 4 layers are recurrent (gated delta net, O(1) per token at decode) and only 1 in 4 uses full attention. For a 32-layer 9B model that is 8 full-attention layers vs 24 recurrent layers. Consequences:

- **5× less CPU attention work per token** at S≈640 (~0.4 GFlop vs Granite's ~1.9 GFlop).
- **The recurrent layers' matmuls offload cleanly** to the NPU with cheap, non-growing CPU state ops, producing far less split/serialization overhead.
- **No KV-cache growth on 75% of layers**, so the model degrades far more slowly with context.

The README's Qwen3.5 9B Q8_0 NPU TG of 3.2 tok/s vs Granite 4.1-8B's ~1.5 is therefore an architecture difference, not an NPU-utilization difference. The NPU is doing the same job in both cases; Qwen3.5's architecture simply produces far less CPU work to interleave with it.

### Methodology caveat

The README's TG numbers come from `llama-bench`'s `tg128` test, which times 128 single-token decodes starting from a 512-token prompt (context grows 512→639) and reports the average. This is a short, low-context window. Granite's dense-attention penalty grows with context, so its real-world TG in a long conversation is below the `tg128` figure — which matches the degradation observed in run3.

## Quantitative comparison of options

| Option | TG (short ctx) | TG (8k ctx) | PP | Cost |
|---|---|---|---|---|
| Current (all-NPU, default Q6_K) | 1.55 tok/s | ~0.67 tok/s | ~45 tok/s | none |
| All CPU-REPACK (`RKNPU_HYBRID=CPU_STANDARD`) | 2.5 tok/s | ~1.1 tok/s | ~7 tok/s | throws away ~6× PP |
| Hybrid (NPU for PP, CPU for TG) | 2.45 tok/s | ~1.1 tok/s | ~45 tok/s | non-trivial impl; backend has no phase toggle today |
| Q4_0 quant (W4A4, all-NPU) | ~3 tok/s (est.) | — | — | accuracy loss; untested for Granite 4.1 |
| Switch to Qwen3.5 9B (hybrid arch) | 3.2 tok/s | high | 44.6 tok/s | different model |

The hybrid option (NPU-PP / CPU-TG) buys ~+60% TG with no PP loss, but the absolute TG is still only ~2.5 tok/s — below the threshold where the NPU adds value over a plain CPU-REPACK build for interactive workloads. As the user concluded: if NPU TG is slower than CPU, the NPU is not worth it for this model class.

## Conclusion

There is no bug to fix in the RKNPU2 backend, the Granite graph builder, or the chat template. The NPU is correctly offloading all eligible matmuls. The performance characteristics are inherent to:

- **The model architecture** (dense full-attention, all 40 layers grow with context and interleave CPU ops with NPU matmuls), and
- **The NPU backend's scope** (only `MUL_MAT` is accelerated; everything else is CPU, and at M=1 the CPU-REPACK GEMV path beats the NPU's launch+sync overhead).

The NPU is a good fit for hybrid/linear-attention models (Qwen3.5, LFM2, Granite-Hybrid) and for prompt-processing of any model, but it is not a good fit for single-token decode of dense-attention models like Granite 4.1-8B.

## Supporting evidence

- `run1.log` — default Q6_K NPU run (91k lines). `make_cpu_buft_list` shows buft order `[RKNPU, CPU_REPACK, CPU]`; `select_buft` shows NPU accepting all Q6_K weights with `W4A4_HADAMARD`/`W8A8_STANDARD`; `graph_compute` shows 6368 NPU matmul executions.
- `run2.log` — misquoted env var run (6.6k lines). `resolve_op` shows `"W8A8_STANDARD" (MISSING)` for every tensor; NPU buffer absent; 0 NPU matmul runs; weights fell to CPU_REPACK/CPU.
- `run3.log` — corrected W8A8 NPU run (134k lines). `resolve_op` shows `W8A8_STANDARD (found)`; NPU buffer 7992 MiB; 12691 NPU matmul executions; 1.2 tok/s, slower than the default 1.55.
- `ggml/src/ggml-rknpu2/ggml-rknpu2.cpp:1275-1323` — `supports_op` only accepts `MUL_MAT`/`NONE`.
- `ggml/src/ggml-rknpu2/rknpu2-configuration.cpp:52-69` — `get_active_pattern` returns `nullptr` for unregistered tensor types even when `RKNPU_HYBRID` is set; `W8A8_STANDARD` has `n_align=32` (looser than `W4A4_HADAMARD`'s 64).
- `src/llama-model.cpp:188-196` — the NPU (registered as `ACCEL`) is added first to the CPU buft list, ahead of REPACK.
- `src/models/granite.cpp` — dense Granite graph builder: 7 NPU matmuls + ~10 CPU ops per layer, all 40 layers full attention.
- `src/models/qwen35.cpp:198-287` and `src/llama-model.cpp:2443-2470` — Qwen3.5 hybrid builder: `full_attn_interval=4`, 3/4 layers recurrent (gated delta net, O(1) at decode).
- `tools/llama-bench/llama-bench.cpp:2026-2045, 2279-2295` — `tg128` times a 128-token generation loop as a single sample, averaging over a short context window.
- `ggml/src/ggml-rknpu2/README.md:50-73, 94-105` — benchmark table and methodology.

## Open questions (for future work)

1. **Phase-aware hybrid backend (NPU-PP / CPU-TG).** The current backend routes per-layer via the cyclic `RKNPU_HYBRID` pattern, but cannot switch backend between PP (large M, want NPU) and TG (M=1, want CPU) within one context. ~+60% TG with no PP loss is the best available lever for dense models.

   The non-obvious implementation constraint: `supports_op` is invoked at *model load* (via `select_weight_buft` → `weight_buft_supported`) to choose the buffer type each weight lands in — a one-time, irreversible decision. Weights get requantized into NPU-native INT8/INT4 and live in NPU-backed memory that the CPU-REPACK GEMV cannot read. So a runtime `supports_op` batch-size gate does **not** work; the real design requires keeping two copies of every weight (NPU-native + CPU-REPACK), roughly doubling memory (~12 GiB for an 8B Q6_K, tight on RK3588's 16 GiB). The on-the-fly dequant-requant alternative re-runs the NPU calibration pipeline per prompt, which is not a small cost. Worth revisiting only if a dense model must be served; for hybrid models (Qwen3.5, LFM2, Granite-Hybrid) the NPU already wins both phases.

2. **W8A8_HADAMARD for the W8A8 layers (accuracy, not speed).** The README's pipeline table shows `W8A8_HADAMARD` (perplexity 20.85) substantially beats `W8A8_STANDARD` (22.50) on Granite-350M, at negligible M=1 cost (a 4096-vector Hadamard transform is ~0.01 ms). For the default Q6_K pattern, switching the W8A8 layers to HADAMARD — `RKNPU_HYBRID="W8A8_HADAMARD,W4A4_HADAMARD"` — is a free accuracy improvement. Orthogonal to the speed investigation; does not help TG.

3. **Q4_0 / all-W4A4 is likely not viable for Granite.** The README's W4A4_HADAMARD perplexity on Granite-350M is 86.16 — catastrophic (4× worse than F16's 20.74, "broken model" territory). Extrapolating the Qwen3.5 Q4_0 speedup (3.2→4.4, 1.375×) to Granite's 1.55 gives only ~2.1 tok/s at a ruinous accuracy cost. Not worth testing for a chat model.

4. **Long-context TG benchmarking.** `tg128` underreports the dense-attention penalty. A `tg4096` or context-sweep benchmark would surface the degradation that real conversations experience.

## Optimization ideas evaluated and rejected

- **Q/K/V weight fusion** (concatenate wq/wk/wv into one matmul to reduce NPU launch count): the ~15-20% TG claim assumes a ~1.55 ms per-launch sync, which is unsupported. Realistic IOMMU fence + ioctl is 0.1-0.5 ms, giving 1-5% at most. The fused matmul does identical compute (N grows 3×), so only the sync count drops. Not worth the graph-builder surgery and loss of per-projection bias handling.
- **Async NPU execution with CPU overlap**: directionally correct (CPU is idle during NPU compute and vice versa), but the RKNPU backend's `graph_compute` only sees MUL_MAT nodes — the interleaving CPU ops (norm, rope, attention) are dispatched by a different backend in separate splits. Real cross-backend pipelining requires scheduler-level work, not a backend-local change. A large project, not the near-term win it's sometimes presented as.
- **Op fusion to reduce graph splits**: correct that fewer splits = fewer syncs, but every relevant fusion (`rms_norm`+`mul_mat`, SwiGLU) is upstream llama.cpp graph optimization, not RKNPU2-backend work. The growing cost is `flash_attn_ext` on the attention layers, which cannot be fused away. Marginal and out of this fork's scope.
- **NPU multi-core utilization at M=1**: already realized. `compute_n_segments` (`ggml-rknpu2.cpp:216`) partitions the N dimension across all 3 NPU cores, `rknn_matmul_set_core_mask` (`:383-391`) pins each segment to a distinct core, and the runs are dispatched in parallel via `#pragma omp parallel for` (`:735`). Run3 logs show `n_segs=3` for every matmul. There is no idle-core headroom to recover; the "2-3× via multi-core" suggestion is moot.
- **CPU attention kernel NEON/dot-product optimization**: the flash-attention QK^T dot product already dispatches through `ggml_get_type_traits_cpu(k->type)->vec_dot` (`ops.cpp:8237`), which for the F16 KV cache uses `ggml_vec_dot_f16` — already heavily SIMD-optimized (SVE or 8-way unrolled NEON FMA, `vec.cpp:264`). The A76's NEON FP16-FMA path is exercised. No evidence of a 2-3× missed-optimization gap; the kernel is not naively scalar.