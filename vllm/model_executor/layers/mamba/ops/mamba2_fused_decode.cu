// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused Mamba-2 decode chain (VLLM_MAMBA2_FUSED_DECODE) for MambaMixer2 with the per-rank shapes
// head_dim 64, d_state 128, 32 heads, 2 groups, conv width 4 (e.g. Nemotron-H at TP4): causal-conv1d update -> selective state update (FlashInfer arithmetic incl.
// Philox stochastic rounding of the fp16 SSM state) -> gated RMSNorm (Inductor arithmetic) -> static FP8 quant
// of the out_proj input, in ONE kernel per Mamba layer.
//
// Exactness contract (checked bitwise by tests/kernels/mamba/test_mamba2_fused_decode.py against the stock kernels):
//  * conv part reproduces vLLM's Triton `_causal_conv1d_update_kernel` instruction by instruction (see tr_* helpers),
//  * SSU part is FlashInfer's `selective_state_update_kernel_simple` (single token) and
//    `selective_state_update_kernel_simple_mtp` (varlen MTP verify) per-row code, copied verbatim (same
//    lane -> element mapping, same reduction order, same Philox counter, same cvt.rs), compiled with FlashInfer's
//    JIT flags (-O3 -use_fast_math, sm_100a),
//  * norm part reproduces the Inductor kernel `triton_per_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_
//    silu_view_0` (same per-lane element ownership, same in-thread accumulation order, same butterfly).
//
// Parallel decomposition: one thread-block CLUSTER per (token or sequence, norm group). A norm group is 16 heads
// (1024 channels) on this model; the cluster has 16/HPC CTAs, each owning HPC heads. The B/C channels of the group
// (2 x 128) are convolved by their owner CTA only (each conv channel and its conv-state column has exactly one owning
// thread), shared through distributed shared memory after a cluster barrier (release/acquire), and the 1024 norm
// inputs of the group are exchanged the same way. All hand-offs: barrier.cluster.arrive.release /
// barrier.cluster.wait.acquire (+ __syncthreads inside a CTA). Every SSM-state element and every conv-state element
// is read and written by one thread only (reads before writes in program order).
// PDL: only model parameters (never written during a forward) are read before griddepcontrol.wait.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "flashinfer/mamba/common.cuh"  // FlashInfer's installed headers (flashinfer/data/include)
#include "flashinfer/mamba/conversion.cuh"
#include "flashinfer/mamba/ssu_mtp_common.cuh"

namespace cg = cooperative_groups;

namespace vllm_mamba2fd {

using namespace flashinfer::mamba;
using namespace flashinfer::mamba::conversion;
using bf16 = __nv_bfloat16;

// ---- model shapes (per TP rank, TP4) ----
constexpr int DIM = 64;         // head_dim
constexpr int DSTATE = 128;     // ssm_state_size
constexpr int HPG = 16;         // heads per norm group (= heads per B/C group here)
constexpr int GSIZE = HPG * DIM;  // 1024: gated-RMSNorm group size
constexpr int XDIM = 2048;      // per-rank intermediate (x / gate / ssm_out width)
constexpr int XBC_OFF = 2048;   // offset of xBC inside projected_states
constexpr int B_OFF = 2048;     // offset of B inside xBC
constexpr int C_OFF = 2304;     // offset of C inside xBC
constexpr int DT_OFF = 4608;    // offset of dt inside projected_states
constexpr int KW = 4;           // conv kernel width

// ---- argument block (flat int64 array, same layout in mambaopt/kernels.py) ----
enum Arg : int {
  A_PROJ = 0, A_PROJ_S, A_CONV_W, A_W_SD, A_W_SW, A_CONV_B,
  A_CS, A_CS_SEQ, A_CS_DIM, A_CS_TOK,
  A_SIDX, A_SIDX_S0, A_SIDX_S1, A_DIDX, A_DIDX_S0, A_DIDX_S1,
  A_CU, A_NACC,
  A_SSM, A_SSM_S,
  A_A, A_D, A_DTB, A_SEED,
  A_NW, A_SCALE, A_OUT, A_OUT_S, A_Y, A_Y_S,
  A_NSEQ, A_PAD, A_NT, A_NARGS
};

struct Params {
  bf16* proj; int64_t proj_s;
  bf16 const* conv_w; int64_t w_sd, w_sw;
  bf16 const* conv_b;
  bf16* cs; int64_t cs_seq, cs_dim, cs_tok;
  int32_t const* sidx; int64_t sidx_s0, sidx_s1;
  int32_t const* didx; int64_t didx_s0, didx_s1;
  int32_t const* cu; int32_t const* nacc;
  __half* ssm; int64_t ssm_s;
  float const* A; bf16 const* D; bf16 const* dtb; int64_t const* seed;
  bf16 const* nw; float const* scale;
  __nv_fp8_e4m3* out; int64_t out_s;
  bf16* y; int64_t y_s;
  int nseq; int pad; int nt;
};

static Params unpack(int64_t const* a) {
  Params p;
  p.proj = reinterpret_cast<bf16*>(a[A_PROJ]); p.proj_s = a[A_PROJ_S];
  p.conv_w = reinterpret_cast<bf16 const*>(a[A_CONV_W]); p.w_sd = a[A_W_SD]; p.w_sw = a[A_W_SW];
  p.conv_b = reinterpret_cast<bf16 const*>(a[A_CONV_B]);
  p.cs = reinterpret_cast<bf16*>(a[A_CS]); p.cs_seq = a[A_CS_SEQ]; p.cs_dim = a[A_CS_DIM]; p.cs_tok = a[A_CS_TOK];
  p.sidx = reinterpret_cast<int32_t const*>(a[A_SIDX]); p.sidx_s0 = a[A_SIDX_S0]; p.sidx_s1 = a[A_SIDX_S1];
  p.didx = reinterpret_cast<int32_t const*>(a[A_DIDX]); p.didx_s0 = a[A_DIDX_S0]; p.didx_s1 = a[A_DIDX_S1];
  p.cu = reinterpret_cast<int32_t const*>(a[A_CU]); p.nacc = reinterpret_cast<int32_t const*>(a[A_NACC]);
  p.ssm = reinterpret_cast<__half*>(a[A_SSM]); p.ssm_s = a[A_SSM_S];
  p.A = reinterpret_cast<float const*>(a[A_A]); p.D = reinterpret_cast<bf16 const*>(a[A_D]);
  p.dtb = reinterpret_cast<bf16 const*>(a[A_DTB]); p.seed = reinterpret_cast<int64_t const*>(a[A_SEED]);
  p.nw = reinterpret_cast<bf16 const*>(a[A_NW]); p.scale = reinterpret_cast<float const*>(a[A_SCALE]);
  p.out = reinterpret_cast<__nv_fp8_e4m3*>(a[A_OUT]); p.out_s = a[A_OUT_S];
  p.y = reinterpret_cast<bf16*>(a[A_Y]); p.y_s = a[A_Y_S];
  p.nseq = (int)a[A_NSEQ]; p.pad = (int)a[A_PAD]; p.nt = (int)a[A_NT];
  return p;
}

// =====================================================================================================
// Triton-equivalent scalar ops (explicit PTX so that neither nvcc nor ptxas re-associates / contracts them;
// each mirrors the instruction Triton/Inductor emitted for the stock kernel, see NOTES "instruction audit").
// =====================================================================================================
__device__ __forceinline__ float tr_mul(float a, float b) {
  float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float tr_add(float a, float b) {
  float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float tr_fma(float a, float b, float c) {
  float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float tr_ex2(float a) {
  float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
__device__ __forceinline__ float tr_div_full(float a, float b) {
  float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float tr_div_rn(float a, float b) {
  float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float tr_rsqrt(float a) {  // libdevice __nv_rsqrtf as compiled by Triton
  float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
__device__ __forceinline__ float bf_mul(float a, float b) {  // Triton bf16 x bf16 -> bf16 (mul.bf16x2, rn)
  unsigned short ra = __bfloat16_as_ushort(__float2bfloat16_rn(a)), rb = __bfloat16_as_ushort(__float2bfloat16_rn(b)), r;
  asm("mul.rn.bf16 %0, %1, %2;" : "=h"(r) : "h"(ra), "h"(rb));
  return __bfloat162float(__ushort_as_bfloat16(r));
}
__device__ __forceinline__ float tr_neg(float a) {
  float r; asm("neg.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) {
  unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v)); return __ushort_as_bfloat16(r);
}
__device__ __forceinline__ float bfround(float v) { return bf2f(f2bf(v)); }
__device__ __forceinline__ float tr_max_nan(float a, float b) {  // triton_helpers.maximum: NaN-propagating
  float r; asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float tr_min_nan(float a, float b) {
  float r; asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ uint8_t f2e4m3(float v) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(v));
  return (uint8_t)(r & 0xFF);
}

// ---- vLLM _causal_conv1d_update_kernel, one channel, one token ----
// PTX of the stock kernel (sm_100, triton 3.8): p_j = mul.bf16x2(x_j, w_j); acc = add.rn.f32.bf16(p_0, bias) ...
// acc = add.rn.f32.bf16(p_j, acc); e = ex2.approx.f32(acc * -log2(e)); d = e + 1; out = div.full.f32(acc, d)
// (all inputs are exact bf16 values held in fp32 registers here)
__device__ __forceinline__ float conv_silu(float bias, float const (&w)[KW], float x0, float x1, float x2, float x3) {
  float acc = bias;
  acc = tr_add(acc, bf_mul(x0, w[0]));
  acc = tr_add(acc, bf_mul(x1, w[1]));
  acc = tr_add(acc, bf_mul(x2, w[2]));
  acc = tr_add(acc, bf_mul(x3, w[3]));
  float e = tr_ex2(tr_mul(acc, -1.4426950408889634f));
  return tr_div_full(acc, tr_add(e, 1.0f));
}

// ---- Inductor gated-RMSNorm + static FP8 quant (triton_per_fused__to_copy_add_clamp_mean_mul_pow_reciprocal_rsqrt_
// silu_view_0 as generated by vLLM's compile; num_warps=1, sizePerThread=[1,16]); instruction-level mirror of its PTX.
__device__ __forceinline__ float ptx_fma_rn_ftz(float a, float b, float c) {
  float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float ptx_fma_rm_ftz(float a, float b, float c) {
  float r; asm("fma.rm.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r;
}
__device__ __forceinline__ float ptx_sat_ftz(float a) {
  float r; asm("cvt.ftz.sat.f32.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
__device__ __forceinline__ float ptx_ex2_ftz(float a) {
  float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
// silu(g) = g / (1 + exp(-g)) with libdevice __nv_expf (ftz) whose final scale multiply is contracted with "+ 1"
__device__ __forceinline__ float ind_silu(float g) {
  float const n = tr_neg(g);
  float t = ptx_fma_rn_ftz(n, __uint_as_float(0x3BBB989Du), 0.5f);
  t = ptx_sat_ftz(t);
  float const j = ptx_fma_rm_ftz(t, __uint_as_float(0x437C0000u), __uint_as_float(0x4B400001u));
  float k = tr_neg(tr_add(j, __uint_as_float(0xCB40007Fu)));
  float u = ptx_fma_rn_ftz(n, __uint_as_float(0x3FB8AA3Bu), k);
  u = ptx_fma_rn_ftz(n, __uint_as_float(0x32A57060u), u);
  float const sc = __uint_as_float(__float_as_uint(j) << 23);
  float const e = ptx_ex2_ftz(u);
  float const d = tr_fma(e, sc, 1.0f);
  return tr_div_full(g, d);
}
// v = float(ssm_out) * silu(float(gate))  (tmp9 = tmp1 * tmp8)
__device__ __forceinline__ float ind_v(float y, float gt) { return tr_mul(y, ind_silu(gt)); }
__device__ __forceinline__ float ind_rstd(float sumsq) {
  float mean = tr_div_full(sumsq, 1024.0f);
  return tr_rsqrt(tr_add(mean, 1e-05f));
}
// out = fp8(clamp(((v * rstd) * w) * (1 / scale)))  (no intermediate bf16 rounding in the compiled kernel)
__device__ __forceinline__ uint8_t ind_out(float v, float rstd, float w, float inv_scale) {
  float q = tr_mul(tr_mul(tr_mul(v, rstd), w), inv_scale);
  q = tr_min_nan(tr_max_nan(q, -448.0f), 448.0f);
  return f2e4m3(q);
}
// Sum of squares over a 1024-element group in the Inductor kernel's order (1 warp, persistent reduction, blocked
// layout sizePerThread=[1,IND_SPT]): lane t owns e(t,i) = (i / IND_SPT) * (32 * IND_SPT) + t * IND_SPT + i % IND_SPT,
// i = 0..31 in register order. Triton's in-thread reduction: pairs P_k = (s_2k, s_2k+1) summed as a balanced
// tree with add.f32x2 (P_0+P_1, P_2+P_3, ...), then x + y, then a butterfly over the warp (16, 8, 4, 2, 1).
constexpr int IND_SPT = 16;
constexpr int IND_NPT = GSIZE / 32;  // 32 elements per lane
__device__ __forceinline__ int ind_elem(int lane, int i) {
  return (i / IND_SPT) * (32 * IND_SPT) + lane * IND_SPT + (i % IND_SPT);
}
// in-thread: Q_m = fma(P_2m, P_2m, P_2m+1 * P_2m+1) on pairs P_k = (s_2k, s_2k+1) (f32x2), Q tree-summed pairwise,
// then x + y, then butterfly 16, 8, 4, 2, 1
template <typename F>
__device__ __forceinline__ float ind_sumsq(int lane, F&& get) {
  float s[IND_NPT];
#pragma unroll
  for (int i = 0; i < IND_NPT; i++) s[i] = get(ind_elem(lane, i));
  float x[IND_NPT / 4], y[IND_NPT / 4];
#pragma unroll
  for (int m = 0; m < IND_NPT / 4; m++) {
    x[m] = tr_fma(s[4 * m], s[4 * m], tr_mul(s[4 * m + 2], s[4 * m + 2]));
    y[m] = tr_fma(s[4 * m + 1], s[4 * m + 1], tr_mul(s[4 * m + 3], s[4 * m + 3]));
  }
#pragma unroll
  for (int n = IND_NPT / 4; n > 1; n /= 2) {
#pragma unroll
    for (int k = 0; k < n / 2; k++) {
      x[k] = tr_add(x[2 * k], x[2 * k + 1]);
      y[k] = tr_add(y[2 * k], y[2 * k + 1]);
    }
  }
  float acc = tr_add(x[0], y[0]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc = tr_add(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  return acc;
}

// The in-thread tree of ind_sumsq splits by chunk: lane t's x/y partial over its first 16 elements
// (global chunk t, elements 16t..16t+15) is S0 = (Q0+Q1)+(Q2+Q3), over its second 16 (chunk 32+t) S1 = (Q4+Q5)+(Q6+Q7),
// and the lane value is (S0+S1).x + (S0+S1).y. So each CTA computes S for the chunks it owns and only 2 float2
// partials per lane cross the cluster: identical operations in identical order.
__device__ __forceinline__ float2 ind_chunk_partial(float const* v16) {
  float x[4], y[4];
#pragma unroll
  for (int m = 0; m < 4; m++) {
    x[m] = tr_fma(v16[4 * m], v16[4 * m], tr_mul(v16[4 * m + 2], v16[4 * m + 2]));
    y[m] = tr_fma(v16[4 * m + 1], v16[4 * m + 1], tr_mul(v16[4 * m + 3], v16[4 * m + 3]));
  }
  return make_float2(tr_add(tr_add(x[0], x[1]), tr_add(x[2], x[3])), tr_add(tr_add(y[0], y[1]), tr_add(y[2], y[3])));
}
__device__ __forceinline__ float ind_combine(float2 s0, float2 s1) {
  float acc = tr_add(tr_add(s0.x, s1.x), tr_add(s0.y, s1.y));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc = tr_add(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  return acc;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// conv channel owned by local index c of a CTA (rank r of group g, HPC heads): x channel or B/C slice
template <int HPC>
__device__ __forceinline__ int conv_channel(int c, int g, int r) {
  constexpr int ROWS = HPC * DIM;
  constexpr int BCS = DSTATE / (HPG / HPC);
  if (c < ROWS) return (g * HPG + r * HPC) * DIM + c;
  if (c < ROWS + BCS) return B_OFF + g * DSTATE + r * BCS + (c - ROWS);
  return C_OFF + g * DSTATE + r * BCS + (c - ROWS - BCS);
}

// =====================================================================================================
// Single-token decode (no spec): grid (CS, ngroups, ntokens), cluster (CS,1,1), 256 threads.
// =====================================================================================================
template <int HPC, int PHILOX_ROUNDS>
__global__ void __launch_bounds__(256) fused_stp_kernel(Params const p) {
  constexpr int CS = HPG / HPC;
  constexpr int ROWS = HPC * DIM;
  constexpr int BCS = DSTATE / CS;
  constexpr int NCH = ROWS + 2 * BCS;
  constexpr int NWARPS = 8;
  constexpr int RPW = ROWS / NWARPS;
  static_assert(NCH <= 256 && ROWS % NWARPS == 0);

  cg::cluster_group cluster = cg::this_cluster();
  int const r = (int)cluster.block_rank();
  int const g = blockIdx.y;
  int const m = blockIdx.z;
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int const h0 = g * HPG + r * HPC;

  __shared__ __align__(16) bf16 s_x[ROWS];
  __shared__ __align__(16) bf16 s_bc_own[2][DSTATE];  // owner slices (peer-read)
  __shared__ __align__(16) bf16 s_B[DSTATE];
  __shared__ __align__(16) bf16 s_C[DSTATE];
  __shared__ __align__(16) bf16 s_y[ROWS];
  __shared__ __align__(16) float s_v[ROWS];          // norm inputs
  __shared__ __align__(16) float2 s_part[ROWS / 16];  // per-chunk norm partials (peer-read)
  __shared__ float s_rstd;

  // ---- prologue: model parameters only ----
  int const ch = tid < NCH ? conv_channel<HPC>(tid, g, r) : 0;
  float w[KW], bias = 0.f;
  if (tid < NCH) {
#pragma unroll
    for (int j = 0; j < KW; j++) w[j] = bf2f(p.conv_w[ch * p.w_sd + j * p.w_sw]);
    bias = bf2f(p.conv_b[ch]);
  }
  float nw = 0.f;
  if (tid < ROWS) nw = bf2f(p.nw[h0 * DIM + tid]);
  float const inv_scale = tr_div_full(1.0f, *p.scale);

  pdl_wait();

  int64_t const slot = (int64_t)p.sidx[(int64_t)m * p.sidx_s0];
  bf16* __restrict__ prow = p.proj + (int64_t)m * p.proj_s;

  // prefetch this warp's SSM state rows (lane owns state[d][4*lane .. 4*lane+3], as in the stock kernel)
  using load_state_t = PackedAligned<__half, getVectorLoadSizeForFullUtilization<__half, DSTATE>()>;
  static_assert(load_state_t::count == 4 && DSTATE == 32 * 4);
  load_state_t rSt[RPW];
#pragma unroll
  for (int rr = 0; rr < RPW; rr++) {
    rSt[rr] = make_zeros<load_state_t>();
    int const row = warp * RPW + rr;
    if (slot != (int64_t)p.pad && m < p.nt)
      rSt[rr] = *reinterpret_cast<load_state_t const*>(
          &p.ssm[slot * p.ssm_s + (int64_t)(h0 + row / DIM) * DIM * DSTATE + (row % DIM) * DSTATE + lane * 4]);
  }

  // ---- conv1d update: one channel per thread ----
  if (tid < NCH) {
    bf16* xp = prow + XBC_OFF + ch;
    bf16 outv = __ushort_as_bfloat16(0);
    if (slot == (int64_t)p.pad) {
      if (m < p.nt) outv = *xp;  // null slot: stock conv kernel returns early -> SSU reads the raw input
    } else {
      bf16 const xin = *xp;
      bf16* sp = p.cs + slot * p.cs_seq + (int64_t)ch * p.cs_dim;
      bf16 const c0 = sp[0], c1 = sp[p.cs_tok], c2 = sp[2 * p.cs_tok];
      float const o = conv_silu(bias, w, bf2f(c0), bf2f(c1), bf2f(c2), bf2f(xin));
      sp[0] = c1; sp[p.cs_tok] = c2; sp[2 * p.cs_tok] = xin;
      outv = f2bf(o);
      *xp = outv;
    }
    if (tid < ROWS) s_x[tid] = outv;
    else if (tid < ROWS + BCS) s_bc_own[0][r * BCS + (tid - ROWS)] = outv;
    else s_bc_own[1][r * BCS + (tid - ROWS - BCS)] = outv;
  }
  // stock: the conv grid covers all index rows, the SSU grid only the first num_decode_tokens rows (p.nt);
  // the norm of the remaining rows reads uninitialized ssm_output in stock, so nothing is written for them.
  if (m >= p.nt) return;  // uniform across the cluster, before any cluster barrier
  float gate = 0.f;
  if (tid < ROWS) gate = bf2f(prow[h0 * DIM + tid]);
  int64_t const rand_seed = p.seed ? *p.seed : 0;

  cluster_arrive();
  cluster_wait();
  // gather the group's B and C (conv outputs) from their owner CTAs
  {
    int const which = tid >> 7, j = tid & 127;
    bf16 const* src = cluster.map_shared_rank(&s_bc_own[which][j], j / BCS);
    (which ? s_C : s_B)[j] = *src;
  }
  __syncthreads();

  // ---- selective state update (FlashInfer selective_state_update_kernel_simple, per-row code verbatim) ----
  {
    int const state_batch_is_pad = (slot == (int64_t)p.pad);
#pragma unroll
    for (int rr = 0; rr < RPW; rr++) {
      int const row = warp * RPW + rr;
      int const hl = row / DIM;
      int const head = h0 + hl;
      int const d = row % DIM;
      auto const* __restrict__ A = p.A;
      auto const A_value = toFloat(A[head]);
      auto dt_value = toFloat(prow[DT_OFF + head]);
      dt_value += toFloat(p.dtb[head]);
      dt_value = thresholded_softplus(dt_value);
      auto const dA = __expf(A_value * dt_value);
      auto d_value = toFloat(p.D[head]);

      auto const state_ptr_offset = slot * p.ssm_s + head * DIM * DSTATE;
      __half* __restrict__ state = p.ssm + state_ptr_offset;
      __half* __restrict__ dst_state = p.ssm + slot * p.ssm_s + head * DIM * DSTATE;

      float x_value = toFloat(s_x[row]);
      float state_decode_scale = 1.f;
      float out_value = d_value * x_value * int(lane == 0);
      [[maybe_unused]] uint32_t rand_ints[4];
      for (int iter = 0, i = lane * load_state_t::count; i < DSTATE;
           iter++, i += flashinfer::mamba::warpSize * load_state_t::count) {
        auto rState = rSt[rr];  // = make_zeros, or *reinterpret_cast<load_state_t*>(&state[d * DSTATE + i]) (prefetched)
        (void)state;
        for (int ii = 0; ii < load_state_t::count; ii++) {
          if constexpr (PHILOX_ROUNDS > 0) {
            if (ii % 4 == 0)
              philox_randint4x<PHILOX_ROUNDS>(rand_seed, state_ptr_offset + d * DSTATE + i + ii, rand_ints[0],
                                              rand_ints[1], rand_ints[2], rand_ints[3]);
          }
          auto state_value = toFloat(rState.val[ii]) * state_decode_scale;
          auto B_value = toFloat(s_B[i + ii]);
          auto C_value = toFloat(s_C[i + ii]);
          auto const dB = B_value * dt_value;
          auto const new_state = state_value * dA + dB * x_value;
          if constexpr (PHILOX_ROUNDS > 0) {
            rState.val[ii] = cvt_rs_f16_f32(new_state, rand_ints[ii % 4] & 0x1FFFu);
          } else {
            convertAndStore(&rState.val[ii], new_state);
          }
          out_value += new_state * C_value;
        }
        if (!state_batch_is_pad) {
          *reinterpret_cast<load_state_t*>(&dst_state[d * DSTATE + i]) = rState;
        }
      }
      out_value = warpReduceSum(out_value);
      if (lane == 0) {
        bf16 yv;
        convertAndStore(&yv, out_value);
        s_y[row] = yv;
      }
    }
  }
  __syncthreads();

  // ---- gated RMSNorm inputs ----
  if (tid < ROWS) {
    float const v = ind_v(bf2f(s_y[tid]), gate);
    s_v[tid] = v;
    if (p.y) p.y[(int64_t)m * p.y_s + h0 * DIM + tid] = s_y[tid];
  }
  __syncthreads();
  if (tid < ROWS / 16) s_part[tid] = ind_chunk_partial(&s_v[16 * tid]);
  cluster_arrive();
  cluster_wait();
  pdl_trigger();
  if (warp == 0) {
    constexpr int CPC = ROWS / 16;  // chunks per CTA
    float2 const s0 = *cluster.map_shared_rank(&s_part[lane % CPC], lane / CPC);
    float2 const s1 = *cluster.map_shared_rank(&s_part[(32 + lane) % CPC], (32 + lane) / CPC);
    float const ss = ind_combine(s0, s1);
    if (lane == 0) s_rstd = ind_rstd(ss);
  }
  __syncthreads();
  cluster_arrive();  // this CTA is done reading peers' shared memory
  if (tid < ROWS) {
    reinterpret_cast<uint8_t*>(p.out)[(int64_t)m * p.out_s + h0 * DIM + tid] = ind_out(s_v[tid], s_rstd, nw, inv_scale);
  }
  cluster_wait();  // peers are done reading this CTA's shared memory
}

// =====================================================================================================
// MTP verify (varlen, NT = max tokens per sequence): grid (CS, ngroups, nseq), cluster (CS,1,1), 256 threads.
// SSU part: FlashInfer selective_state_update_kernel_simple_mtp / update_state_simple per-row code verbatim.
// =====================================================================================================
template <int HPC, int NT, int PHILOX_ROUNDS>
__global__ void __launch_bounds__(512) fused_mtp_kernel(Params const p) {
  constexpr int CS = HPG / HPC;
  constexpr int ROWS = HPC * DIM;
  constexpr int BCS = DSTATE / CS;
  constexpr int NCH = ROWS + 2 * BCS;
  constexpr int NTHREADS = 512;  // one row pass: 16 warps x 4 rows
  constexpr int NWARPS = NTHREADS / 32;
  static_assert(NCH <= NTHREADS && NT <= NWARPS);
  // update_state_simple constants (state_t = half, DSTATE = 128)
  constexpr int lanesPerRow = 8;  // FlashInfer mtp::simple_horiz::LANES_PER_ROW
  constexpr int rowsPerWarp = 32 / lanesPerRow;
  constexpr int ROWS_PER_PASS = NWARPS * rowsPerWarp;  // 64
  constexpr int numPasses = ROWS / ROWS_PER_PASS;
  constexpr int DSTATE_PADDED = mtp::nextPow2(DSTATE);
  constexpr int stateValuesPerThread = DSTATE_PADDED / lanesPerRow;
  constexpr int bankSize = sizeof(uint32_t);
  constexpr int stateValuesPerBank = bankSize / sizeof(__half);
  constexpr int numBanks = 32;
  constexpr int sramReadsPerThreadPerTile = numBanks / lanesPerRow;
  constexpr int elemsPerTileMember = sramReadsPerThreadPerTile * stateValuesPerBank;
  constexpr int elemsPerTile = elemsPerTileMember * lanesPerRow;
  constexpr int numTiles = stateValuesPerThread / elemsPerTileMember;
  constexpr int pairsPerTileMember = elemsPerTileMember / 2;
  using packed_tile_t = PackedAligned<__half, elemsPerTileMember>;
  static_assert(ROWS % ROWS_PER_PASS == 0);
  constexpr int64_t SKIP = -1;

  cg::cluster_group cluster = cg::this_cluster();
  int const r = (int)cluster.block_rank();
  int const g = blockIdx.y;
  int const s = blockIdx.z;
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int const h0 = g * HPG + r * HPC;

  __shared__ __align__(16) bf16 s_x[NT][ROWS];
  __shared__ __align__(16) bf16 s_bc_own[NT][2][DSTATE];
  __shared__ __align__(16) bf16 s_B[NT][DSTATE];
  __shared__ __align__(16) bf16 s_C[NT][DSTATE];
  __shared__ float s_dt[HPC][NT];
  __shared__ float s_out[NT][ROWS];
  __shared__ __align__(16) float s_v[NT][ROWS];
  __shared__ __align__(16) float2 s_part[NT][ROWS / 16];
  __shared__ int64_t s_dst[NT];
  __shared__ float s_rstd[NT];

  // ---- prologue: model parameters only ----
  int const ch = tid < NCH ? conv_channel<HPC>(tid, g, r) : 0;
  float w[KW], bias = 0.f;
  if (tid < NCH) {
#pragma unroll
    for (int j = 0; j < KW; j++) w[j] = bf2f(p.conv_w[ch * p.w_sd + j * p.w_sw]);
    bias = bf2f(p.conv_b[ch]);
  }
  float const inv_scale = tr_div_full(1.0f, *p.scale);

  pdl_wait();

  int const bos = p.cu[s];
  int const L = p.cu[s + 1] - bos;
  if (L <= 0) return;  // uniform across the cluster (no barrier is reached)
  int const num_acc = p.nacc[s];
  int64_t const conv_slot = (int64_t)p.sidx[(int64_t)s * p.sidx_s0];
  int const init_token_idx = max(num_acc - 1, 0);
  int64_t const state_batch =
      (int64_t)p.sidx[(int64_t)s * p.sidx_s0 + (int64_t)init_token_idx * p.sidx_s1];
  bool const is_pad = (state_batch == (int64_t)p.pad);
  // prefetch the SSM state of this thread's rows (raw fp16; converted exactly as stock: __half22float2)
  constexpr int lanesPerRow_ = 8;
  uint4 rRaw[ROWS / (NWARPS * (32 / lanesPerRow_))][2];
  {
    int const member = lane % lanesPerRow_, grp = lane / lanesPerRow_;
#pragma unroll
    for (int ps = 0; ps < ROWS / (NWARPS * (32 / lanesPerRow_)); ps++) {
      int const local_row = ps * NWARPS * (32 / lanesPerRow_) + warp * (32 / lanesPerRow_) + grp;
      __half const* srow = p.ssm + state_batch * p.ssm_s + (int64_t)(h0 + local_row / DIM) * DIM * DSTATE +
                           (int64_t)(local_row % DIM) * DSTATE;
#pragma unroll
      for (int t = 0; t < 2; t++) {
        rRaw[ps][t] = make_uint4(0, 0, 0, 0);
        if (!is_pad) rRaw[ps][t] = *reinterpret_cast<uint4 const*>(&srow[t * 64 + member * 8]);
      }
    }
  }
  if (tid < NT) {
    int const step = tid;
    if (is_pad || step >= L) {
      s_dst[step] = SKIP;
    } else {
      int64_t const di = (int64_t)p.didx[(int64_t)s * p.didx_s0 + (int64_t)step * p.didx_s1];
      s_dst[step] = (di == (int64_t)p.pad) ? SKIP : di;
    }
  }

  // ---- conv1d update (spec-decode rolling window, varlen), one channel per thread ----
  if (tid < NCH) {
    bf16 outv[NT];
    bf16 xin[NT];
#pragma unroll
    for (int t = 0; t < NT; t++)
      if (t < L) xin[t] = p.proj[(int64_t)(bos + t) * p.proj_s + XBC_OFF + ch];
    if (conv_slot != (int64_t)p.pad) {
      int64_t const off = (int64_t)num_acc - 1;  // conv_state_token_offset
      bf16* sp = p.cs + conv_slot * p.cs_seq + (int64_t)ch * p.cs_dim;
      bf16 const c0 = sp[(off + 0) * p.cs_tok], c1 = sp[(off + 1) * p.cs_tok], c2 = sp[(off + 2) * p.cs_tok];
      // new state: [old[off+1], old[off+2], x_0 .. x_{L-1}]  (positions 0 .. L+1)
      sp[0] = c1;
      sp[p.cs_tok] = c2;
#pragma unroll
      for (int t = 0; t < NT; t++)
        if (t < L) sp[(int64_t)(2 + t) * p.cs_tok] = xin[t];
      float a0 = bf2f(c0), a1 = bf2f(c1), a2 = bf2f(c2);
#pragma unroll
      for (int t = 0; t < NT; t++) {
        if (t < L) {
          float const xt = bf2f(xin[t]);
          float const o = conv_silu(bias, w, a0, a1, a2, xt);
          a0 = a1; a1 = a2; a2 = xt;
          outv[t] = f2bf(o);
          p.proj[(int64_t)(bos + t) * p.proj_s + XBC_OFF + ch] = outv[t];
        }
      }
    } else {
#pragma unroll
      for (int t = 0; t < NT; t++) outv[t] = xin[t];
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
      if (t < L) {
        if (tid < ROWS) s_x[t][tid] = outv[t];
        else if (tid < ROWS + BCS) s_bc_own[t][0][r * BCS + (tid - ROWS)] = outv[t];
        else s_bc_own[t][1][r * BCS + (tid - ROWS - BCS)] = outv[t];
      }
    }
  }
  // dt (load_simple): dt_bias first, then dt + bias, softplus
  for (int i = tid; i < HPC * NT; i += NTHREADS) {
    int const hl = i / NT, step = i % NT;
    if (step < L) {
      int const head = h0 + hl;
      float dt_bias_val = toFloat(p.dtb[head]);
      float dt_val = toFloat(p.proj[(int64_t)(bos + step) * p.proj_s + DT_OFF + head]);
      dt_val += dt_bias_val;
      dt_val = thresholded_softplus(dt_val);
      s_dt[hl][step] = dt_val;
    }
  }
  [[maybe_unused]] int64_t const rand_seed = p.seed ? *p.seed : 0;
  constexpr int EPT = (NT * ROWS + NTHREADS - 1) / NTHREADS;  // epilogue elements per thread
  float gate_r[EPT], nw_r[EPT];
#pragma unroll
  for (int k = 0; k < EPT; k++) {
    int const i = tid + k * NTHREADS, t = i / ROWS, e = i % ROWS;
    gate_r[k] = 0.f; nw_r[k] = 0.f;
    if (i < NT * ROWS && t < L) {
      gate_r[k] = bf2f(p.proj[(int64_t)(bos + t) * p.proj_s + h0 * DIM + e]);
      nw_r[k] = bf2f(p.nw[h0 * DIM + e]);
    }
  }
  cluster_arrive();
  cluster_wait();
  // gather B/C of all L tokens: one 16-byte DSMEM load per (token, B|C, owner slice of 8 channels)
  {
    constexpr int CHUNK = 8;
    constexpr int NCHUNK = DSTATE / CHUNK;  // 16 per array
    for (int i = tid; i < NT * 2 * NCHUNK; i += NTHREADS) {
      int const t = i / (2 * NCHUNK), which = (i / NCHUNK) % 2, c = i % NCHUNK;
      if (t < L) {
        int const j = c * CHUNK;
        uint4 const* src = cluster.map_shared_rank(reinterpret_cast<uint4 const*>(&s_bc_own[t][which][j]), j / BCS);
        *reinterpret_cast<uint4*>(which ? &s_C[t][j] : &s_B[t][j]) = *src;
      }
    }
  }
  __syncthreads();

  // ---- SSU MTP: update_state_simple per-row code verbatim (state in registers, lanesPerRow lanes per row) ----
  {
    int const member = lane % lanesPerRow;
    int const group = lane / lanesPerRow;
    auto baseCol = [&](int t, int e) -> int { return t * elemsPerTile + member * elemsPerTileMember + e; };
#pragma unroll
    for (int pass = 0; pass < numPasses; pass++) {
      int const local_row = pass * ROWS_PER_PASS + warp * rowsPerWarp + group;  // row within the CTA
      int const hl = local_row / DIM;
      int const head = h0 + hl;
      int const dd = local_row % DIM;
      float const A_val = toFloat(p.A[head]);
      float const D_val = toFloat(p.D[head]);
      auto const state_ptr_offset = state_batch * p.ssm_s + head * DIM * DSTATE;
      static_assert(numTiles == 2 && elemsPerTileMember == 8 && elemsPerTile == 64);
      float2 rState[numTiles][pairsPerTileMember];
#pragma unroll
      for (int t = 0; t < numTiles; t++) {
        __half2 const* h2 = reinterpret_cast<__half2 const*>(&rRaw[pass][t]);  // = state[dd][baseCol(t, 0) .. +8]
#pragma unroll
        for (int pp = 0; pp < pairsPerTileMember; pp++) {
          int const c0 = baseCol(t, pp * 2);
          if (c0 >= DSTATE || is_pad) {
            rState[t][pp] = {0.f, 0.f};
          } else {
            rState[t][pp] = toFloat2(h2[pp]);
          }
        }
      }

      // Philox draws of convertAndStoreSRHorizontal (same (seed, offset) arguments, same int truncation)
      [[maybe_unused]] uint32_t rnd[numTiles][elemsPerTileMember / 4][4];
      if constexpr (PHILOX_ROUNDS > 0) {
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
          int const col0 = baseCol(t, 0);
#pragma unroll
          for (int q = 0; q < elemsPerTileMember / 4; q++) {
            int const e = 4 * q;
            int const off32 = (int)state_ptr_offset;  // convertAndStoreSRHorizontal takes an int offset
            philox_randint4x<PHILOX_ROUNDS>(rand_seed, off32 + dd * DSTATE + col0 + e, rnd[t][q][0], rnd[t][q][1],
                                            rnd[t][q][2], rnd[t][q][3]);
          }
        }
      }
      for (int step = 0; step < NT; step++) {
        if (step >= L) break;
        int64_t const dst_slot = s_dst[step];
        float const dt_value = s_dt[hl][step];
        float const dA = __expf(A_val * dt_value);
        float const x_value = toFloat(s_x[step][local_row]);

        float2 out2 = {0.f, 0.f};
        float2 const dA2 = {dA, dA};
        float const dtx_value = dt_value * x_value;
        float2 const dtx2 = {dtx_value, dtx_value};
        bf16 const* __restrict__ B_step = &s_B[step][0];
        bf16 const* __restrict__ C_step = &s_C[step][0];
#pragma unroll
        for (int t = 0; t < numTiles; t++) {
#pragma unroll
          for (int pp = 0; pp < pairsPerTileMember; pp++) {
            int const c0 = baseCol(t, pp * 2);
            if (c0 >= DSTATE) continue;
            float2 const B2 = toFloat2(&B_step[c0]);
            float2 const C2 = toFloat2(&C_step[c0]);
            float2 dBx;
            mtp::mul_f32x2(dBx, B2, dtx2);
            mtp::fma_f32x2(rState[t][pp], dA2, rState[t][pp], dBx);
            mtp::fma_f32x2(out2, rState[t][pp], C2, out2);
          }
        }
        float out_value = out2.x + out2.y;
#pragma unroll
        for (int offset = lanesPerRow / 2; offset >= 1; offset /= 2) {
          out_value += __shfl_down_sync(UINT32_MAX, out_value, offset);
        }
        if (member == 0) {
          s_out[step][local_row] = out_value + D_val * x_value;
        }
        if (dst_slot != SKIP) {
          auto const dst_base = dst_slot * p.ssm_s + (int64_t)head * DIM * DSTATE + (int64_t)dd * DSTATE;
#pragma unroll
          for (int t = 0; t < numTiles; t++) {
            int const col0 = baseCol(t, 0);
            if (col0 >= DSTATE) continue;
            packed_tile_t rOut;
#pragma unroll
            for (int e = 0; e < elemsPerTileMember; e += 2) {
              float2 s2 = rState[t][e / 2];
              if constexpr (PHILOX_ROUNDS > 0) {
                // convertAndStoreSRHorizontal with its Philox draws hoisted out of the step loop: the counter
                // (int)state_ptr_offset + dd * DSTATE + col0 + e does not depend on the step, so the bits are
                // the ones the stock kernel recomputes at every step.
                uint32_t const packed = cvt_rs_f16x2_f32(s2.x, s2.y, rnd[t][e / 4][e / 2 % 2]);
                rOut.val[e] = __ushort_as_half(static_cast<uint16_t>(packed & 0xFFFFu));
                rOut.val[e + 1] = __ushort_as_half(static_cast<uint16_t>(packed >> 16));
              } else {
                uint32_t dummy[4];
                mtp::convertAndStoreSRHorizontal<__half, DSTATE, PHILOX_ROUNDS>(
                    rOut.val[e], rOut.val[e + 1], s2.x, s2.y, rand_seed, state_ptr_offset, dd, col0, e, dummy);
              }
            }
            *reinterpret_cast<packed_tile_t*>(&p.ssm[dst_base + col0]) = rOut;
          }
        }
      }
    }
  }
  __syncthreads();
  // ---- epilogue: ssm out -> bf16 (convertAndStore), gated-RMSNorm inputs ----
#pragma unroll
  for (int k = 0; k < EPT; k++) {
    int const i = tid + k * NTHREADS, t = i / ROWS, e = i % ROWS;
    if (i < NT * ROWS && t < L) {
      bf16 yv;
      convertAndStore(&yv, s_out[t][e]);
      s_v[t][e] = ind_v(bf2f(yv), gate_r[k]);
      if (p.y) p.y[(int64_t)(bos + t) * p.y_s + h0 * DIM + e] = yv;
    }
  }
  __syncthreads();
  for (int i = tid; i < NT * (ROWS / 16); i += NTHREADS) {
    int const t = i / (ROWS / 16), c = i % (ROWS / 16);
    if (t < L) s_part[t][c] = ind_chunk_partial(&s_v[t][16 * c]);
  }
  cluster_arrive();
  cluster_wait();
  pdl_trigger();
  if (warp < NT && warp < L) {
    int const t = warp;
    constexpr int CPC = ROWS / 16;
    float2 const s0 = *cluster.map_shared_rank(&s_part[t][lane % CPC], lane / CPC);
    float2 const s1 = *cluster.map_shared_rank(&s_part[t][(32 + lane) % CPC], (32 + lane) / CPC);
    float const ss = ind_combine(s0, s1);
    if (lane == 0) s_rstd[t] = ind_rstd(ss);
  }
  __syncthreads();
  cluster_arrive();
#pragma unroll
  for (int k = 0; k < EPT; k++) {
    int const i = tid + k * NTHREADS, t = i / ROWS, e = i % ROWS;
    if (i < NT * ROWS && t < L) {
      reinterpret_cast<uint8_t*>(p.out)[(int64_t)(bos + t) * p.out_s + h0 * DIM + e] = ind_out(s_v[t][e], s_rstd[t], nw_r[k], inv_scale);
    }
  }
  cluster_wait();
}

// =====================================================================================================
// Standalone gated-RMSNorm + FP8 quant (exact Inductor replica): one warp per (token, group) row.
// Used by the non-fused (prefill / mixed) path of the op and by the tests.
// =====================================================================================================
__global__ void __launch_bounds__(128) norm_quant_kernel(bf16 const* __restrict__ y, int64_t y_s,
                                                         bf16 const* __restrict__ gate, int64_t g_s,
                                                         bf16 const* __restrict__ nw, float const* __restrict__ scale,
                                                         __nv_fp8_e4m3* __restrict__ out, int64_t o_s, int nrows) {
  int const row = blockIdx.x * 4 + (threadIdx.x >> 5);
  int const lane = threadIdx.x & 31;
  if (row >= nrows) return;
  int const m = row / (XDIM / GSIZE), g = row % (XDIM / GSIZE);
  bf16 const* yr = y + (int64_t)m * y_s + g * GSIZE;
  bf16 const* gr = gate + (int64_t)m * g_s + g * GSIZE;
  float v[IND_NPT];
#pragma unroll
  for (int i = 0; i < IND_NPT; i++) {
    int const e = ind_elem(lane, i);
    v[i] = ind_v(bf2f(yr[e]), bf2f(gr[e]));
  }
  float const ss = ind_sumsq(lane, [&](int e) -> float {
    // inverse of ind_elem for this lane (e is always one of this lane's elements)
    return v[(e / (32 * IND_SPT)) * IND_SPT + (e % IND_SPT)];
  });
  float const rstd = ind_rstd(ss);
  float const inv_scale = tr_div_full(1.0f, *scale);
#pragma unroll
  for (int i = 0; i < IND_NPT; i++) {
    int const e = ind_elem(lane, i);
    reinterpret_cast<uint8_t*>(out)[(int64_t)m * o_s + g * GSIZE + e] = ind_out(v[i], rstd, bf2f(nw[g * GSIZE + e]), inv_scale);
  }
}

template <typename K>
static int launch_cluster(K kernel, dim3 grid, dim3 block, int cs, int pdl, cudaStream_t stream, Params const& p) {
  // all kernels share this instantiation (same signature): remember which ones got the attribute
  static void const* done[32] = {};
  static int ndone = 0;
  bool seen = false;
  for (int i = 0; i < ndone; i++) seen |= (done[i] == (void const*)kernel);
  if (!seen) {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (e != cudaSuccess) return (int)e;
    if (ndone < 32) done[ndone++] = (void const*)kernel;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cs;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  return (int)cudaLaunchKernelEx(&cfg, kernel, p);
}

}  // namespace vllm_mamba2fd

using namespace vllm_mamba2fd;

extern "C" {

int mamba2fd_num_args() { return A_NARGS; }
const char* mamba2fd_err(int e) { return e < 0 ? "no instantiation" : cudaGetErrorString((cudaError_t)e); }
// hpc: heads per CTA (1 -> cluster 16, 2 -> cluster 8); philox: 0 or 5 (must match the stock FlashInfer module)
int mamba2fd_stp(int64_t const* args, int ntok, int ngroups, int hpc, int philox, int pdl, void* stream) {
  Params const p = unpack(args);
  if (ntok <= 0) return 0;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  int const cs = HPG / hpc;
  dim3 grid(cs, ngroups, ntok), block(256);
#define MO_STP(H, PR) return launch_cluster(fused_stp_kernel<H, PR>, grid, block, cs, pdl, st, p)
  if (hpc == 1 && philox == 5) MO_STP(1, 5);
  if (hpc == 2 && philox == 5) MO_STP(2, 5);
  if (hpc == 1 && philox == 0) MO_STP(1, 0);
  if (hpc == 2 && philox == 0) MO_STP(2, 0);
#undef MO_STP
  return -1;
}

int mamba2fd_mtp(int64_t const* args, int nseq, int ngroups, int nt, int hpc, int philox, int pdl,
                       void* stream) {
  Params const p = unpack(args);
  if (nseq <= 0) return 0;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  int const cs = HPG / hpc;
  dim3 grid(cs, ngroups, nseq), block(512);
#define MO_MTP(H, N, PR) return launch_cluster(fused_mtp_kernel<H, N, PR>, grid, block, cs, pdl, st, p)
  if (nt == 6 && hpc == 1 && philox == 5) MO_MTP(1, 6, 5);
  if (nt == 6 && hpc == 2 && philox == 5) MO_MTP(2, 6, 5);
  if (nt == 6 && hpc == 1 && philox == 0) MO_MTP(1, 6, 0);
  if (nt == 7 && hpc == 1 && philox == 5) MO_MTP(1, 7, 5);
  if (nt == 8 && hpc == 1 && philox == 5) MO_MTP(1, 8, 5);
  if (nt == 8 && hpc == 1 && philox == 0) MO_MTP(1, 8, 0);
#undef MO_MTP
  return -1;
}

int mamba2fd_norm_quant(void const* y, int64_t y_s, void const* gate, int64_t g_s, void const* nw, void const* scale,
                        void* out, int64_t o_s, int ntok, void* stream) {
  int const nrows = ntok * (XDIM / GSIZE);
  if (nrows <= 0) return 0;
  norm_quant_kernel<<<(nrows + 3) / 4, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<bf16 const*>(y), y_s, reinterpret_cast<bf16 const*>(gate), g_s,
      reinterpret_cast<bf16 const*>(nw), reinterpret_cast<float const*>(scale),
      reinterpret_cast<__nv_fp8_e4m3*>(out), o_s, nrows);
  return (int)cudaGetLastError();
}

}  // extern "C"
