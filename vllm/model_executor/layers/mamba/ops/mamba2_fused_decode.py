# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Mamba-2 decode chain (VLLM_MAMBA2_FUSED_DECODE).

One CUDA kernel per Mamba-2 layer for decode-only batches replaces
  causal_conv1d_update (Triton) -> FlashInfer selective_state_update (with
  optional Philox stochastic rounding of an fp16 SSM cache) -> gated RMSNorm +
  static per-tensor FP8 quantization of the out_proj input (Inductor kernel)
and produces bit-identical results (outputs, conv state, SSM state); see
mamba2_fused_decode.cu for the exactness contract.

The kernel is JIT-compiled once per machine with nvcc against FlashInfer's own
Mamba headers (the SSU arithmetic is FlashInfer's device code), using the same
flags as FlashInfer's SSU JIT module, and loaded through ctypes. It only runs
under the exact configuration it reproduces (checked by
`fused_decode_supported`), otherwise the stock path is used.
"""

import ctypes
import hashlib
import os
import subprocess

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mamba2_fused_decode.cu")

# Must match `enum Arg` in mamba2_fused_decode.cu.
_ARGS = ["PROJ", "PROJ_S", "CONV_W", "W_SD", "W_SW", "CONV_B", "CS", "CS_SEQ",
         "CS_DIM", "CS_TOK", "SIDX", "SIDX_S0", "SIDX_S1", "DIDX", "DIDX_S0",
         "DIDX_S1", "CU", "NACC", "SSM", "SSM_S", "A", "D", "DTB", "SEED", "NW",
         "SCALE", "OUT", "OUT_S", "Y", "Y_S", "NSEQ", "PAD", "NT"]  # fmt: skip
_IDX = {k: i for i, k in enumerate(_ARGS)}

# (tokens per sequence, heads per CTA, philox rounds) instantiated for the MTP
# verify kernel; single-token decode is instantiated for philox 0 and 5.
MTP_INSTANTIATIONS = {(6, 1, 5), (6, 2, 5), (6, 1, 0), (7, 1, 5), (8, 1, 5), (8, 1, 0)}
STP_PHILOX = (0, 5)

_LIB: ctypes.CDLL | None = None


def _nvcc() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    return os.path.join(cuda_home, "bin", "nvcc")


def _flags(fi_include: str) -> list[str]:
    # FlashInfer's SSU JIT flags (flashinfer.jit: -O3 -use_fast_math, sm_100a)
    # so the copied FlashInfer device code compiles the same way.
    return [
        "-std=c++20", "-O3", "-use_fast_math", "--expt-relaxed-constexpr",
        "-static-global-template-stub=false", "-DNDEBUG",
        "-DFLASHINFER_ENABLE_F16", "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3", "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DFLASHINFER_ENABLE_FP8_E8M0", "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-DFLASHINFER_MAMBA_ENABLE_SM90", "-DFLASHINFER_MAMBA_ENABLE_SM100",
        "-gencode=arch=compute_100a,code=sm_100a", f"-I{fi_include}",
        "-Xcompiler", "-fPIC", "-shared",
    ]  # fmt: skip


def _build() -> str:
    import filelock
    import flashinfer

    import vllm.envs as envs

    fi_include = os.path.join(os.path.dirname(flashinfer.__file__), "data", "include")
    flags = _flags(fi_include)
    with open(_SRC, "rb") as f:
        src = f.read()
    key = hashlib.sha256(src + " ".join(flags).encode() + flashinfer.__version__.encode())
    out_dir = os.path.join(envs.VLLM_CACHE_ROOT, "mamba2_fused_decode", key.hexdigest()[:16])
    so = os.path.join(out_dir, "libmamba2_fused_decode.so")
    os.makedirs(out_dir, exist_ok=True)
    with filelock.FileLock(so + ".lock"):
        if not os.path.exists(so):
            logger.info("Building the fused Mamba-2 decode kernel (nvcc) -> %s", so)
            tmp = so + f".tmp{os.getpid()}"
            subprocess.run([_nvcc(), *flags, "-o", tmp, _SRC], check=True, capture_output=True)
            os.replace(tmp, so)
    return so


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(_build())
        L.mamba2fd_num_args.restype = ctypes.c_int
        assert L.mamba2fd_num_args() == len(_ARGS)
        P = ctypes.POINTER(ctypes.c_int64)
        i, v = ctypes.c_int, ctypes.c_void_p
        L.mamba2fd_stp.argtypes = [P, i, i, i, i, i, v]
        L.mamba2fd_mtp.argtypes = [P, i, i, i, i, i, i, v]
        L.mamba2fd_norm_quant.argtypes = [v, ctypes.c_int64, v, ctypes.c_int64, v, v, v, ctypes.c_int64, i, v]
        L.mamba2fd_err.restype = ctypes.c_char_p
        L.mamba2fd_err.argtypes = [i]
        for f in (L.mamba2fd_stp, L.mamba2fd_mtp, L.mamba2fd_norm_quant):
            f.restype = ctypes.c_int
        _LIB = L
    return _LIB


def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def pack_args(*, proj, conv_w, conv_b, conv_state, sidx, didx, cu, nacc, ssm, A, D, dt_bias, seed,
              norm_w, scale, out, y=None, nseq, pad, nt) -> ctypes.Array:
    """conv_state: (slots, dim, state_len) view; sidx/didx: 2-D int32 state indices."""
    a = (ctypes.c_int64 * len(_ARGS))()
    vals = {
        "PROJ": _ptr(proj), "PROJ_S": proj.stride(0), "CONV_W": _ptr(conv_w),
        "W_SD": conv_w.stride(0), "W_SW": conv_w.stride(1), "CONV_B": _ptr(conv_b),
        "CS": _ptr(conv_state), "CS_SEQ": conv_state.stride(0),
        "CS_DIM": conv_state.stride(1), "CS_TOK": conv_state.stride(2),
        "SIDX": _ptr(sidx), "SIDX_S0": sidx.stride(0), "SIDX_S1": sidx.stride(1),
        "DIDX": _ptr(didx), "DIDX_S0": didx.stride(0), "DIDX_S1": didx.stride(1),
        "CU": _ptr(cu), "NACC": _ptr(nacc), "SSM": _ptr(ssm), "SSM_S": ssm.stride(0),
        "A": _ptr(A), "D": _ptr(D), "DTB": _ptr(dt_bias), "SEED": _ptr(seed),
        "NW": _ptr(norm_w), "SCALE": _ptr(scale), "OUT": _ptr(out), "OUT_S": out.stride(0),
        "Y": _ptr(y), "Y_S": y.stride(0) if y is not None else 0,
        "NSEQ": nseq, "PAD": pad, "NT": nt,
    }  # fmt: skip
    for k, val in vals.items():
        a[_IDX[k]] = int(val)
    return a


_SMS: dict[int | None, int] = {}


def num_sms(device: torch.device) -> int:
    if device.index not in _SMS:
        _SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device.index]


def _stream() -> ctypes.c_void_p:
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {rc} {lib().mamba2fd_err(rc)!r}")


def fused_decode_stp(args, nseq: int, *, hpc: int = 1, philox: int = 5, pdl: bool = True) -> None:
    """Single-token decode: conv for nseq index rows, SSU+norm for args.NT rows."""
    _check(lib().mamba2fd_stp(args, nseq, 2, hpc, philox, int(pdl), _stream()), "mamba2fd_stp")


def fused_decode_mtp(args, nseq: int, *, nt: int, hpc: int = 1, philox: int = 5,
                     pdl: bool = True) -> None:  # fmt: skip
    """Multi-token (speculative verify) decode, varlen via cu_seqlens."""
    _check(lib().mamba2fd_mtp(args, nseq, 2, nt, hpc, philox, int(pdl), _stream()), "mamba2fd_mtp")


def gated_norm_fp8_quant(y, gate, norm_w, scale, out, ntok: int) -> None:
    """Exact replica of the Inductor gated-RMSNorm + static FP8 quant kernel."""
    _check(
        lib().mamba2fd_norm_quant(_ptr(y), y.stride(0), _ptr(gate), gate.stride(0), _ptr(norm_w),
                                  _ptr(scale), _ptr(out), out.stride(0), ntok, _stream()),
        "mamba2fd_norm_quant",
    )  # fmt: skip
