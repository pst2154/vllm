# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.layers.fused_moe.oracle import nvfp4
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend


def _config(moe_backend: str):
    return SimpleNamespace(
        moe_backend=moe_backend,
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
    )


def _clear_nvfp4_backend_env(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_MOE_FP4", raising=False)
    monkeypatch.delenv("VLLM_FLASHINFER_MOE_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_TEST_FORCE_FP8_MARLIN", raising=False)


def _mock_backend_support(monkeypatch, supported_backends: set[NvFp4MoeBackend]):
    def backend_to_kernel_cls(backend: NvFp4MoeBackend):
        class FakeExperts:
            @staticmethod
            def is_supported_config(
                cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            ):
                return backend in supported_backends, "mock unsupported"

        return [FakeExperts]

    monkeypatch.setattr(nvfp4, "backend_to_kernel_cls", backend_to_kernel_cls)


def test_auto_prefers_vllm_cutlass_over_flashinfer_cutlass(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
            NvFp4MoeBackend.VLLM_CUTLASS,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("auto"), None, None)

    assert backend == NvFp4MoeBackend.VLLM_CUTLASS


def test_auto_prefers_other_flashinfer_backends_before_vllm_cutlass(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            NvFp4MoeBackend.VLLM_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("auto"), None, None)

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL


def test_auto_falls_back_to_flashinfer_cutlass_when_native_unsupported(
    monkeypatch,
):
    _clear_nvfp4_backend_env(monkeypatch)
    _mock_backend_support(monkeypatch, {NvFp4MoeBackend.FLASHINFER_CUTLASS})

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("auto"), None, None)

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTLASS


def test_explicit_cutlass_backend_is_respected(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.VLLM_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("cutlass"), None, None)

    assert backend == NvFp4MoeBackend.VLLM_CUTLASS


def test_explicit_flashinfer_cutlass_backend_is_respected(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.VLLM_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(
        _config("flashinfer_cutlass"), None, None
    )

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTLASS


def test_flashinfer_only_env_uses_flashinfer_order(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            NvFp4MoeBackend.VLLM_CUTLASS,
            NvFp4MoeBackend.FLASHINFER_CUTLASS,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("auto"), None, None)

    assert backend == NvFp4MoeBackend.FLASHINFER_CUTEDSL


def test_flashinfer_disabled_env_removes_flashinfer_backends(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "0")
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
            NvFp4MoeBackend.VLLM_CUTLASS,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("auto"), None, None)

    assert backend == NvFp4MoeBackend.VLLM_CUTLASS


def test_force_marlin_env_still_forces_marlin(monkeypatch):
    _clear_nvfp4_backend_env(monkeypatch)
    monkeypatch.setenv("VLLM_TEST_FORCE_FP8_MARLIN", "1")
    _mock_backend_support(
        monkeypatch,
        {
            NvFp4MoeBackend.MARLIN,
            NvFp4MoeBackend.FLASHINFER_TRTLLM,
        },
    )

    backend, _ = nvfp4.select_nvfp4_moe_backend(_config("auto"), None, None)

    assert backend == NvFp4MoeBackend.MARLIN
