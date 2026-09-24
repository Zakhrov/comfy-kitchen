from contextlib import nullcontext

import pytest
import torch

from comfy_kitchen.hip_inference import HIPInferencePool


def _mock_hip_runtime(monkeypatch, device_count: int = 2) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.version, "hip", "test")


def test_inference_pool_distributes_requests_and_preserves_order(monkeypatch):
    _mock_hip_runtime(monkeypatch)
    created_devices = []

    def worker_factory(device):
        created_devices.append(device.index)
        return lambda request: (device.index, request * 2)

    with HIPInferencePool(worker_factory, move_inputs=False) as pool:
        results = pool.map([1, 2, 3, 4, 5])

    assert created_devices == [0, 1]
    assert results == [(0, 2), (1, 4), (0, 6), (1, 8), (0, 10)]


def test_inference_pool_requires_hip_runtime(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        HIPInferencePool(lambda _device: lambda request: request)


def test_inference_pool_rejects_non_gpu_devices(monkeypatch):
    _mock_hip_runtime(monkeypatch)

    with pytest.raises(ValueError, match="HIP GPU"):
        HIPInferencePool(lambda _device: lambda request: request, devices=["cpu"])
