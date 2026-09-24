from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import torch

Request = TypeVar("Request")
Result = TypeVar("Result")


def _move_to_device(value: Any, device: torch.device, non_blocking: bool) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device, non_blocking) for item in value]
    if isinstance(value, Mapping):
        return type(value)(
            (key, _move_to_device(item, device, non_blocking))
            for key, item in value.items()
        )
    return value


class HIPInferencePool:
    """Run independent inference requests concurrently across visible HIP GPUs.

    ``worker_factory`` is called once per device on that device's dedicated
    thread. It should construct or move the model there and return a callable
    accepting one request. Requests are assigned round-robin and results retain
    input order.
    """

    def __init__(
        self,
        worker_factory: Callable[[torch.device], Callable[[Request], Result]],
        devices: Iterable[int | torch.device] | None = None,
        *,
        move_inputs: bool = True,
        output_device: int | str | torch.device | None = None,
        non_blocking: bool = False,
    ) -> None:
        if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
            raise RuntimeError("HIPInferencePool requires a PyTorch ROCm/HIP runtime")

        if devices is None:
            resolved_devices = [
                torch.device("cuda", index)
                for index in range(torch.cuda.device_count())
            ]
        else:
            resolved_devices = [
                (
                    torch.device("cuda", device)
                    if isinstance(device, int)
                    else torch.device(device)
                )
                for device in devices
            ]
        if not resolved_devices:
            raise RuntimeError("HIPInferencePool requires at least one visible HIP GPU")
        if any(device.type != "cuda" for device in resolved_devices):
            raise ValueError("HIPInferencePool devices must be HIP GPU devices")

        self.devices = tuple(resolved_devices)
        self.move_inputs = move_inputs
        self.output_device = (
            torch.device(output_device) if output_device is not None else None
        )
        self.non_blocking = non_blocking
        self._executors = [
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"hip-inference-{device.index}"
            )
            for device in self.devices
        ]
        try:
            self._workers = [
                executor.submit(self._create_worker, worker_factory, device).result()
                for executor, device in zip(self._executors, self.devices, strict=True)
            ]
        except BaseException:
            self.shutdown(wait=True, cancel_futures=True)
            raise

    @staticmethod
    def _create_worker(
        worker_factory: Callable[[torch.device], Callable[[Request], Result]],
        device: torch.device,
    ) -> Callable[[Request], Result]:
        with torch.cuda.device(device):
            return worker_factory(device)

    def _run(
        self,
        worker: Callable[[Request], Result],
        request: Request,
        device: torch.device,
    ) -> Result:
        with torch.cuda.device(device):
            if self.move_inputs:
                request = _move_to_device(request, device, self.non_blocking)
            result = worker(request)
            if self.output_device is not None:
                result = _move_to_device(result, self.output_device, self.non_blocking)
            return result

    def map(self, requests: Iterable[Request]) -> list[Result]:
        """Process requests across the pool and return results in input order."""
        futures: list[Future[Result]] = []
        for index, request in enumerate(requests):
            worker_index = index % len(self.devices)
            futures.append(
                self._executors[worker_index].submit(
                    self._run,
                    self._workers[worker_index],
                    request,
                    self.devices[worker_index],
                )
            )
        return [future.result() for future in futures]

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Release all worker threads after their queued requests finish."""
        for executor in self._executors:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> HIPInferencePool:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown()
