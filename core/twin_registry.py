"""Twin/adapter registry for the phys-MCP prototype."""

from __future__ import annotations

from typing import Iterable

from adapters.base_adapter import BaseAdapter
from descriptors.capability_schema import SubstrateDescriptor


class TwinRegistry:
    """Registry holding all currently available backend adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    def register(self, adapter: BaseAdapter, *, overwrite: bool = False) -> None:
        """Register one backend adapter."""
        backend_id = adapter.backend_id()
        if backend_id in self._adapters and not overwrite:
            raise ValueError(f"Backend '{backend_id}' is already registered.")
        self._adapters[backend_id] = adapter

    def unregister(self, backend_id: str) -> None:
        """Remove one backend adapter from the registry."""
        if backend_id not in self._adapters:
            raise KeyError(f"Backend '{backend_id}' is not registered.")
        del self._adapters[backend_id]

    def has_backend(self, backend_id: str) -> bool:
        """Return True if the backend is known."""
        return backend_id in self._adapters

    def get_adapter(self, backend_id: str) -> BaseAdapter:
        """Return the registered adapter for one backend."""
        try:
            return self._adapters[backend_id]
        except KeyError as exc:
            raise KeyError(f"Backend '{backend_id}' is not registered.") from exc

    def list_backend_ids(self) -> list[str]:
        """Return all registered backend identifiers."""
        return sorted(self._adapters.keys())

    def list_adapters(self) -> list[BaseAdapter]:
        """Return all registered adapters."""
        return [self._adapters[key] for key in self.list_backend_ids()]

    def list_descriptors(self) -> list[SubstrateDescriptor]:
        """Return descriptors for all registered backends."""
        return [adapter.describe() for adapter in self.list_adapters()]

    def iter_adapters(self) -> Iterable[BaseAdapter]:
        """Yield all adapters in stable backend-id order."""
        for backend_id in self.list_backend_ids():
            yield self._adapters[backend_id]

    def telemetry_snapshot(self) -> dict[str, dict[str, float | int | str | bool | None]]:
        """Return a snapshot of telemetry across all registered backends."""
        return {
            adapter.backend_id(): adapter.collect_telemetry()
            for adapter in self.iter_adapters()
        }

    def size(self) -> int:
        """Return the number of registered backends."""
        return len(self._adapters)
