"""Registry mapping domain names to :class:`~domains.base.DomainAdapter` instances.

Adapters are lazily imported so users can pick a domain at CLI time without
paying the import cost (or risk of missing dependencies) of the other.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from domains.base import DomainAdapter

_ADAPTER_FACTORIES: Dict[str, Callable[..., DomainAdapter]] = {}


def register_domain(name: str, factory: Callable[..., DomainAdapter]) -> None:
    """Register a domain adapter factory under ``name``."""
    _ADAPTER_FACTORIES[name.lower()] = factory


def _default_factories() -> Dict[str, Callable[..., DomainAdapter]]:
    def _craft_factory(**kwargs: Any) -> DomainAdapter:
        from domains.craft.adapter import CraftAdapter

        return CraftAdapter(**kwargs)

    def _crafter_factory(**kwargs: Any) -> DomainAdapter:
        from domains.crafter.adapter import CrafterAdapter

        return CrafterAdapter(**kwargs)

    return {
        "craft": _craft_factory,
        "crafter": _crafter_factory,
    }


for _name, _factory in _default_factories().items():
    _ADAPTER_FACTORIES.setdefault(_name, _factory)


def get_adapter(name: str, **kwargs: Any) -> DomainAdapter:
    """Instantiate the adapter registered under ``name``."""
    key = name.lower()
    if key not in _ADAPTER_FACTORIES:
        raise KeyError(
            f"Unknown domain '{name}'. Known domains: {sorted(_ADAPTER_FACTORIES)}"
        )
    return _ADAPTER_FACTORIES[key](**kwargs)


def list_domains() -> List[str]:
    return sorted(_ADAPTER_FACTORIES)
