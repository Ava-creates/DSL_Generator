"""Domain plugin package.

Import ``get_adapter`` from ``domains.registry`` to obtain a
:class:`~domains.base.DomainAdapter` instance by name.
"""

from domains.base import DomainAdapter, DomainSpec, EnvLike, EnvFactoryLike
from domains.registry import get_adapter, list_domains, register_domain

__all__ = [
    "DomainAdapter",
    "DomainSpec",
    "EnvLike",
    "EnvFactoryLike",
    "get_adapter",
    "list_domains",
    "register_domain",
]
