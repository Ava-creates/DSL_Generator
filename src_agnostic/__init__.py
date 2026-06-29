"""Domain-agnostic DSL Generator pipeline.

This package mirrors :mod:`src` but routes every domain-specific decision
through a :class:`domains.base.DomainAdapter`. The existing :mod:`src`
package continues to work unchanged for backwards compatibility.
"""
