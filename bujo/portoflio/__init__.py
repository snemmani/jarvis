"""Portoflio package.

This package provides Portfolio management helpers.
"""

__all__ = ["PortfolioManager"]


def __getattr__(name):
    if name == "PortfolioManager":
        from .manage import PortfolioManager
        return PortfolioManager
    raise AttributeError(name)
