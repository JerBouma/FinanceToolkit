"""Finance Toolkit Initialization"""

import importlib
from typing import TYPE_CHECKING, Any

# The classes are imported when first used (PEP 562), so that importing one of them does
# not load every module of the Finance Toolkit and the libraries those need.
_CLASSES = {
    "Toolkit": "financetoolkit.toolkit_controller",
    "Economics": "financetoolkit.economics.economics_controller",
    "FixedIncome": "financetoolkit.fixedincome.fixedincome_controller",
    "Discovery": "financetoolkit.discovery.discovery_controller",
    "Portfolio": "financetoolkit.portfolio.portfolio_controller",
}

__all__ = ["Discovery", "Economics", "FixedIncome", "Portfolio", "Toolkit"]

if TYPE_CHECKING:
    from .discovery.discovery_controller import Discovery
    from .economics.economics_controller import Economics
    from .fixedincome.fixedincome_controller import FixedIncome
    from .portfolio.portfolio_controller import Portfolio
    from .toolkit_controller import Toolkit


def __getattr__(name: str) -> Any:
    """
    Imports a class of the Finance Toolkit when it is first used.

    Args:
        name (str): The name of the class, e.g. "Toolkit".

    Returns:
        Any: The class.

    Raises:
        AttributeError: When the Finance Toolkit has no such class.
    """
    if name in _CLASSES:
        value = getattr(importlib.import_module(_CLASSES[name]), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'financetoolkit' has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Lists the classes of the Finance Toolkit with the module's other attributes.

    Returns:
        list[str]: The attribute names.
    """
    return sorted(set(globals()) | set(_CLASSES))
