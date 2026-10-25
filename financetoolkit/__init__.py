"""Finance Toolkit Initialization"""

import importlib
from typing import TYPE_CHECKING, Any

# The classes and functions are imported when first used (PEP 562), so that importing one
# of them does not load every module of the Finance Toolkit and the libraries those need.
_CLASSES = {
    "Toolkit": "financetoolkit.toolkit_controller",
    "Economics": "financetoolkit.economics.economics_controller",
    "Econometrics": "financetoolkit.econometrics.econometrics_controller",
    "FixedIncome": "financetoolkit.fixedincome.fixedincome_controller",
    "Discovery": "financetoolkit.discovery.discovery_controller",
    "Portfolio": "financetoolkit.portfolio.portfolio_controller",
    "PortfolioDefinition": "financetoolkit.portfolio.definition_model",
    "get_normalization_files": "financetoolkit.normalization_model",
}

__all__ = [
    "Discovery",
    "Econometrics",
    "Economics",
    "FixedIncome",
    "Portfolio",
    "PortfolioDefinition",
    "Toolkit",
    "get_normalization_files",
]

if TYPE_CHECKING:
    from .discovery.discovery_controller import Discovery
    from .econometrics.econometrics_controller import Econometrics
    from .economics.economics_controller import Economics
    from .fixedincome.fixedincome_controller import FixedIncome
    from .normalization_model import get_normalization_files
    from .portfolio.definition_model import PortfolioDefinition
    from .portfolio.portfolio_controller import Portfolio
    from .toolkit_controller import Toolkit


def __getattr__(name: str) -> Any:
    """
    Imports a class or function of the Finance Toolkit when it is first used.

    Args:
        name (str): The name of the class or function, e.g. "Toolkit".

    Returns:
        Any: The class or function.

    Raises:
        AttributeError: When the Finance Toolkit has no such class or function.
    """
    if name in _CLASSES:
        try:
            module = importlib.import_module(_CLASSES[name])
        except ModuleNotFoundError as error:
            if name != "Econometrics":
                raise
            raise ImportError(
                "The Econometrics module requires the optional 'econometrics' extra "
                f"({error.name} is missing). Install it with: pip install "
                "financetoolkit[econometrics]"
            ) from error
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'financetoolkit' has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Lists the classes and functions of the Finance Toolkit with the module's other attributes.

    Returns:
        list[str]: The attribute names.
    """
    return sorted(set(globals()) | set(_CLASSES))
