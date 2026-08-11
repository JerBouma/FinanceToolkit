"""
Dynamic tool registration for the Finance Toolkit MCP Server — Router Pattern.

Consolidates 500+ individual methods into 22 categorical master tools, each
accepting an indicator parameter (Literal enum) that routes to the
correct underlying method.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import types
import typing
from datetime import datetime, timedelta
from typing import Annotated, Any, NamedTuple

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field as PydanticField

from financetoolkit.mcp_server.coercion_model import (
    coerce_value,
    to_boolean,
    validate_date,
)
from financetoolkit.mcp_server.formatting_model import format_result
from financetoolkit.mcp_server.inspection_controller import ControllerInspector
from financetoolkit.mcp_server.provider_model import ToolkitProvider
from financetoolkit.utilities.dataframe_model import filter_columns as _filter_columns
from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()

# Types that are not representable in JSON Schema — dropped during normalization.
_OPAQUE_TYPES = {
    "list",
    "dict",
    "set",
    "tuple",
    "range",
    "ndarray",
    "Series",
    "DataFrame",
}

# Injected into each tool's Args docstring so FastMCP can surface them.
_PARAM_DESCRIPTIONS: dict[str, str] = {
    "indicator": (
        "Name of the specific metric to calculate, e.g. 'get_asset_turnover_ratio'. "
        "Required — omitting it returns the list of available indicators."
    ),
    "tickers": "Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOGL'.",
    "countries": "Comma-separated country names, e.g. 'United States,Germany,Japan'.",
    "start_date": "Start of the date range in YYYY-MM-DD format.",
    "end_date": "End of the date range in YYYY-MM-DD format.",
    "quarterly": "Return quarterly data instead of annual when True.",
    "benchmark_ticker": "Ticker used as the market benchmark, e.g. 'SPY' or '^GSPC'.",
    "growth": "Return period-over-period growth rates instead of absolute values.",
    "lag": "Number of periods to lag when computing growth rates.",
    "standardize": (
        "Return the Z-Score (standard score) instead of the raw values, i.e. how many "
        "standard deviations each value is from the mean of its own series. When "
        "combined with growth=True, the growth values are standardized instead of the "
        "raw values."
    ),
    "cumulative": (
        "Return the cumulative value compounded over time instead of the discrete "
        "value per period. Always rebased to start at 1 at the beginning of the "
        "selected date range."
    ),
    "rolling": (
        "Rolling window size in number of periods. When set, the metric is computed over "
        "a smoothly overlapping trailing window across the full history (e.g. period='monthly' "
        "and rolling=6 gives a rolling 6-month value) instead of one value per period, or "
        "(for economics indicators) a simple moving average used to smooth the raw series."
    ),
    "trailing": (
        "Trailing window size in number of periods. Sums the raw values over the trailing N "
        "periods (e.g. trailing=4 on quarterly data gives a trailing-4-quarter / TTM-style sum) "
        "instead of returning one value per period."
    ),
    "threshold_percentile": (
        "Only used when distribution='evt'. The percentile of losses above which the "
        "Generalized Pareto Distribution is fitted, e.g. 0.95 fits on the worst 5% of losses."
    ),
    "minimum_acceptable_return": (
        "The minimum acceptable return (MAR) threshold below which returns are considered "
        "downside, e.g. 0.0 for downside relative to a zero return."
    ),
    "days": "Number of calendar days used in day-count-based calculations.",
    "period": "Observation frequency, e.g. 'monthly', 'quarterly', or 'annual'.",
    "measure": "Sub-measure selector, e.g. 'M1', 'M2', or 'M3' for money supply.",
    "gmdb_source": (
        "Use the Global Macro Database as the data source when True, rather than the "
        "OECD. The two are independent providers with different country and period "
        "coverage; both return rates and ratios as decimal fractions."
    ),
    "inflation_adjusted": "Adjust nominal values for inflation when True.",
    "bond_price": "Clean price of the bond per 100 face value.",
    "coupon_rate": "Annual coupon rate as a decimal, e.g. 0.05 for 5 %.",
    "years_to_maturity": "Years remaining until the bond matures.",
    "maturities": "Comma-separated bond maturity labels, e.g. '3month,2year,10year'.",
    "strike_rate": "Option strike price.",
    "factors_to_calculate": "Comma-separated factor names to include in the calculation.",
    "growth_rate": "Assumed constant growth rate as a decimal.",
    "perpetual_growth_rate": "Terminal (perpetual) growth rate used in DCF models.",
    "weighted_average_cost_of_capital": "WACC as a decimal, e.g. 0.09 for 9 %.",
    "show_columns": (
        "Comma-separated names to filter the output. "
        "For historical data use the key names visible in any response record "
        "(e.g. 'Close,Volume,Return'). "
        "For financial statements use the 'metric' field values from the response "
        "(e.g. 'Revenue,Net Income,EBITDA'). "
        "Call the tool once without this parameter to see all available names, "
        "then repeat with show_columns to reduce response size and token usage."
    ),
}


# Beyond this many characters the per-indicator default listing stops helping.
_DISPUTED_DEFAULT_DETAIL_LIMIT = 320


def _annotate_with_description(name: str, ann: Any, suffix: str = "") -> Any:
    """Wrap *ann* in ``Annotated[ann, Field(description=...)]`` for JSON schema output.

    Pydantic reads the ``Field(description=...)`` metadata and includes it in the
    ``properties[name].description`` field of the generated JSON schema, which is
    what Smithery (and other MCP clients) use to display parameter descriptions.

    Args:
        name (str): The parameter name, used to look up the shared description.
        ann (Any): The annotation to wrap.
        suffix (str): Extra text appended to the description, used to spell out
            defaults that differ between the indicators of a router group.

    Returns:
        Any: The annotated type carrying the description metadata.
    """
    desc = _PARAM_DESCRIPTIONS.get(name, f"Value for {name}.")

    if suffix:
        desc = f"{desc} {suffix}"

    return Annotated[ann, PydanticField(description=desc)]  # type: ignore[valid-type]


def _describe_disputed_default(per_method: dict[str, Any]) -> str:
    """Describe a parameter whose default differs between indicators of a group.

    The tool cannot advertise one default for all of them, so the description has
    to carry what the schema cannot: which indicators require the parameter and
    what each of the others falls back to when it is left unset.

    Args:
        per_method (dict[str, Any]): Per indicator, the default it declares, with
            ``inspect.Parameter.empty`` standing for "required".

    Returns:
        str: A sentence to append to the parameter description.
    """
    required = sorted(
        name
        for name, default in per_method.items()
        if default is inspect.Parameter.empty
    )

    grouped: list[tuple[Any, list[str]]] = []

    for name, default in sorted(per_method.items()):
        if default is inspect.Parameter.empty:
            continue

        for value, names in grouped:
            if type(value) is type(default) and value == default:
                names.append(name)
                break
        else:
            grouped.append((default, [name]))

    grouped.sort(key=lambda item: (-len(item[1]), str(item[0])))

    sentences = ["Leave unset to use the default of the indicator you selected."]

    if required:
        sentences.append(f"Required by: {', '.join(required)}.")

    detail = "; ".join(f"{value!r} for {', '.join(names)}" for value, names in grouped)

    if detail and len(detail) <= _DISPUTED_DEFAULT_DETAIL_LIMIT:
        sentences.append(f"Defaults are {detail}.")
    elif detail:
        sentences.append("Defaults differ between indicators.")

    return " ".join(sentences)


def _simplify_annotation(ann: Any) -> Any:
    """Reduce complex union/collection annotations to JSON-schema-compatible scalars.

    FastMCP generates "unknown" schema entries for union types that contain
    collections (list, dict, range, ndarray) or mixed numeric types. This
    function strips those parts out and returns the simplest scalar equivalent:

    - ``int | list[int]``                → ``int``
    - ``int | float | None``             → ``float | None``
    - ``float | range | list | None``    → ``float | None``
    - ``list[str] | str | None``         → ``str | None``
    - ``float | list | dict[str,float]`` → ``float``
    """
    if ann is inspect.Parameter.empty or ann is None:
        return str

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    # Bare list[T] → T
    if origin is list:
        return args[0] if args else str

    # A bare opaque type outside a union falls back to str, as the union case does.
    if origin is None:
        name = getattr(ann, "__name__", "")
        if name in _OPAQUE_TYPES:
            return str

    # Union types (both `X | Y` and `Union[X, Y]`)
    is_union = origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(ann, types.UnionType)
    )
    if not is_union:
        return ann

    has_none = type(None) in args
    scalars: list[Any] = []
    for a in args:
        if a is type(None):
            continue
        a_origin = typing.get_origin(a)
        # Drop generic collections (list[str], dict[str, float], …)
        if a_origin in (list, dict, set, tuple):
            continue
        # Drop bare opaque types by name
        name = getattr(a, "__name__", "")
        if name in _OPAQUE_TYPES:
            continue
        scalars.append(a)

    if not scalars:
        return str | None if has_none else str

    # Merge int + float → float (float is the wider type)
    if int in scalars and float in scalars:
        scalars = [x for x in scalars if x is not int]

    scalar = scalars[0]
    return scalar | None if has_none else scalar


class RouterGroupSpec(NamedTuple):
    """
    Immutable specification for a single router group tool.

    Each instance describes one categorical master tool — the metadata needed
    to discover controller methods, build a dispatcher wrapper, and register
    the tool on the FastMCP instance.

    Args:
        tool_name (str): Unique name for the FastMCP tool (e.g. "get_valuation_ratios").
        display_name (str): Human-readable label shown in tool descriptions.
        module_name (str): Key into module_class_map identifying the controller class.
        category (str): Dispatch category ("ticker", "standalone", "toolkit",
            "discovery", or "mixed").
        collect_method (str | None): Optional name of a collect_* method on the
            controller class whose source is parsed to discover the ordered list of
            get_* methods.
        method_override (list[str] | None): Explicit ordered list of method names.
            When set, method discovery is bypassed entirely.
        method_to_module (dict[str, dict[str, str]] | None): Per-method routing table
            used by Mixed groups, mapping method names to {"module": ..., "category": ...}
            dicts.
        index_category (str | None): Category key used when inserting the tool into
            the tool index. Defaults to module_name when None.
        description (str | None): Optional custom description for the tool. When None,
            a default description is generated from display_name and the method list.
    """

    tool_name: str
    display_name: str
    module_name: str
    category: str
    collect_method: str | None = None
    method_override: list[str] | None = None
    method_to_module: dict[str, dict[str, str]] | None = None
    index_category: str | None = None
    description: str | None = None


class ToolRegistry:
    """Dynamically builds and registers categorical master tools on a FastMCP instance.

    Reads router-group specifications from ``tool_groups`` in the config dict,
    resolves ``module_class_map`` string paths to actual class objects via
    :meth:`_resolve_class_map`, introspects the relevant Finance Toolkit controller
    classes via :class:`ControllerInspector`, and calls ``mcp.add_tool()`` for each
    group. The resulting tool index can be queried via :meth:`get_tool_index`.
    """

    def __init__(
        self,
        mcp: FastMCP,
        provider: ToolkitProvider,
        inspector: ControllerInspector,
        module_class_map: dict[str, str],
        skip_methods: list[str],
        direct_methods: list[str],
        tool_groups: list[dict[str, Any]],
        blocked_periods: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialise the registry with the FastMCP instance and shared subsystems.

        The ``module_class_map`` values may be either fully-qualified dotted class
        path strings (as read from ``config.yaml``) or already-resolved types.
        String paths are resolved to actual class objects at construction time via
        :meth:`_resolve_class_map`.

        Args:
            mcp (FastMCP): The FastMCP server instance to register tools on.
            provider (ToolkitProvider): Provider used to route actual data calls.
            inspector (ControllerInspector): Inspection helper for building tool
                signatures and discovering group methods.
            module_class_map (dict[str, str]): Mapping of module names to
                fully-qualified class path strings
                (e.g. ``"financetoolkit.ratios.ratios_controller.Ratios"``).
                Values may also be pre-resolved class objects.
            skip_methods (list[str]): List of method names to ignore when discovering
                group methods.
            direct_methods (list[str]): List of method names that should be directly
                exposed as top-level tools instead of via router groups.
            tool_groups (list[dict[str, Any]]): List of router group specifications
                from the config dict.
        """
        self._mcp = mcp
        self._provider = provider
        self._inspector = inspector
        self._tool_index: dict[str, list[dict[str, str]]] = {}
        self._module_class_map: dict[str, type] = self._resolve_class_map(
            module_class_map
        )
        self._skip_methods = frozenset(skip_methods)
        self._direct_methods = frozenset(direct_methods)
        self._tool_groups = tool_groups
        self._blocked_periods: dict[str, frozenset[str]] = {
            tool: frozenset(periods)
            for tool, periods in (blocked_periods or {}).items()
        }

    @staticmethod
    def _resolve_class_map(class_map: dict[str, str]) -> dict[str, type]:
        """Resolve a mapping of module names to fully-qualified class path strings
        into a mapping of module names to actual class objects.

        Each value in ``class_map`` should be a dotted import path such as
        ``"financetoolkit.ratios.ratios_controller.Ratios"``. Values that are
        already class objects (i.e. already resolved) are passed through unchanged.
        Entries whose class path cannot be imported or whose attribute cannot be
        found are skipped with a warning, so a partial map is returned rather than
        raising an exception.

        Args:
            class_map (dict[str, str]): Mapping of short module keys to fully-qualified
                class path strings (or already-resolved types).

        Returns:
            dict[str, type]: Mapping of module keys to resolved class objects.
        """
        resolved: dict[str, type] = {}
        for key, class_path in class_map.items():
            if isinstance(class_path, type):
                resolved[key] = class_path
                continue
            try:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                resolved[key] = getattr(module, class_name)
            except (ImportError, AttributeError, ValueError) as exc:
                logger.warning(
                    "Failed to resolve class '%s' for key '%s': %s",
                    class_path,
                    key,
                    exc,
                )
        return resolved

    def get_tool_index(self) -> dict[str, list[dict[str, str]]]:
        """Return the populated tool index mapping categories to registered tools.

        Returns:
            dict[str, list[dict[str, str]]]: Mapping of category name to a list
                of ``{"tool": str, "description": str}`` dicts for every tool
                registered under that category.
        """
        return self._tool_index

    def build_router_group_specs(self) -> list[RouterGroupSpec]:
        """Build the list of ``RouterGroupSpec`` objects from the config.

        Reads the ``tool_groups`` list from the config dict and converts each
        entry into a typed ``RouterGroupSpec`` NamedTuple. Groups configured with
        ``use_direct_methods: true`` have their ``method_override`` set to
        the sorted list of ``direct_methods`` from the config.

        Returns:
            list[RouterGroupSpec]: Ordered list of group specifications ready for
                ``_register_router_group``.
        """
        specifications = []
        toolkit_methods = sorted(self._direct_methods)
        for raw in self._tool_groups:
            method_override = raw.get("method_override")
            if raw.get("use_direct_methods"):
                method_override = toolkit_methods
            specifications.append(
                RouterGroupSpec(
                    tool_name=raw["tool_name"],
                    display_name=raw["display_name"],
                    module_name=raw.get("module_name", ""),
                    category=raw["category"],
                    collect_method=raw.get("collect_method"),
                    method_override=method_override,
                    method_to_module=raw.get("method_to_module"),
                    index_category=raw.get("index_category"),
                    description=raw.get("description"),
                )
            )
        return specifications

    def _build_router_wrapper(
        self,
        spec: RouterGroupSpec,
        cls: type | None,
        group_methods: list[str],
        extra_params: list[inspect.Parameter],
        method_to_cls: dict[str, type] | None = None,
        method_dispatch: dict[str, tuple[str, str]] | None = None,
    ) -> Any:
        """Build the ``**kwargs`` callable that powers a router group tool.

        Constructs a closure (``wrapper``) that dispatches to the correct
        Finance Toolkit method based on the ``indicator`` keyword argument, coerces
        inputs, calls the provider, and formats the result as Markdown. The
        closure's ``__signature__`` is replaced with a proper FastMCP-compatible
        signature derived from the group's common and extra parameters.

        Args:
            spec (RouterGroupSpec): Specification for the router group being built.
            cls (type | None): Primary controller class for the group. ``None``
                for ``Mixed`` groups where ``method_to_cls`` is used instead.
            group_methods (list[str]): Ordered list of indicator method names for
                this group.
            extra_params (list[inspect.Parameter]): Extra method-specific parameters
                to append to the tool signature beyond the common params.
            method_to_cls (dict[str, type] | None): Per-method class override used
                for ``Mixed`` groups.
            method_dispatch (dict[str, tuple[str, str]] | None): Per-method
                ``(module_name, category)`` routing table for ``Mixed`` groups.

        Returns:
            Any: A callable with a replaced ``__signature__`` ready to be passed
                to ``mcp.add_tool()``.
        """
        tool_name = spec.tool_name
        module_name = spec.module_name
        category = spec.category
        inspector = self._inspector
        provider = self._provider
        blocked_periods_for_tool = self._blocked_periods.get(tool_name, frozenset())

        if method_to_cls:
            method_param_names = {
                m: inspector.get_method_param_names(method_to_cls.get(m) or cls, m)
                for m in group_methods
            }
        else:
            method_param_names = {
                m: inspector.get_method_param_names(cls, m) for m in group_methods
            }
        param_meta = [(p.name, p.annotation, p.default) for p in extra_params]
        all_indicators = group_methods

        # Parameters the indicators in this group disagree about, or that some of
        # them require. One schema default cannot describe all of them, so they are
        # advertised as unset and only forwarded when the caller actually supplies
        # a value, leaving each indicator's own default to apply.
        parameter_defaults = inspector.collect_group_parameter_defaults(
            cls, group_methods, spec.collect_method, method_to_cls
        )
        disputed_params = inspector.find_disputed_parameters(parameter_defaults)

        def wrapper(**kwargs):
            """
            Dispatch a router group tool call to the correct Finance Toolkit method.

            Resolves the indicator name, coerces all typed parameters, validates
            required inputs (tickers, period), routes the call through the provider,
            and returns a formatted Markdown string suitable for LLM consumption.
            """
            raw_indicator = kwargs.pop("indicator", None)
            if not raw_indicator:
                return (
                    f"Please specify an `indicator`. "
                    f"Available for `{tool_name}`: "
                    + ", ".join(f"`{m}`" for m in all_indicators)
                )
            candidate = raw_indicator
            if (
                not candidate.startswith("get_")
                and f"get_{candidate}" in method_param_names
            ):
                candidate = f"get_{candidate}"
            if candidate not in method_param_names:
                close = difflib.get_close_matches(
                    candidate, all_indicators, n=3, cutoff=0.6
                )
                suggestion = (
                    (" Did you mean: " + ", ".join(close) + "?") if close else ""
                )
                return (
                    f"Unknown indicator `{raw_indicator}` for `{tool_name}`."
                    f"{suggestion}\n"
                    "Available: " + ", ".join(all_indicators)
                )
            method_name = candidate

            raw_tickers = kwargs.pop("tickers", None)
            tickers = (
                [t.strip().upper() for t in str(raw_tickers).split(",") if t.strip()]
                if raw_tickers
                else None
            )
            raw_countries = kwargs.pop("countries", None)
            countries = (
                [c.strip() for c in str(raw_countries).split(",") if c.strip()]
                if raw_countries
                else None
            )

            # Return an actionable error rather than a confusing AttributeError later.
            effective_category = category
            if method_dispatch and method_name in method_dispatch:
                _, effective_category = method_dispatch[method_name]
            if (
                effective_category
                in (
                    inspector.categories.get("ticker", "ticker"),
                    inspector.categories.get("toolkit", "toolkit"),
                )
                and not tickers
            ):
                return (
                    f"`{tool_name}` (`{method_name}`) requires a `tickers` parameter. "
                    "Please provide one or more ticker symbols, e.g. `tickers='AAPL'` "
                    "or `tickers='AAPL,MSFT'`."
                )

            quarterly = to_boolean(kwargs.pop("quarterly", False))
            start_date = validate_date(
                kwargs.pop("start_date", inspector.start_date) or inspector.start_date,
                default_date=(
                    datetime.now() - timedelta(days=90 * 5 if quarterly else 365 * 5)
                ).strftime("%Y-%m-%d"),
            )
            end_date = validate_date(
                kwargs.pop("end_date", inspector.end_date) or inspector.end_date,
                default_date=datetime.now().strftime("%Y-%m-%d"),
            )
            benchmark_ticker = kwargs.pop(
                "benchmark_ticker", inspector.benchmark_ticker
            )

            # show_columns is handled entirely here — never forwarded to methods
            raw_show_columns = kwargs.pop("show_columns", None)
            show_columns = (
                [c.strip() for c in str(raw_show_columns).split(",") if c.strip()]
                if raw_show_columns
                else None
            )

            accepted_params = method_param_names.get(method_name, set())
            method_kwargs = {}
            for pname, pann, _ in param_meta:
                if pname in kwargs:
                    val = kwargs.pop(pname)
                    if pname not in accepted_params:
                        continue
                    # A disputed parameter is advertised as unset, so an unset value
                    # means "no preference" and must reach the indicator's own
                    # default rather than another indicator's.
                    if pname in disputed_params and val in (None, ""):
                        continue
                    method_kwargs[pname] = coerce_value(val, pann)

            if kwargs:
                logger.warning(
                    "Unknown parameter(s) passed to %s (%s) and ignored: %s",
                    tool_name,
                    method_name,
                    ", ".join(kwargs.keys()),
                )

            # Validate that the requested period (if any) is not blocked for this tool
            if blocked_periods_for_tool and "period" in method_kwargs:
                requested_period = str(method_kwargs["period"]).lower()
                if requested_period in blocked_periods_for_tool:
                    allowed = ["weekly", "monthly", "quarterly", "yearly"]
                    return (
                        f"`{tool_name}` (`{method_name}`) does not support "
                        f"`period='{requested_period}'`. "
                        f"Please use one of: {', '.join(allowed)}."
                    )

            if method_dispatch and method_name in method_dispatch:
                dispatch_module, dispatch_category = method_dispatch[method_name]
            else:
                dispatch_module, dispatch_category = module_name, category

            try:
                result = provider.call_method(
                    module_name=dispatch_module,
                    method_name=method_name,
                    category=dispatch_category,
                    tickers=tickers,
                    countries=countries,
                    start_date=start_date,
                    end_date=end_date,
                    quarterly=quarterly,
                    benchmark_ticker=benchmark_ticker,
                    **method_kwargs,
                )
                if show_columns is not None:
                    result = _filter_columns(result, show_columns)
                notes: list[str] = []
                if dispatch_category in ("ticker", "toolkit") and tickers:
                    notes = provider.get_transformation_notes(
                        tickers=tickers,
                        start_date=start_date,
                        end_date=end_date,
                        quarterly=quarterly,
                        benchmark_ticker=benchmark_ticker,
                        api_key=provider._api_key,
                        fred_api_key=provider._fred_api_key,
                    )
                formatted = format_result(result, notes=notes or None)
                return formatted
            except (ValueError, KeyError) as exc:
                return f"Invalid input for `{tool_name}` (`{method_name}`): {exc}"
            except TypeError as exc:
                return f"Parameter error for `{tool_name}` (`{method_name}`): {exc}"
            except ConnectionError as exc:
                return f"API connection failed: {exc}"
            except Exception as exc:
                logger.warning(
                    "Tool %s (%s) failed: %s",
                    tool_name,
                    method_name,
                    exc,
                    exc_info=True,
                )
                return (
                    f"`{tool_name}` (`{method_name}`) failed with error: "
                    f"{type(exc).__name__}: {exc}"
                )

        P = inspect.Parameter
        POS = P.POSITIONAL_OR_KEYWORD
        indicator_choices = group_methods
        indicator_ann = (
            typing.Literal[tuple(indicator_choices)] if indicator_choices else str
        )
        indicator_default = P.empty
        indicator_param = P(
            "indicator",
            POS,
            default=indicator_default,
            annotation=_annotate_with_description("indicator", indicator_ann),
        )

        mixed_cats = None
        if category == inspector.categories["mixed"] and method_dispatch:
            mixed_cats = {cat for _, cat in method_dispatch.values()}

        common = inspector.build_common_signature_params(
            category, mixed_categories=mixed_cats
        )
        sig_params = [indicator_param] + [
            P(
                p.name,
                POS,
                default=p.default,
                annotation=_annotate_with_description(p.name, p.annotation),
            )
            for p in common
        ]

        for p in extra_params:
            ann = (
                _simplify_annotation(p.annotation)
                if p.annotation is not P.empty
                else str
            )

            if p.name in disputed_params:
                # Advertised as unset with the per-indicator defaults spelled out,
                # because no single value is correct for the whole group.
                default = None
                annotation = _annotate_with_description(
                    p.name,
                    typing.Optional[ann],  # noqa: UP045
                    suffix=_describe_disputed_default(parameter_defaults[p.name]),
                )
            else:
                default = p.default if p.default is not P.empty else ""
                annotation = _annotate_with_description(p.name, ann)

            sig_params.append(P(p.name, POS, default=default, annotation=annotation))

        # Appended last so it does not disturb positional ordering of other params.
        sig_params.append(
            P(
                "show_columns",
                POS,
                default=None,
                annotation=_annotate_with_description("show_columns", str | None),
            )
        )

        wrapper.__signature__ = inspect.Signature(sig_params, return_annotation=str)
        wrapper.__annotations__ = {p.name: p.annotation for p in sig_params}
        wrapper.__annotations__["return"] = str

        return wrapper

    def _register_router_group(self, spec: RouterGroupSpec) -> bool:
        """Build and register a single router group tool on the FastMCP instance.

        Resolves the controller class(es) for the group, discovers or applies the
        method list, builds the router wrapper via ``_build_router_wrapper``, and
        calls ``mcp.add_tool()``. Also populates the internal tool index.

        Args:
            spec (RouterGroupSpec): Specification for the router group to register.

        Returns:
            bool: ``True`` if the tool was successfully registered, ``False`` if
                the group was skipped due to a missing class, empty method list, or
                an exception during tool construction.
        """
        method_to_cls = None
        method_dispatch = None
        cls = None

        if spec.category == self._inspector.categories["mixed"]:
            if not spec.method_to_module:
                logger.warning(
                    f"Mixed group '{spec.tool_name}' has no method_to_module, skipping"
                )
                return False
            method_to_cls = {}
            method_dispatch = {}
            for mname, info in spec.method_to_module.items():
                mod = info.get("module", "")
                cat = info.get("category", "")
                target_cls = self._module_class_map.get(mod)
                if target_cls is None:
                    logger.warning(
                        f"No class for module '{mod}' in mixed group '{spec.tool_name}' — skipping method"
                    )
                    continue
                method_to_cls[mname] = target_cls
                method_dispatch[mname] = (mod, cat)
            group_methods = (
                list(spec.method_override)
                if spec.method_override is not None
                else sorted(method_to_cls.keys())
            )
            extra_params = self._inspector.collect_group_extra_params(
                None, group_methods, spec.collect_method, method_to_cls=method_to_cls
            )
        else:
            cls = self._module_class_map.get(spec.module_name)
            if cls is None:
                logger.warning(
                    "No class found for module '%s' — skipping", spec.module_name
                )
                return False
            group_methods = (
                list(spec.method_override)
                if spec.method_override is not None
                else self._inspector.discover_group_methods(
                    cls, spec.collect_method, self._skip_methods
                )
            )
            extra_params = self._inspector.collect_group_extra_params(
                cls, group_methods, spec.collect_method
            )
            method_to_cls = {m: cls for m in group_methods}

        if not group_methods:
            logger.warning(f"No methods found for group '{spec.tool_name}' — skipping")
            return False

        method_list = ", ".join(group_methods)
        if spec.description:
            description = f"{spec.description}\n\nAvailable indicators: {method_list}."
        else:
            description = (
                f"{spec.display_name}. Set `indicator` to one of: {method_list}."
            )
        description = description[:1500]

        try:
            fn = self._build_router_wrapper(
                spec,
                cls,
                group_methods,
                extra_params,
                method_to_cls=method_to_cls,
                method_dispatch=method_dispatch,
            )
            fn.__name__ = spec.tool_name
            fn.__doc__ = description
            self._mcp.add_tool(
                fn,
                name=spec.tool_name,
                description=description,
                annotations=ToolAnnotations(
                    title=spec.display_name,
                    readOnlyHint=True,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
            index_cat = spec.index_category or spec.module_name
            if index_cat not in self._tool_index:
                self._tool_index[index_cat] = []
            self._tool_index[index_cat].append(
                {"tool": spec.tool_name, "description": description}
            )
            return True
        except Exception as exc:
            logger.warning(f"Failed to register group '{spec.tool_name}': {exc}")
            return False

    def register_all_tools(self) -> int:
        """Register every router group defined in the config.

        Iterates over all ``RouterGroupSpec`` objects produced by
        ``build_router_group_specs()`` and registers each via
        ``_register_router_group()``. Logs the outcome of each group.

        Returns:
            int: Total number of router group tools successfully registered.
        """
        total = 0
        for spec in self.build_router_group_specs():
            registered = self._register_router_group(spec)
            logger.info(
                f"Registered router group '{spec.tool_name}' ({spec.module_name}) → {registered} tool(s)"
            )
            total += registered
        logger.info(f"Total master tools registered: {total}")
        return total
