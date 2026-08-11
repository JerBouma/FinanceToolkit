"""
Static class and method inspection utilities for the Finance Toolkit MCP server.

Provides helpers for introspecting controller classes, extracting and filtering
method parameters, unwrapping decorated functions, and building FastMCP tool
signatures. These utilities are used at server startup (registration time) and
carry no runtime dependencies on live controller instances.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta
from typing import Any

from financetoolkit.utilities.logger_model import get_logger

logger = get_logger()


class ControllerInspector:
    """
    Stateless inspector for Finance Toolkit controller classes.

    Encapsulates all static inspection logic needed at server startup to
    discover group methods, extract parameters, and build FastMCP tool
    signatures. Configuration is injected at construction time; all public
    read-only properties expose the relevant config values so that other
    components (e.g. :class:`ToolRegistry`) do not need to reach into
    private attributes.
    """

    def __init__(
        self,
        categories: dict[str, str],
        skip_params: frozenset[str] | list[str] | None = None,
        init_handled_params: frozenset[str] | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        benchmark_ticker: str = "SPY",
    ) -> None:
        """
        Initializes the ControllerInspector.

        Args:
            categories (dict[str, str]): Mapping of category role names (e.g.
                "ticker", "standalone", "mixed") to their string identifiers
                used in config and dispatch.
            skip_params (frozenset[str] | list[str] | None): Parameter names to
                exclude from every inspected method signature (e.g. "self",
                "progress_bar"). Accepts either a frozenset or a list; None
                is treated as an empty set.
            init_handled_params (frozenset[str] | list[str] | None): Parameter
                names consumed by the router wrapper and not forwarded to individual
                method calls (e.g. "tickers", "start_date"). Accepts either
                a frozenset or a list; None is treated as an empty set.
            start_date (str | None): Default start date string (YYYY-MM-DD) used
                for tool signature defaults. When None, defaults to five years
                before today.
            end_date (str | None): Default end date string (YYYY-MM-DD) used for
                tool signature defaults. When None, defaults to today.
            benchmark_ticker (str): Default benchmark ticker symbol used for tool
                signature defaults. Defaults to "SPY".
        """
        self._categories: dict[str, str] = categories
        self._skip_params: frozenset[str] = frozenset(skip_params or [])
        self._init_handled_params: frozenset[str] = frozenset(init_handled_params or [])
        self._benchmark_ticker: str = benchmark_ticker

        self._start_date: str = (
            start_date
            if start_date is not None
            else (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        )
        self._end_date: str = (
            end_date if end_date is not None else datetime.today().strftime("%Y-%m-%d")
        )

    def unwrap_to_class_source(self, fn: Any, cls: type) -> Any:
        """
        Return the function implementation whose source file matches that of the given class.

        Some decorators (for example, error-handling wrappers) do not set the
        __wrapped__ attribute, which prevents the standard inspect.unwrap or
        __wrapped__ chain from reaching the original implementation. This helper
        inspects a possibly wrapped callable and attempts to return the first callable
        whose __code__.co_filename equals the source file of cls.

        Args:
            fn (Any): A callable or wrapper to inspect.
            cls (type): A class (type) whose source file will be used to identify the
                target implementation.

        Returns:
            The callable object that originates from the same source file as
            cls if found, otherwise the original fn.
        """
        try:
            cls_file = inspect.getfile(cls)
        except (TypeError, OSError):
            return fn

        # Quick exit: already pointing at the right file.
        if hasattr(fn, "__code__") and fn.__code__.co_filename == cls_file:
            return fn

        # Standard __wrapped__ chain first.
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
            if hasattr(fn, "__code__") and fn.__code__.co_filename == cls_file:
                return fn

        # Fall back: search the closure for the original function.
        if hasattr(fn, "__closure__") and fn.__closure__:
            for cell in fn.__closure__:
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if (
                    callable(val)
                    and hasattr(val, "__code__")
                    and val.__code__.co_filename == cls_file
                ):
                    return val

        return fn

    def discover_group_methods(
        self,
        cls: type,
        collect_method: str | None,
        skip_methods: frozenset[str],
    ) -> list[str]:
        """
        Inspect a class to determine which of its ``get_`` instance methods
        should be considered part of a logical router group.

        If a specific collect_method name is provided, the function will attempt
        to unwrap that method to its original source (to cope with decorators
        that do not set __wrapped__) and then parse the source code for explicit
        ``self.get_*()`` calls. The order of discovery follows the order of
        appearance in the source, duplicates are removed while preserving order,
        and any methods listed in skip_methods are ignored.

        If source inspection fails (e.g. original source cannot be obtained)
        or yields no results, the function falls back to returning all public
        ``get_*`` functions declared on the class (sorted alphabetically),
        excluding any in skip_methods.

        Args:
            cls (type): The class to inspect for groupable ``get_`` methods.
            collect_method (str | None): Optional name of a method on cls whose
                source will be inspected for ``self.get_*()`` calls. If None, no
                source inspection is attempted and the fallback behavior is used.
            skip_methods (frozenset[str]): A set of method names to exclude from
                the result (e.g., utility or internal getters).

        Returns:
            list[str]: A list of method names that start with ``get_`` and belong
            to the router group. When collect_method source is used, the order
            reflects the order of calls in the source (deduplicated). Otherwise,
            all matching class methods are returned in sorted order.
        """
        if collect_method is not None:
            try:
                raw_fn = getattr(cls, collect_method)
                orig_fn = self.unwrap_to_class_source(raw_fn, cls)
                src = inspect.getsource(orig_fn)
                found = re.findall(r"self\.(get_[a-zA-Z_]+)\(", src)
                seen: set[str] = set()
                result: list[str] = []
                for m in found:
                    if m not in seen and m not in skip_methods:
                        seen.add(m)
                        result.append(m)
                if result:
                    return result
            except (OSError, AttributeError):
                pass  # Fall through to the generic approach

        return sorted(
            name
            for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
            if name.startswith("get_") and name not in skip_methods
        )

    def extract_extra_params(self, method: Any) -> list[inspect.Parameter]:
        """
        Return method parameters that are not handled by the wrapper.

        Inspects the signature of the provided callable and returns the list of
        parameters that should be considered "extra" for building a tool signature.
        Parameters that are consumed by the wrapper or by the Toolkit __init__
        (defined in SKIP_PARAMS and INIT_HANDLED_PARAMS) are excluded. Only
        positional-or-keyword and keyword-only parameters are returned.

        Args:
            method (Any): The callable whose signature will be inspected.

        Returns:
            list[inspect.Parameter]: Ordered list of inspect.Parameter objects
                representing the extra parameters. Returns an empty list if the
                signature cannot be obtained.
        """
        try:
            sig = inspect.signature(method)
        except (ValueError, TypeError):
            return []

        return [
            p
            for name, p in sig.parameters.items()
            if name not in self._skip_params
            and name not in self._init_handled_params
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

    def get_method_param_names(self, cls: type, method_name: str) -> set[str]:
        """
        Return the set of parameter names accepted by a method on a class.

        This inspects the callable attribute named ``method_name`` on ``cls`` and
        returns the set of parameter names from its signature. The helper is
        robust to missing attributes and to callables where inspect.signature
        may fail (for example, some builtins or heavily-wrapped/decorated callables).

        Args:
            cls (type): The class (or object) that should expose the method.
            method_name (str): The name of the method whose parameters will be inspected.

        Returns:
            set[str]: A set containing the parameter names accepted by the method.
                Returns an empty set if the method is not found or the signature
                cannot be obtained.
        """
        meth = getattr(cls, method_name, None)
        if meth is None:
            return set()
        try:
            return set(inspect.signature(meth).parameters.keys())
        except (ValueError, TypeError):
            return set()

    def collect_group_extra_params(
        self,
        cls: type | None,
        group_methods: list[str],
        collect_method: str | None,
        method_to_cls: dict[str, type] | None = None,
    ) -> list[inspect.Parameter]:
        """
        Return the union of extra parameters across all methods in the group.

        Collects parameters that are not handled by the wrapper (see self._skip_params
        and self._init_handled_params) from each method in the supplied group. If a
        per-method class mapping (method_to_cls) is provided, as is the case for
        Mixed groups, each method is inspected on its corresponding class;
        otherwise the supplied cls is used.

        First-seen wins for duplicate parameter names: the inspect.Parameter object
        from the first occurrence is preserved and subsequent duplicates are ignored.

        Args:
            cls (type | None): Default class to inspect. May be None when
                method_to_cls provides per-method class overrides (Mixed groups).
            group_methods (list[str]): List of method names that belong to the router group.
            collect_method (str | None): Optional collect_* method name to include in the union.
            method_to_cls (dict[str, type] | None): Optional per-method class override mapping.

        Returns:
            list[inspect.Parameter]: Ordered list of unique extra parameters suitable for
            constructing the FastMCP tool signature.
        """
        all_methods = list(group_methods)
        if collect_method:
            all_methods.append(collect_method)

        seen: dict[str, inspect.Parameter] = {}
        for meth_name in all_methods:
            target_cls = (method_to_cls or {}).get(meth_name) if method_to_cls else cls
            if target_cls is None:
                target_cls = cls
            if target_cls is None:
                continue
            meth_func = getattr(target_cls, meth_name, None)
            if meth_func is None:
                continue
            for p in self.extract_extra_params(meth_func):
                if p.name not in seen:
                    seen[p.name] = p
        return list(seen.values())

    def collect_group_parameter_defaults(
        self,
        cls: type | None,
        group_methods: list[str],
        collect_method: str | None,
        method_to_cls: dict[str, type] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Return, per extra parameter, the default each method in the group gives it.

        One router tool fronts every indicator in its group, so a parameter name
        shared by several of them can only carry one default in the generated
        schema. ``collect_group_extra_params`` resolves that by first-seen wins,
        which is fine for building the signature but hides a real disagreement:
        ``window`` defaults to 20 on the Chaikin Money Flow and to 252 on the
        New Highs / New Lows, and only one of those can be advertised. Recording
        every method's own default lets the caller decide what to do about it
        instead of silently applying whichever method happened to be inspected
        first.

        A method that requires the parameter is recorded with
        ``inspect.Parameter.empty``, so "required here" is distinguishable from
        "defaults to None here".

        Args:
            cls (type | None): Default class to inspect. May be None when
                method_to_cls provides per-method class overrides (Mixed groups).
            group_methods (list[str]): Method names that belong to the router group.
            collect_method (str | None): Optional collect_* method name to include.
            method_to_cls (dict[str, type] | None): Optional per-method class override.

        Returns:
            dict[str, dict[str, Any]]: Parameter name to a mapping of method name to
                the default that method declares for it.
        """
        all_methods = list(group_methods)

        if collect_method:
            all_methods.append(collect_method)

        defaults: dict[str, dict[str, Any]] = {}

        for meth_name in all_methods:
            target_cls = (method_to_cls or {}).get(meth_name) if method_to_cls else cls

            if target_cls is None:
                target_cls = cls
            if target_cls is None:
                continue

            meth_func = getattr(target_cls, meth_name, None)

            if meth_func is None:
                continue

            for p in self.extract_extra_params(meth_func):
                defaults.setdefault(p.name, {})[meth_name] = p.default

        return defaults

    @staticmethod
    def find_disputed_parameters(
        parameter_defaults: dict[str, dict[str, Any]],
    ) -> set[str]:
        """
        Return the parameters whose default the group cannot agree on.

        A parameter is disputed when two indicators in the same group declare
        different defaults for it, or when at least one indicator requires it and
        so has no default to advertise at all. Those are exactly the parameters
        where baking one value into the tool schema would answer a question the
        caller never asked.

        Args:
            parameter_defaults (dict[str, dict[str, Any]]): Per parameter, the
                default declared by each method, as returned by
                ``collect_group_parameter_defaults``.

        Returns:
            set[str]: The names of the disputed parameters.
        """
        disputed: set[str] = set()

        for name, per_method in parameter_defaults.items():
            values = list(per_method.values())

            if any(value is inspect.Parameter.empty for value in values):
                disputed.add(name)

                continue

            first = values[0]

            # Compared with `==` because a default may be an unhashable list.
            if any(
                type(value) is not type(first) or value != first for value in values[1:]
            ):
                disputed.add(name)

        return disputed

    def build_common_signature_params(
        self,
        category: str,
        mixed_categories: set[str] | None = None,
    ) -> list[inspect.Parameter]:
        """
        Build the common MCP-level parameters for a given category.

        Generates a list of inspect.Parameter objects used to define the exposed
        FastMCP tool signature for a router group. The returned parameters include
        common inputs such as tickers, countries, date range, quarterly flag, and
        benchmark_ticker. Which parameters are included depends on the group's
        routing category; for Mixed groups the mixed_categories set controls which
        common parameters are present.

        Args:
            category (str): Routing category (e.g. "ticker", "standalone",
                "toolkit", "discovery", or "mixed").
            mixed_categories (set[str] | None): For Mixed groups, the set of
                sub-categories present across the group's methods. Used to decide
                which common params (tickers, countries, dates, etc.) to include.

        Returns:
            list[inspect.Parameter]: Ordered list of common signature parameters.
        """
        P = inspect.Parameter
        POS = P.POSITIONAL_OR_KEYWORD
        params: list[inspect.Parameter] = []

        # For mixed groups, derive effective category set from sub-categories.
        effective: set[str] = (
            mixed_categories
            if (category == self._categories["mixed"] and mixed_categories)
            else {category}
        )

        if (
            self._categories.get("ticker") in effective
            or self._categories.get("toolkit") in effective
        ):
            params.append(P("tickers", POS, default="", annotation=str))

        if self._categories.get("standalone") in effective:
            params.append(P("countries", POS, default="", annotation=str))

        if self._categories.get("discovery") not in effective:
            params.extend(
                [
                    P("start_date", POS, default=self._start_date, annotation=str),
                    P("end_date", POS, default=self._end_date, annotation=str),
                    P("quarterly", POS, default=False, annotation=bool),
                ]
            )

        if (
            self._categories.get("ticker") in effective
            or self._categories.get("toolkit") in effective
        ):
            params.append(
                P(
                    "benchmark_ticker",
                    POS,
                    default=self._benchmark_ticker,
                    annotation=str,
                )
            )

        return params

    # ── Public read-only properties ─────────────────────────────────────────

    @property
    def start_date(self) -> str:
        """Default start date (YYYY-MM-DD) used for tool signature defaults."""
        return self._start_date

    @property
    def end_date(self) -> str:
        """Default end date (YYYY-MM-DD) used for tool signature defaults."""
        return self._end_date

    @property
    def benchmark_ticker(self) -> str:
        """Default benchmark ticker symbol used for tool signature defaults."""
        return self._benchmark_ticker

    @property
    def categories(self) -> dict[str, str]:
        """Category identifier mapping (e.g. 'ticker', 'standalone', 'mixed')."""
        return self._categories
