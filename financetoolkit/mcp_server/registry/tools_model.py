"""
Utility MCP tools for the Finance Toolkit MCP server.

Registers the three built-in utility tools that operate on the *tool index* (the
registry) rather than directly on Finance Toolkit controllers:

- ``search_metric_categories`` — table of registered categories + counts
- ``search_by_category``       — all metrics within a single category
- ``search_metrics``           — fuzzy keyword search across all metrics

Looking up which instruments exist (by name, ISIN or CUSIP) is left to the Finance
Database MCP server, so the two servers do not offer overlapping tools.

These tools complement the dynamically-registered router groups produced by
``ToolRegistry``. They cannot be generated via the router pattern because they
either operate *on* the registry itself or perform cross-cutting queries that
are independent of any single controller module.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from financetoolkit.mcp_server.analytics.analytics_model import (
    UsageAnalytics,
    run_in_thread,
)
from financetoolkit.mcp_server.registry.registry_controller import (
    keep_unknown_arguments,
    name_unknown_arguments,
)
from financetoolkit.utilities.logger_model import get_logger

if TYPE_CHECKING:
    from financetoolkit.mcp_server.registry.registry_controller import ToolRegistry

logger = get_logger()

# The longest search and the most terms of it that are matched.
MAX_QUERY_CHARACTERS = 200


# A search result lists at most this many indicators, each with its call line: the
# parameters beyond the tool's common ones (up to this many characters), and at most this
# many values of each: the default, then the options the query names, then the others.
MAXIMUM_SEARCH_RESULTS = 10
MAXIMUM_CALL_CHARACTERS = 150
MAXIMUM_CALL_OPTIONS = 3
# The aliases a search result shows, in characters.
MAXIMUM_ALIAS_CHARACTERS = 100


def _call_value(value: object) -> str:
    """A default or option as written in a call line: 'historic', 0.05, True or None."""
    if value is None or isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return f"'{value}'"
    text = (
        json.dumps(value) if isinstance(value, (int, float, list, dict)) else str(value)
    )

    return text if len(text) <= 30 else text[:27] + "..."  # noqa: PLR2004


def call_line(entry: dict, query_tokens: list[str]) -> str:
    """An indicator as a call: its tool, name and the parameters it takes beyond the tool's
    common ones, with their defaults and options, e.g. ``risk get_value_at_risk(alpha=0.05,
    method='historic'|'gaussian'|...)``.

    Required parameters come first and always show; the parameters that only apply with
    another setting ("Only used with the GARCH method") come last. A longer list ends
    with "...", as do options beyond the first few; the default leads the options, followed
    by those the query names.

    Args:
        entry (dict): The indicator index entry.
        query_tokens (list[str]): The search words, to show the options they name.

    Returns:
        str: The call line.
    """
    common = set(entry.get("common_parameters", ()))
    parameters = sorted(
        (
            parameter
            for parameter in entry.get("call_parameters", ())
            if parameter["required"] or parameter["name"] not in common
        ),
        key=lambda parameter: (not parameter["required"], parameter["conditional"]),
    )
    parts: list[str] = []
    length = 0
    for parameter in parameters:
        if parameter["required"]:
            part = f"{parameter['name']} (required)"
        else:
            default = parameter["default"]
            options = [
                option
                for option in parameter["options"]
                if option != default and option is not None
            ]
            named = [
                option
                for option in options
                if str(option).lower().replace("-", " ").replace("_", " ")
                in query_tokens
            ]
            shown = [default] + named + [o for o in options if o not in named]
            values = "|".join(
                _call_value(value) for value in shown[:MAXIMUM_CALL_OPTIONS]
            )
            if len(shown) > MAXIMUM_CALL_OPTIONS:
                values += "|..."
            part = f"{parameter['name']}={values}"
        if (
            not parameter["required"]
            and parts
            and length + len(part) > MAXIMUM_CALL_CHARACTERS
        ):
            parts.append("...")
            break
        parts.append(part)
        length += len(part) + 2

    return f"{entry['tool']} {entry['indicator']}({', '.join(parts)})"


def describe_indicators(entries: list[dict]) -> str:
    """What each indicator of that name calculates, its tool and its parameters.

    Args:
        entries (list[dict]): The indicator index entries of one name (an indicator can be
            part of more than one tool).

    Returns:
        str: The description in Markdown.
    """
    sections = []
    for entry in entries:
        parameters = "\n".join(
            f"- `{name}`: {text}" for name, text in entry["parameters"].items()
        )
        # The tool's own parameters (dates, growth, rolling) apply to all its indicators;
        # those its schema describes are only named.
        tool_parameters = entry.get("tool_parameters", {})
        explained = "\n".join(
            f"- `{name}`: {text}" for name, text in tool_parameters.items() if text
        )
        named = ", ".join(name for name, text in tool_parameters.items() if not text)
        sections.append(
            f"### `{entry['indicator']}` (tool `{entry['tool']}`)\n\n"
            f"{entry['description']}\n\n"
            + (f"Parameters:\n{parameters}\n\n" if parameters else "")
            + (
                f"Parameters of the `{entry['tool']}` tool:\n{explained}\n\n"
                if explained
                else ""
            )
            + (
                f"The tool also takes {named} (see its schema), and "
                if named
                else "Every tool takes "
            )
            + "limit, rank, rebase_to, sample and show_columns (see the server instructions).\n\n"
            f"Call `{entry['tool']}` with `indicator='{entry['indicator']}'`."
        )

    return "\n\n".join(sections)


MAX_QUERY_TERMS = 12


def join_initials(text: str) -> str:
    """Writes abbreviations such as P/E and P/B as one word (PE, PB) for the search."""
    return re.sub(r"\b([A-Za-z])/([A-Za-z])\b", r"\1\2", text)


# Words for what an analyst wants to know that no indicator is named after, and the
# indicators that answer them. A query with such a word lists these first.
SEARCH_CONCEPTS: dict[str, tuple[str, ...]] = {
    "predictive": ("get_factor_analysis",),
    "predict": ("get_factor_analysis",),
    "oversold": ("get_indicator_summary", "get_relative_strength_index"),
    "overbought": ("get_indicator_summary", "get_relative_strength_index"),
    "expensive": (
        "get_price_to_earnings_ratio",
        "get_ev_to_ebitda_ratio",
        "get_price_to_book_ratio",
    ),
    "cheap": (
        "get_price_to_earnings_ratio",
        "get_ev_to_ebitda_ratio",
        "get_price_to_book_ratio",
    ),
    "overvalued": (
        "get_price_to_earnings_ratio",
        "get_intrinsic_valuation",
        "get_ev_to_ebitda_ratio",
    ),
    "undervalued": (
        "get_price_to_earnings_ratio",
        "get_intrinsic_valuation",
        "get_ev_to_ebitda_ratio",
    ),
    "bankruptcy": (
        "get_altman_z_score",
        "get_ohlson_o_score",
        "get_zmijewski_score",
        "get_campbell_hilscher_szilagyi_score",
    ),
    "distress": (
        "get_altman_z_score",
        "get_ohlson_o_score",
        "get_zmijewski_score",
        "get_campbell_hilscher_szilagyi_score",
    ),
    "constraints": (
        "get_kaplan_zingales_index",
        "get_whited_wu_index",
        "get_size_age_index",
    ),
    "constrained": (
        "get_kaplan_zingales_index",
        "get_whited_wu_index",
        "get_size_age_index",
    ),
    "default": (
        "get_altman_z_score",
        "get_default_rates",
        "get_rating_transition_matrix",
    ),
    "correlation": ("get_correlation_matrix",),
    "insider": ("get_insider_trade_statistics", "get_insider_trading_latest"),
    "institutional": ("get_institutional_ownership",),
    "13f": ("get_institutional_ownership",),
    "ownership": ("get_institutional_ownership", "get_insider_trade_statistics"),
    "target": (
        "get_price_targets",
        "get_price_target_history",
        "get_analyst_estimates",
    ),
    "recommendation": ("get_analyst_recommendations", "get_stock_grades"),
    "forecast": (
        "get_analyst_estimates",
        "get_inflation_expectations",
        "get_arima_forecast",
    ),
    "revenue": (
        "get_income_statement",
        "get_revenue_product_segmentation",
        "get_revenue_geographic_segmentation",
    ),
    "history": ("get_historical_data",),
    "greeks": ("get_delta", "get_gamma", "get_theta", "get_vega", "get_rho"),
    "bond": ("get_present_value", "get_yield_to_maturity", "get_duration"),
    "buyback": ("get_buyback_yield", "get_shareholder_yield"),
    "buybacks": ("get_buyback_yield", "get_shareholder_yield"),
    "breakeven": ("get_drawdown_series", "get_drawdown_table"),
    "recover": ("get_drawdown_series", "get_drawdown_table"),
    "exceedance": ("get_move_significance", "get_var_backtest"),
}

# Shorter search words must match a whole word, as fuzzy matching makes them match anything.
FUZZY_MINIMUM_LENGTH = 4

# The share of the search words an indicator has to match to be listed.
MINIMUM_INDICATOR_SCORE = 0.6


class UtilityToolRegistry:
    """
    Registers the built-in utility and discovery tools on a FastMCP instance.

    These tools are separate from the router groups managed by ``ToolRegistry``
    because they operate on the tool index rather than routing to a
    Finance Toolkit controller method.

    ``ToolRegistry.register_all_tools()`` must be called before calling
    :meth:`register_all_tools` on this registry so that ``get_tool_index``
    returns a complete index.

    Args:
        mcp (FastMCP): The FastMCP server instance to register tools on.
        registry (ToolRegistry): The populated tool registry.
        search_stop_words (list[str]): List of stop words to ignore during search.
        category_descriptions (dict[str, str]): Descriptions for each category.
    """

    def __init__(
        self,
        mcp: FastMCP,
        registry: ToolRegistry,
        search_stop_words: list[str],
        category_descriptions: dict[str, str],
        analytics: UsageAnalytics | None = None,
    ) -> None:
        """
        Initializes the UtilityToolRegistry.

        Args:
            mcp (FastMCP): The FastMCP server instance to register utility tools on.
            registry (ToolRegistry): The populated tool registry used to query the
                tool index. register_all_tools() must be called on the ToolRegistry
                before calling register_all_tools() on this registry.
            search_stop_words (list[str]): Words to exclude from keyword search
                tokenization (e.g. "the", "and", "for").
            category_descriptions (dict[str, str]): Human-readable description for
                each category, keyed by category name.
            analytics (UsageAnalytics | None): Counts the calls of every tool when the
                server's usage analytics are on. Defaults to None.
        """
        self._mcp = mcp
        self._analytics = analytics
        self._registry = registry
        self._search_stop_words: frozenset[str] = frozenset(search_stop_words)
        self._category_descriptions: dict[str, str] = category_descriptions

    def register_all_tools(self) -> int:
        """Register all utility tools on the FastMCP instance.

        Passes each bound tool method to ``mcp.add_tool()``. Because bound methods
        exclude ``self`` from their inspected signature, FastMCP receives the correct
        parameter list for each tool.

        Returns:
            int: Number of utility tools successfully registered.
        """
        _utility_annotations = ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=False,
        )
        tools = [
            (
                self.list_categories,
                "search_metric_categories",
                "List Metric Categories",
            ),
            (
                self.list_metrics_by_category,
                "search_by_category",
                "List Metrics by Category",
            ),
            (self.search_metrics, "search_metrics", "Search Metrics"),
        ]
        for method, tool_name, title in tools:
            tool = run_in_thread(name_unknown_arguments(method, tool_name))
            self._mcp.add_tool(
                (
                    self._analytics.counted(tool_name, tool)
                    if self._analytics is not None
                    else tool
                ),
                name=tool_name,
                description=method.__doc__ or "",
                annotations=ToolAnnotations(
                    title=title,
                    **_utility_annotations.model_dump(
                        exclude={"title"}, exclude_none=True
                    ),
                ),
                structured_output=False,
            )
            # A misspelt or foreign argument is named in the answer rather than dropped.
            keep_unknown_arguments(self._mcp, tool_name)
        logger.info("Registered %d utility tools.", len(tools))
        return len(tools)

    # ── Tool methods ──────────────────────────────────────────────────────────
    def list_categories(self) -> str:
        """Finance Toolkit: list the categories of what this server calculates (prices,
        financial statements, ratios, models, risk, economics) and how many tools, indicators
        and metrics each has.

        An indicator counts as one metric per formula variant it offers through its
        ``method`` parameter (``get_sharpe_ratio`` with method="smart" or "deflated"), so the
        metrics are what the server calculates and the indicators what it can be asked for.
        Each category counts the indicators of its tools; the total counts a method that
        several tools list once.

        Use this first to understand what is available, then call
        ``search_by_category`` with a specific category name.

        Returns:
            str: Markdown table of categories, tool, indicator and metric counts, and
                descriptions, with the totals.
        """
        index = self._registry.get_tool_index()
        indicators_per_tool, metrics_per_tool, unique_metrics = self._count_metrics()
        lines = [
            "| Category | Tools | Indicators | Metrics | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cat in sorted(index.keys()):
            tools = index.get(cat, [])
            indicators = sum(indicators_per_tool[tool["tool"]] for tool in tools)
            metrics = sum(metrics_per_tool[tool["tool"]] for tool in tools)
            desc = self._category_descriptions.get(cat, "N/A")
            lines.append(
                f"| `{cat}` | {len(tools)} | {indicators} | {metrics} | {desc} |"
            )
        total_tools = sum(len(v) for v in index.values())
        # A method several tools list counts once in the total, while two methods that
        # share a name (an ETF's and a company's get_profile) count as two.
        lines.append(
            f"\n**Total: {sum(unique_metrics.values())} metrics "
            f"({len(unique_metrics)} indicators) in {total_tools} tools**"
        )
        lines.append(
            "\n**Tip:** Use `search_metrics('what you want to measure')` to find the indicator and its tool."
        )
        result = "\n".join(lines)
        return result

    def _count_metrics(self) -> tuple[Counter[str], Counter[str], dict[str, int]]:
        """
        Counts the indicators and metrics: an indicator is one metric per formula variant
        the server offers through its ``method`` parameter.

        Returns:
            tuple[Counter[str], Counter[str], dict[str, int]]: The indicators and the
                metrics per tool, and the metrics per indicator method, each method once
                however many tools list it.
        """
        indicators_per_tool: Counter[str] = Counter()
        metrics_per_tool: Counter[str] = Counter()
        unique_metrics: dict[str, int] = {}
        for entry in self._registry.get_indicator_index():
            metrics = max(1, len(entry.get("variants", ())))
            indicators_per_tool[entry["tool"]] += 1
            metrics_per_tool[entry["tool"]] += metrics
            key = entry.get("key") or entry["indicator"]
            unique_metrics[key] = max(metrics, unique_metrics.get(key, 0))

        return indicators_per_tool, metrics_per_tool, unique_metrics

    def list_metrics_by_category(
        self,
        category: Annotated[
            str,
            Field(
                description="Category name as returned by search.categories, "
                "e.g. 'ratios', 'technicals', 'economics', 'discovery'."
            ),
        ],  # noqa: E501
    ) -> str:
        """Finance Toolkit: list the tools within a category of prices, financial
        statements, ratios and other calculations, with how many indicators and metrics
        each has.

        Args:
            category: One of the category names returned by ``search_metric_categories``,
                e.g. ``ratios``, ``technicals``, ``economics``, ``discovery``.

        Returns:
            str: Markdown table of the tools of the requested category with their
                indicator and metric counts and descriptions, or an error message if the
                category is unknown.
        """
        index = self._registry.get_tool_index()
        cat = category.lower().strip()
        if cat not in index:
            available = ", ".join(sorted(index.keys()))
            return f"Unknown category `{cat}`. Available: {available}"

        tools = index[cat]
        if not tools:
            return f"No tools registered for `{cat}`."

        indicators_per_tool, metrics_per_tool, _ = self._count_metrics()
        lines = [
            f"### {cat} — {len(tools)} {'tool' if len(tools) == 1 else 'tools'}, "
            f"{sum(indicators_per_tool[t['tool']] for t in tools)} indicators, "
            f"{sum(metrics_per_tool[t['tool']] for t in tools)} metrics\n",
            "| Tool | Indicators | Metrics | Description |",
            "| --- | --- | --- | --- |",
        ]
        for t in tools:
            lines.append(
                f"| `{t['tool']}` | {indicators_per_tool[t['tool']]} | "
                f"{metrics_per_tool[t['tool']]} | {t['description']} |"
            )
        result = "\n".join(lines)
        return result

    def _search_indicators(self, query_tokens: list[str]) -> list[dict[str, str]]:
        """
        Ranks the indicators by how well their name, aliases and summary match the query.

        A word of the name or an alias counts fully and a word of the summary half. A close
        spelling (four letters or more) only counts for a word that matches nothing exactly,
        so "sharpe" does not also find "share" while "equitty" still finds "equity". The whole
        query as the name or an alias ranks first; short tokens such as "pe" must match a
        whole word.

        Args:
            query_tokens (list[str]): The lowercase search words, without stop words.

        Returns:
            list[dict[str, str]]: At most MAXIMUM_SEARCH_RESULTS indicators, the best match
                first.
        """

        # Single letters are no search words ("Jensen's alpha" is jensen and alpha).
        def words(text: str) -> list[str]:
            return [
                word
                for word in re.findall(r"[a-z0-9]+", join_initials(text).lower())
                if len(word) > 1 and word not in self._search_stop_words
            ]

        entries = []
        for entry in self._registry.get_indicator_index():
            name = words(entry["indicator"].removeprefix("get_").replace("_", " "))
            aliases = [words(alias) for alias in entry["aliases"].split(",")]
            title_words = set(name).union(*aliases)
            entries.append(
                (entry, name, aliases, title_words, set(words(entry["summary"])))
            )

        def exact(token: str, vocabulary: set[str]) -> bool:
            return token in vocabulary or (
                len(token) >= FUZZY_MINIMUM_LENGTH
                and any(word.startswith(token) for word in vocabulary)
            )

        known = {
            token
            for token in query_tokens
            for _, _, _, title_words, summary_words in entries
            if exact(token, title_words) or token in summary_words
        }

        # Indicators named in SEARCH_CONCEPTS for a word of the query rank first.
        concept_hits: dict[str, float] = {}
        for token in query_tokens:
            names = SEARCH_CONCEPTS.get(token, ())
            for position, name in enumerate(names):
                # The first indicator of a concept is the most direct answer.
                concept_hits[name] = max(
                    concept_hits.get(name, 0.0), 2.0 + 0.1 * (len(names) - position)
                )

        scored: list[tuple[float, int, dict[str, str]]] = []

        for position, (entry, name, aliases, title_words, summary_words) in enumerate(
            entries
        ):
            score = 0.0

            for token in query_tokens:
                if exact(token, title_words):
                    score += 1.0
                elif token in summary_words:
                    score += 0.5
                elif token in known or len(token) < FUZZY_MINIMUM_LENGTH:
                    continue
                elif difflib.get_close_matches(token, title_words, n=1, cutoff=0.8):
                    score += 0.7
                elif difflib.get_close_matches(token, summary_words, n=1, cutoff=0.8):
                    score += 0.3

            score /= len(query_tokens)

            # The query as the indicator's name, or as one of its aliases.
            if query_tokens == name:
                score += 1.5
            elif query_tokens in aliases:
                score += 1.0

            score += concept_hits.get(entry["indicator"], 0.0)

            if score >= MINIMUM_INDICATOR_SCORE:
                scored.append((score, position, entry))

        scored.sort(key=lambda item: (-item[0], item[1]))

        return [entry for _, _, entry in scored[:MAXIMUM_SEARCH_RESULTS]]

    def search_metrics(
        self,
        query: Annotated[
            str,
            Field(
                description="Free-text keyword to search across all metric names and "
                "descriptions, e.g. 'sharpe', 'debt', or 'moving average'."
            ),
        ],  # noqa: E501
    ) -> str:
        """Finance Toolkit: search the metrics this server calculates (prices, financial
        statements, ratios and more) by keyword, with typo tolerance.

        Supports minor typos and common financial abbreviations. Tokens
        shorter than four characters bypass fuzzy matching and require an
        exact substring hit.

        Args:
            query: Free-text search string, e.g. ``'debt'``,
                ``'moving average'``, ``'sharpe'``, or ``'retun on equty'``.

        Returns:
            str: Markdown table of matching tools sorted by relevance score,
                or a guidance message when no strong matches are found.
        """
        # Every term is matched against every indicator and description with typo tolerance,
        # so the work grows with the length of the query; a search is a few words.
        if len(query) > MAX_QUERY_CHARACTERS:
            return (
                f"The search is {len(query)} characters long; keep it to a few keywords "
                f"(at most {MAX_QUERY_CHARACTERS} characters)."
            )
        # An indicator's own name answers with what it calculates and its parameters.
        named = [
            entry
            for entry in self._registry.get_indicator_index()
            if entry["indicator"] == query.strip().strip("`'\" ")
        ]
        if named:
            return describe_indicators(named)

        index = self._registry.get_tool_index()
        query_tokens = [
            w.lower()
            # An indicator's name reads as its words (get_return_on_equity).
            # Words of two characters or more with a letter, so "2s10s" and "13f" count.
            for w in re.findall(
                r"\b[a-zA-Z0-9]*[a-zA-Z][a-zA-Z0-9]*\b",
                join_initials(query.replace("_", " ")),
            )
            if len(w) > 1 and w.lower() not in self._search_stop_words
        ][:MAX_QUERY_TERMS]
        if not query_tokens:
            return (
                f"No meaningful search terms in '{query}'. Try a more specific keyword."
            )

        # Indicators first: an agent asking for "free cash flow yield" needs the indicator
        # and its tool, not every tool whose description mentions cash flow.
        indicator_hits = self._search_indicators(query_tokens)

        if indicator_hits:
            lines = [
                f"### Search results for '{query}' — {len(indicator_hits)} indicators\n"
            ]
            for entry in indicator_hits:
                aliases = entry["aliases"]
                if len(aliases) > MAXIMUM_ALIAS_CHARACTERS:
                    aliases = (
                        aliases[:MAXIMUM_ALIAS_CHARACTERS].rsplit(",", 1)[0] + ", ..."
                    )
                lines.append(
                    f"- `{call_line(entry, query_tokens)}`: {entry['summary']}."
                    + (f" Also known as: {aliases}." if aliases else "")
                )
            # The parameters most indicators of a tool take, once per tool in the results.
            common = {
                entry["tool"]: entry.get("common_parameters", [])
                for entry in indicator_hits
            }
            shared = "; ".join(
                f"{', '.join(names)} ({tool})"
                for tool, names in common.items()
                if names
            )
            lines.append(
                "\nCall the tool with indicator='<name>' and any of these arguments (the "
                "first value is the default, '...' marks more)."
                + (f" Most indicators also take {shared}." if shared else "")
                + " search_metrics with an indicator's name explains each parameter."
            )
            return "\n".join(lines)

        results: list[tuple[float, dict]] = []
        for cat, tools in index.items():
            for t in tools:
                search_text = f"{t['tool']} {t['description']}".lower()
                text_words = set(re.findall(r"\b[a-z]{3,}\b", search_text))
                score = 0.0
                for token in query_tokens:
                    if token in search_text:
                        score += 1.0
                    elif len(token) >= 4:  # noqa
                        close = difflib.get_close_matches(
                            token, text_words, n=1, cutoff=0.82
                        )
                        if close:
                            score += 0.7
                if score > 0:
                    results.append((score / len(query_tokens), {**t, "category": cat}))

        results.sort(key=lambda x: x[0], reverse=True)
        hits = [(s, t) for s, t in results if s >= 0.3][:30]  # noqa

        if not hits:
            return f"No strong matches for '{query}'. Try `search_metric_categories` to browse all modules."

        lines = [
            f"### Search results for '{query}' — {len(hits)} matches\n",
            "| Category | Tool | Description |",
            "| --- | --- | --- |",
        ]
        for _, h in hits:
            lines.append(f"| `{h['category']}` | `{h['tool']}` | {h['description']} |")
        result = "\n".join(lines)
        return result
