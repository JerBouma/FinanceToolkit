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
import re
from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from financetoolkit.mcp_server.analytics_model import UsageAnalytics
from financetoolkit.utilities.logger_model import get_logger

if TYPE_CHECKING:
    from financetoolkit.mcp_server.registry_controller import ToolRegistry

logger = get_logger()


def join_initials(text: str) -> str:
    """Writes abbreviations such as P/E and P/B as one word (PE, PB) for the search."""
    return re.sub(r"\b([A-Za-z])/([A-Za-z])\b", r"\1\2", text)


# Words for what an analyst wants to know that no indicator is named after, and the
# indicators that answer them. A query with such a word lists these first.
SEARCH_CONCEPTS: dict[str, tuple[str, ...]] = {
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
    "bankruptcy": ("get_altman_z_score", "get_ohlson_o_score", "get_zmijewski_score"),
    "distress": ("get_altman_z_score", "get_ohlson_o_score", "get_zmijewski_score"),
    "default": (
        "get_altman_z_score",
        "get_default_rates",
        "get_rating_transition_matrix",
    ),
    "correlation": ("get_correlation_matrix",),
    "insider": ("get_insider_trade_statistics", "get_insider_trading_latest"),
    "target": ("get_analyst_estimates", "get_stock_grades"),
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
            self._mcp.add_tool(
                (
                    self._analytics.counted(tool_name, method)
                    if self._analytics is not None
                    else method
                ),
                name=tool_name,
                description=method.__doc__ or "",
                annotations=ToolAnnotations(
                    title=title,
                    **_utility_annotations.model_dump(
                        exclude={"title"}, exclude_none=True
                    ),
                ),
            )
        logger.info("Registered %d utility tools.", len(tools))
        return len(tools)

    # ── Tool methods ──────────────────────────────────────────────────────────
    def list_categories(self) -> str:
        """Finance Toolkit: list the categories of what this server calculates (prices,
        financial statements, ratios, models, risk, economics) and how many tools each has.

        Use this first to understand what is available, then call
        ``search_by_category`` with a specific category name.

        Returns:
            str: Markdown table of categories, tool counts, and descriptions.
        """
        index = self._registry.get_tool_index()
        lines = ["| Category | Tools | Description |", "| --- | --- | --- |"]
        for cat in sorted(index.keys()):
            n = len(index.get(cat, []))
            desc = self._category_descriptions.get(cat, "N/A")
            lines.append(f"| `{cat}` | {n} | {desc} |")
        total = sum(len(v) for v in index.values())
        lines.append(f"\n**Total tools: {total}**")
        lines.append(
            "\n**Tip:** Use `search_metrics('keyword')` to find tools by keyword."
        )
        result = "\n".join(lines)
        return result

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
        """Finance Toolkit: list every metric/tool within a category of prices, financial
        statements, ratios and other calculations.

        Args:
            category: One of the category names returned by ``search_metric_categories``,
                e.g. ``ratios``, ``technicals``, ``economics``, ``discovery``.

        Returns:
            str: Markdown table of tool names and their descriptions for the
                requested category, or an error message if the category is unknown.
        """
        index = self._registry.get_tool_index()
        cat = category.lower().strip()
        if cat not in index:
            available = ", ".join(sorted(index.keys()))
            return f"Unknown category `{cat}`. Available: {available}"

        tools = index[cat]
        if not tools:
            return f"No tools registered for `{cat}`."

        lines = [
            f"### {cat} — {len(tools)} metrics\n",
            "| Tool | Description |",
            "| --- | --- |",
        ]
        for t in tools:
            lines.append(f"| `{t['tool']}` | {t['description']} |")
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
            list[dict[str, str]]: At most 15 indicators, the best match first.
        """

        def words(text: str) -> list[str]:
            return [
                word
                for word in re.findall(r"[a-z0-9]+", join_initials(text).lower())
                if word not in self._search_stop_words
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

        return [entry for _, _, entry in scored[:15]]

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
        index = self._registry.get_tool_index()
        query_tokens = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z]{2,}\b", join_initials(query))
            if w.lower() not in self._search_stop_words
        ]
        if not query_tokens:
            return (
                f"No meaningful search terms in '{query}'. "
                "Try a more specific keyword."
            )

        # Indicators first: an agent asking for "free cash flow yield" needs the indicator
        # and its tool, not every tool whose description mentions cash flow.
        indicator_hits = self._search_indicators(query_tokens)

        if indicator_hits:
            lines = [
                f"### Search results for '{query}' — {len(indicator_hits)} indicators\n",
                "| Tool | Indicator | Description |",
                "| --- | --- | --- |",
            ]
            for entry in indicator_hits:
                aliases = (
                    f" Also known as: {entry['aliases']}." if entry["aliases"] else ""
                )
                lines.append(
                    f"| `{entry['tool']}` | `{entry['indicator']}` | {entry['summary']}.{aliases} |"
                )
            first = indicator_hits[0]
            lines.append(
                f"\nCall the tool with the indicator, e.g. `{first['tool']}` with "
                f"`indicator='{first['indicator']}'`; the tool's description lists its parameters."
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
            return (
                f"No strong matches for '{query}'. "
                "Try `search_metric_categories` to browse all modules."
            )

        lines = [
            f"### Search results for '{query}' — {len(hits)} matches\n",
            "| Category | Tool | Description |",
            "| --- | --- | --- |",
        ]
        for _, h in hits:
            lines.append(f"| `{h['category']}` | `{h['tool']}` | {h['description']} |")
        result = "\n".join(lines)
        return result
