"""
Utility MCP tools for the Finance Toolkit MCP server.

Registers the four built-in discovery and utility tools that operate on the
*tool index* (the registry) rather than directly on Finance Toolkit controllers:

- ``list_categories``          — table of registered categories + counts
- ``list_metrics_by_category`` — all metrics within a single category
- ``search_metrics``           — fuzzy keyword search across all metrics
- ``search_instruments``       — look up ticker symbols by name/ISIN/CIK/…

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

from financetoolkit.mcp_server.diagnostics_model import redact
from financetoolkit.mcp_server.formatting_model import format_result
from financetoolkit.utilities.logger_model import get_logger

if TYPE_CHECKING:
    from financetoolkit.mcp_server.provider_model import ToolkitProvider
    from financetoolkit.mcp_server.registry_controller import ToolRegistry

logger = get_logger()


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
        provider (ToolkitProvider): Provider used for live data calls
            (``search_instruments``).
        search_stop_words (list[str]): List of stop words to ignore during search.
        category_descriptions (dict[str, str]): Descriptions for each category.
    """

    def __init__(
        self,
        mcp: FastMCP,
        registry: ToolRegistry,
        provider: ToolkitProvider,
        search_stop_words: list[str],
        category_descriptions: dict[str, str],
    ) -> None:
        """
        Initializes the UtilityToolRegistry.

        Args:
            mcp (FastMCP): The FastMCP server instance to register utility tools on.
            registry (ToolRegistry): The populated tool registry used to query the
                tool index. register_all_tools() must be called on the ToolRegistry
                before calling register_all_tools() on this registry.
            provider (ToolkitProvider): Provider used for live data calls in
                search_instruments.
            search_stop_words (list[str]): Words to exclude from keyword search
                tokenization (e.g. "the", "and", "for").
            category_descriptions (dict[str, str]): Human-readable description for
                each category, keyed by category name.
        """
        self._mcp = mcp
        self._registry = registry
        self._provider = provider
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
            (self.list_categories, "search_categories", "List Categories"),
            (
                self.list_metrics_by_category,
                "search_by_category",
                "List Metrics by Category",
            ),
            (self.search_metrics, "search_metrics", "Search Metrics"),
            (self.search_instruments, "search_instruments", "Search Instruments"),
        ]
        for method, tool_name, title in tools:
            self._mcp.add_tool(
                method,
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
        """List all available metric categories and how many tools each contains.

        Use this first to understand what is available, then call
        ``list_metrics_by_category`` with a specific category name.

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
        """List every available metric/tool within a category.

        Args:
            category: One of the category names returned by ``list_categories``,
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
        """Search across all metrics by keyword with typo tolerance.

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
            for w in re.findall(r"\b[a-zA-Z]{2,}\b", query)
            if w.lower() not in self._search_stop_words
        ]
        if not query_tokens:
            return (
                f"No meaningful search terms in '{query}'. "
                "Try a more specific keyword."
            )

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
                "Try `list_categories()` to browse all modules."
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

    def search_instruments(
        self,
        query: Annotated[
            str,
            Field(
                description="Company name, ticker symbol, CIK, CUSIP, or ISIN to look up, e.g. "
                "'Apple', 'AAPL', or '0000320193'."
            ),
        ],  # noqa: E501
        search_method: Annotated[
            str,
            Field(
                description="Lookup strategy: 'name', 'symbol', 'cik', 'cusip', or 'isin'. Defaults to 'name'."
            ),
        ] = "name",  # noqa: E501
    ) -> str:
        """Search for ticker symbols by company name, symbol, CIK, CUSIP, or ISIN.

        Args:
            query: The search term, e.g. ``'Apple'``, ``'META'``, ``'0000320193'``.
            search_method: Lookup strategy — one of ``'name'``, ``'symbol'``,
                ``'cik'``, ``'cusip'``, or ``'isin'``. Defaults to ``'name'``.

        Returns:
            str: Formatted Markdown table of matching instruments, or an error
                message if the search fails.
        """
        try:
            result = self._provider.call_method(
                module_name="discovery",
                method_name="search_instruments",
                category="discovery",
                query=query,
                search_method=search_method,
            )
            formatted = format_result(result)
            return formatted
        except Exception as exc:
            return f"Search failed: {redact(exc)}"
