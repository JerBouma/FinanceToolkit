"""
The usage dashboard a browser gets at ``/stats``.

Renders the totals of ``Analytics.summary`` as one self-contained HTML page: key figures,
a few generated observations and inline SVG charts of the daily calls, the user growth
and the most used tools. No scripts and nothing loaded from elsewhere, so it works behind
any proxy. It shows the same aggregate figures as the JSON, never anything per user.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any

ACCENT = "#38bdf8"
MUTED = "#94a3b8"
# Two weeks of history are needed to compare the last week with the one before.
TWO_WEEKS = 14


def _number(value: float) -> str:
    return f"{value:,.0f}"


def _month_day(day: str) -> str:
    return datetime.strptime(day, "%Y-%m-%d").strftime("%-d %b")


def _rolling_mean(values: list[float], window: int = 7) -> list[float]:
    """The trailing mean over `window` days, over fewer at the start."""
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1)
        for i in range(len(values))
    ]


def _axis_labels(history: list[dict[str, Any]], width: float, height: float) -> str:
    """About six evenly spread date labels under a chart."""
    count = len(history)
    step = max(1, count // 6)
    labels = []
    for i in range(0, count, step):
        x = (i + 0.5) / count * width
        anchor = "start" if i == 0 else "middle"
        labels.append(
            f'<text x="{x:.1f}" y="{height + 22}" fill="{MUTED}" font-size="12" '
            f'text-anchor="{anchor}">{_month_day(history[i]["date"])}</text>'
        )
    return "".join(labels)


def _calls_chart(history: list[dict[str, Any]]) -> str:
    """Daily calls as bars with the 7-day average as a line."""
    width, height = 900, 220
    calls = [entry["calls"] for entry in history]
    top = max(calls) * 1.08 if calls and max(calls) else 1
    count = len(calls)
    slot = width / count
    bars = "".join(
        f'<rect x="{i * slot + slot * 0.15:.1f}" y="{height - value / top * height:.1f}" '
        f'width="{slot * 0.7:.1f}" height="{value / top * height:.1f}" rx="1.5" '
        f'fill="rgba(56,189,248,0.45)"><title>{entry["date"]}: {_number(value)} calls</title></rect>'
        for i, (entry, value) in enumerate(zip(history, calls, strict=True))
    )
    average = _rolling_mean(calls)
    points = " ".join(
        f"{(i + 0.5) * slot:.1f},{height - value / top * height:.1f}"
        for i, value in enumerate(average)
    )
    line = (
        f'<polyline points="{points}" fill="none" stroke="#fbbf24" stroke-width="2.5" '
        'stroke-linejoin="round"/>'
    )
    return (
        f'<svg viewBox="0 -10 {width} {height + 40}" role="img" aria-label="Daily calls">'
        f"{bars}{line}{_axis_labels(history, width, height)}</svg>"
    )


def _users_chart(history: list[dict[str, Any]]) -> str:
    """The number of users so far, as an area."""
    width, height = 900, 160
    users = [entry["users"] for entry in history]
    top = max(users) * 1.1 if users and max(users) else 1
    count = len(users)
    points = [
        (
            (i + 0.5) / count * width,
            height - value / top * height,
        )
        for i, value in enumerate(users)
    ]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = (
        f"{points[0][0]:.1f},{height} {path} {points[-1][0]:.1f},{height}"
        if points
        else ""
    )
    return (
        f'<svg viewBox="0 -10 {width} {height + 40}" role="img" aria-label="Users over time">'
        '<defs><linearGradient id="u" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.35"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="{area}" fill="url(#u)"/>'
        f'<polyline points="{path}" fill="none" stroke="{ACCENT}" stroke-width="2.5" stroke-linejoin="round"/>'
        f"{_axis_labels(history, width, height)}</svg>"
    )


def _tools_table(tools: list[dict[str, Any]], total: int) -> str:
    """The most used tools with their share of all calls and their response time."""
    if not tools:
        return '<p class="muted">No calls yet.</p>'
    top = tools[0]["calls"]
    rows = "".join(
        f'<div class="tool"><span class="name">{html.escape(str(tool["tool"]))}</span>'
        f'<span class="bar"><i style="width:{tool["calls"] / top * 100:.1f}%"></i></span>'
        f'<span class="num">{_number(tool["calls"])}</span>'
        f'<span class="num muted">{tool["calls"] / total * 100:.1f}%</span>'
        f'<span class="num muted">{tool["average_seconds"]:.2f}s</span></div>'
        for tool in tools
    )
    head = (
        '<div class="tool head"><span>Tool</span><span></span><span class="num">Calls</span>'
        '<span class="num">Share</span><span class="num">Avg. time</span></div>'
    )
    return head + rows


def _observations(summary: dict[str, Any]) -> list[str]:
    """A few plain-language observations drawn from the totals."""
    history = summary.get("history") or []
    notes = []
    if len(history) >= TWO_WEEKS:
        last = sum(entry["calls"] for entry in history[-7:])
        before = sum(entry["calls"] for entry in history[-14:-7])
        if before:
            change = (last - before) / before * 100
            direction = "up" if change >= 0 else "down"
            notes.append(
                f"Calls over the last 7 days are {direction} {abs(change):.0f}% on the 7 days before."
            )
    if history:
        busiest = max(history, key=lambda entry: entry["calls"])
        if busiest["calls"]:
            notes.append(
                f"The busiest day so far was {_month_day(busiest['date'])} "
                f"{busiest['date'][:4]}, with {_number(busiest['calls'])} calls."
            )
    tools = summary.get("top_tools") or []
    total = summary.get("total_calls") or 0
    if tools and total:
        notes.append(
            f"<code>{html.escape(str(tools[0]['tool']))}</code> answers "
            f"{tools[0]['calls'] / total * 100:.0f}% of all calls."
        )
    if summary.get("median_calls_per_user"):
        notes.append(
            f"Half of the users made at least {_number(summary['median_calls_per_user'])} calls; "
            f"the average is {_number(summary['average_calls_per_user'])}, pulled up by a few heavy users."
        )
    if summary.get("new_users_last_7_days"):
        notes.append(
            f"{summary['new_users_last_7_days']} new users joined in the last 7 days."
        )
    return notes


def render(summary: dict[str, Any]) -> str:
    """
    Render the usage totals as an HTML dashboard.

    Args:
        summary (dict[str, Any]): The totals as returned by ``Analytics.summary``.

    Returns:
        str: The page.
    """
    history = summary.get("history") or []
    total = summary.get("total_calls") or 0
    success = summary.get("success_rate")
    seconds = summary.get("average_seconds")
    since = summary.get("since")
    cards = [
        ("Total calls", _number(total)),
        ("Users", _number(summary.get("unique_users") or 0)),
        ("Calls, last 7 days", _number(summary.get("calls_last_7_days") or 0)),
        ("New users, last 7 days", _number(summary.get("new_users_last_7_days") or 0)),
        ("Success rate", f"{success * 100:.1f}%" if success is not None else "–"),
        ("Avg. response", f"{seconds:.2f}s" if seconds is not None else "–"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )
    notes = "".join(f"<li>{note}</li>" for note in _observations(summary))
    since_text = f" since {_month_day(since)} {since[:4]}" if since else ""
    charts = ""
    if history:
        charts = (
            f"<section><h2>Daily calls{since_text}</h2>"
            f'<p class="legend"><i style="background:rgba(56,189,248,0.6)"></i>Calls '
            '<i style="background:#fbbf24"></i>7-day average</p>'
            f"{_calls_chart(history)}</section>"
            f"<section><h2>Users over time</h2>{_users_chart(history)}</section>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="300">
<title>Finance Toolkit MCP · Usage</title>
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
<style>
  body {{ margin: 0; background: linear-gradient(135deg, #0f172a 0%, #020617 100%) fixed; color: #f8fafc;
         font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 64rem; margin: 0 auto; padding: 2.5rem 1.25rem 3rem; }}
  .head {{ display: flex; align-items: center; gap: 0.9rem; }}
  .head img {{ width: 3rem; height: 3rem; border-radius: 0.8rem; }}
  h1 {{ margin: 0; font-size: 1.8rem; }} h1 span {{ color: {ACCENT}; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 0.8rem; color: #e2e8f0; }}
  .sub {{ color: {MUTED}; margin: 0.4rem 0 1.6rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: 0.8rem; }}
  .card, section, .notes {{
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 1rem;
  }}
  .card {{ padding: 1rem 1.1rem; }}
  .label {{ color: {MUTED}; font-size: 0.85rem; }}
  .value {{ font-size: 1.7rem; font-weight: 700; margin-top: 0.3rem; font-variant-numeric: tabular-nums; }}
  .notes {{ margin: 1rem 0; padding: 1rem 1.4rem; }}
  .notes ul {{ margin: 0; padding-left: 1.1rem; color: #cbd5e1; line-height: 1.8; }}
  section {{ margin-top: 1rem; padding: 1.2rem 1.4rem; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .legend {{ color: {MUTED}; font-size: 0.85rem; margin: -0.4rem 0 0.6rem; }}
  .legend i {{
    display: inline-block;
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 0.2rem;
    margin: 0 0.35rem 0 0.8rem;
    vertical-align: -0.1rem;
  }}
  .legend i:first-child {{ margin-left: 0; }}
  .tool {{
    display: grid;
    grid-template-columns: 9.5rem 1fr 4.5rem 4rem 4.5rem;
    gap: 0.8rem;
    align-items: center;
    padding: 0.35rem 0;
  }}
  .tool.head {{ color: {MUTED}; font-size: 0.8rem; }}
  .tool .name {{ font-family: ui-monospace, "JetBrains Mono", monospace; font-size: 0.9rem; }}
  .bar {{ height: 0.6rem; background: rgba(148,163,184,0.12); border-radius: 0.3rem; overflow: hidden; }}
  .bar i {{ display: block; height: 100%; background: linear-gradient(90deg, #0284c7, {ACCENT}); border-radius: 0.3rem; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .muted {{ color: {MUTED}; }}
  code {{ font-family: ui-monospace, "JetBrains Mono", monospace; color: {ACCENT}; }}
  footer {{ margin-top: 1.5rem; color: {MUTED}; font-size: 0.85rem; }}
  footer a {{ color: {ACCENT}; }}
  @media (max-width: 40rem) {{
    .tool {{ grid-template-columns: 7rem 1fr 3.8rem 3.4rem;
    }} .tool > :last-child {{ display: none;
    }};
  }}
</style>
</head>
<body>
<main>
  <div class="head"><img src="/oauth/icon" alt=""><h1>Finance Toolkit <span>MCP</span> usage</h1></div>
  <p class="sub">Live totals of the hosted server{since_text}. No per-user figures are published;
  users are counted by a one-way hash of their key.</p>
  <div class="cards">{card_html}</div>
  {f'<div class="notes"><ul>{notes}</ul></div>' if notes else ""}
  {charts}
  <section><h2>Most used tools</h2>{_tools_table(summary.get("top_tools") or [], total or 1)}</section>
  <footer>Updates every 5 minutes. The same figures as JSON: <code>curl {html.escape("https://financetoolkit.jeroenbouma.com/stats")}</code>.
  Set it up yourself: <a href="https://www.jeroenbouma.com/projects/financetoolkit/mcp">Finance Toolkit MCP</a>.</footer>
</main>
</body>
</html>
"""
