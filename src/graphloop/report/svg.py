"""Tiny, dependency-free SVG charts that render on GitHub in light and dark themes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

#: Colour per agent; mid-tones chosen to read on both light and dark backgrounds.
PALETTE = {"loop": "#d1603d", "graph": "#3b82c4", "graph+memory": "#2a9d74"}
_FALLBACK = ("#8a63d2", "#c99a2e", "#5f7a8a")
_INK = "#7f8a96"  # neutral axis/label colour, legible in both themes
_FONT = "font-family='ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial,sans-serif'"


def colour(agent: str, i: int = 0) -> str:
    return PALETTE.get(agent, _FALLBACK[i % len(_FALLBACK)])


@dataclass(frozen=True)
class Bar:
    label: str
    value: float
    low: float | None = None
    high: float | None = None
    colour: str = "#3b82c4"
    text: str | None = None


def bar_chart(
    title: str,
    bars: Sequence[Bar],
    unit: str = "",
    width: int = 560,
    max_value: float | None = None,
) -> str:
    """Vertical bars with optional confidence-interval whiskers."""
    height, top, bottom, left = 300, 44, 46, 48
    plot_h = height - top - bottom
    plot_w = width - left - 16
    vmax = max_value or max((b.high or b.value) for b in bars) * 1.15 or 1.0
    slot = plot_w / max(1, len(bars))
    bw = min(90.0, slot * 0.55)
    parts = [_open(width, height, title)]
    for t in range(5):
        v = vmax * t / 4
        y = top + plot_h - plot_h * t / 4
        parts.append(
            f"<line x1='{left}' x2='{width - 16}' y1='{y:.1f}' y2='{y:.1f}' "
            f"stroke='{_INK}' stroke-opacity='0.18'/>"
        )
        parts.append(_text(left - 6, y + 4, _fmt(v, unit), anchor="end", size=11))
    for i, b in enumerate(bars):
        cx = left + slot * i + slot / 2
        h = plot_h * b.value / vmax
        y = top + plot_h - h
        parts.append(
            f"<rect x='{cx - bw / 2:.1f}' y='{y:.1f}' width='{bw:.1f}' "
            f"height='{h:.1f}' rx='4' fill='{b.colour}'/>"
        )
        if b.low is not None and b.high is not None:
            y1 = top + plot_h - plot_h * b.low / vmax
            y2 = top + plot_h - plot_h * b.high / vmax
            parts.append(
                f"<path d='M{cx:.1f},{y1:.1f}V{y2:.1f}M{cx - 8:.1f},{y1:.1f}"
                f"h16M{cx - 8:.1f},{y2:.1f}h16' stroke='{_INK}' stroke-width='1.5'/>"
            )
        label_y = (min(y, top + plot_h - plot_h * (b.high or b.value) / vmax)) - 6
        parts.append(_text(cx, label_y, b.text or _fmt(b.value, unit), size=12, weight=600))
        parts.append(_text(cx, height - bottom + 18, b.label, size=12))
    parts.append("</svg>")
    return "".join(parts)


def line_chart(
    title: str,
    x_labels: Sequence[str],
    series: dict[str, Sequence[float]],
    unit: str = "",
    width: int = 560,
) -> str:
    height, top, bottom, left = 300, 44, 64, 48
    plot_h = height - top - bottom
    plot_w = width - left - 24
    vmax = max((v for s in series.values() for v in s), default=1.0) * 1.15 or 1.0
    n = max(1, len(x_labels) - 1)
    parts = [_open(width, height, title)]
    for t in range(5):
        v = vmax * t / 4
        y = top + plot_h - plot_h * t / 4
        parts.append(
            f"<line x1='{left}' x2='{width - 24}' y1='{y:.1f}' y2='{y:.1f}' "
            f"stroke='{_INK}' stroke-opacity='0.18'/>"
        )
        parts.append(_text(left - 6, y + 4, _fmt(v, unit), anchor="end", size=11))
    for i, lab in enumerate(x_labels):
        parts.append(_text(left + plot_w * i / n, top + plot_h + 18, lab, size=11))
    for k, (name, values) in enumerate(series.items()):
        col = colour(name, k)
        pts = " ".join(
            f"{left + plot_w * i / n:.1f},{top + plot_h - plot_h * v / vmax:.1f}"
            for i, v in enumerate(values)
        )
        parts.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2.5'/>")
        for i, v in enumerate(values):
            parts.append(
                f"<circle cx='{left + plot_w * i / n:.1f}' "
                f"cy='{top + plot_h - plot_h * v / vmax:.1f}' r='3.5' fill='{col}'/>"
            )
        lx = left + k * 150
        parts.append(
            f"<rect x='{lx}' y='{height - 20}' width='12' height='12' rx='2' fill='{col}'/>"
        )
        parts.append(_text(lx + 18, height - 10, name, anchor="start", size=12))
    parts.append("</svg>")
    return "".join(parts)


def _open(width: int, height: int, title: str) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' role='img' aria-label='{escape(title)}' "
        f"{_FONT}><title>{escape(title)}</title>" + _text(width / 2, 22, title, size=15, weight=600)
    )


def _text(
    x: float, y: float, s: str, anchor: str = "middle", size: int = 12, weight: int = 400
) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' font-size='{size}' "
        f"font-weight='{weight}' fill='{_INK}'>{escape(s)}</text>"
    )


def _fmt(v: float, unit: str) -> str:
    if unit == "%":
        return f"{v:.0%}"
    if unit == "$":
        return f"${v:.4f}" if v < 0.1 else f"${v:.2f}"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"
