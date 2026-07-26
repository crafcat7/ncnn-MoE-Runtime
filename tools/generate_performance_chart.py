#!/usr/bin/env python3

import argparse
import html
import json
import sys
from pathlib import Path


COLORS = {
    "cold": "#2563eb",
    "warm": "#94a3b8",
    "reference": "#0f766e",
    "text": "#334155",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "rule": "#cbd5e1",
}


REFERENCE_DATA = {
    "hybrid": [
        {"label": "20B · 32 tok", "cold": 14.103, "warm": 14.855},
        {"label": "20B · 256 tok", "cold": 17.069, "warm": 16.597},
        {"label": "120B · 32 tok · 1S", "cold": 2.885, "warm": 4.718},
        {"label": "120B · 256 tok · 1S", "cold": 4.971, "warm": 5.171},
        {"label": "120B · 32 tok · 2S", "cold": 5.284, "warm": 10.916},
        {"label": "120B · 256 tok · 2S", "cold": 9.870, "warm": 10.295},
    ],
    "storage": [
        {"label": "1 GiB · 32 tok", "cold": 1.042, "warm": 1.050},
        {"label": "10 GiB · 32 tok", "cold": 1.812, "warm": 2.006},
        {"label": "16 GiB · 32 tok", "cold": 2.119, "warm": 2.444},
        {"label": "1 GiB · 256 tok", "cold": 1.143, "warm": 1.133},
        {"label": "10 GiB · 256 tok", "cold": 2.145, "warm": 2.160},
        {"label": "16 GiB · 256 tok", "cold": 2.637, "warm": 2.636},
    ],
    "logical_reads": [
        {"label": "1 GiB · 32 tok", "cold": 73.9, "warm": 73.8},
        {"label": "10 GiB · 32 tok", "cold": 34.7, "warm": 29.5},
        {"label": "16 GiB · 32 tok", "cold": 26.6, "warm": 20.8},
        {"label": "1 GiB · 256 tok", "cold": 500.0, "warm": 500.0},
        {"label": "10 GiB · 256 tok", "cold": 183.7, "warm": 182.1},
        {"label": "16 GiB · 256 tok", "cold": 115.0, "warm": 115.4},
    ],
    "physical_reads": [
        {"label": "1 GiB · 32 tok", "cold": 57.8, "warm": 114.4},
        {"label": "10 GiB · 32 tok", "cold": 26.0, "warm": 50.6},
        {"label": "16 GiB · 32 tok", "cold": 19.1, "warm": 37.6},
        {"label": "1 GiB · 256 tok", "cold": 392.1, "warm": 783.5},
        {"label": "10 GiB · 256 tok", "cold": 140.8, "warm": 285.5},
        {"label": "16 GiB · 256 tok", "cold": 88.4, "warm": 174.8},
    ],
    "short_reference": [
        {"label": "120B cold · 1×32", "reference": 3.294},
        {"label": "120B warm · 1×32", "reference": 11.045},
        {"label": "120B concurrent · 2×16", "reference": 20.510},
        {"label": "120B long concurrent · 2×96", "reference": 9.778},
        {"label": "20B eager · 1×64", "reference": 16.898},
    ],
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate the published GPT-OSS benchmark SVG."
    )
    parser.add_argument(
        "--report",
        default="build-gptoss-vulkan-msvc/performance-matrix-unified/performance-matrix.json",
        help="Unified matrix JSON; published values are used when absent.",
    )
    parser.add_argument(
        "--output",
        default="assets/gpt-oss-performance.svg",
        help="Output SVG path.",
    )
    return parser.parse_args()


def from_report(path):
    if not path.is_file():
        return None

    try:
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        results = {
            result["case"]["name"]: result["report"]
            for result in aggregate["results"]
        }
        hybrid = []
        for name, label in (
            ("gpt-oss-20b-single-short", "20B · 32 tok"),
            ("gpt-oss-20b-single-long", "20B · 256 tok"),
            ("gpt-oss-120b-single-short", "120B · 32 tok · 1S"),
            ("gpt-oss-120b-single-long", "120B · 256 tok · 1S"),
            ("gpt-oss-120b-service-2xshort", "120B · 32 tok · 2S"),
            ("gpt-oss-120b-service-2xlong", "120B · 256 tok · 2S"),
        ):
            hybrid.append(
                {
                    "label": label,
                    "cold": results[f"{name}-cold"]["median"][
                        "decode_tokens_per_second"
                    ],
                    "warm": results[f"{name}-warm"]["median"][
                        "decode_tokens_per_second"
                    ],
                }
            )

        storage = []
        logical_reads = []
        physical_reads = []
        for window in ("short", "long"):
            cold_rows = {
                row["expert_cache_mb"]: row
                for row in results[f"gpt-oss-120b-offload-{window}-cold"][
                    "rows"
                ]
            }
            warm_rows = {
                row["expert_cache_mb"]: row
                for row in results[f"gpt-oss-120b-offload-{window}-warm"][
                    "rows"
                ]
            }
            tokens = 32 if window == "short" else 256
            for cache_mb in (1024, 10240, 16384):
                label = f"{cache_mb // 1024} GiB · {tokens} tok"
                cold = cold_rows[cache_mb]
                warm = warm_rows[cache_mb]
                storage.append(
                    {
                        "label": label,
                        "cold": cold["decode_tokens_per_second"],
                        "warm": warm["decode_tokens_per_second"],
                    }
                )
                logical_reads.append(
                    {
                        "label": label,
                        "cold": cold["runtime_logical_read_bytes"] / 1e9,
                        "warm": warm["runtime_logical_read_bytes"] / 1e9,
                    }
                )
                physical_reads.append(
                    {
                        "label": label,
                        "cold": cold["system_physical_read_bytes"] / 1e9,
                        "warm": warm["system_physical_read_bytes"] / 1e9,
                    }
                )

        return {
            "hybrid": hybrid,
            "storage": storage,
            "logical_reads": logical_reads,
            "physical_reads": physical_reads,
            "short_reference": REFERENCE_DATA["short_reference"],
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(
            f"warning: cannot parse {path}: {error}; using published values",
            file=sys.stderr,
        )
        return None


def escape(value):
    return html.escape(str(value), quote=True)


def svg_text(x, y, value, size=14, weight=400, fill=None, anchor="start"):
    color = fill or COLORS["text"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def format_value(value, unit):
    if unit.startswith("token/s"):
        return f"{value + 1e-9:.2f}"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def bar_panel(
    x, y, width, height, title, unit, rows, maximum, series_names=("cold", "warm")
):
    parts = []
    parts.append(
        f'<g role="group" aria-label="{escape(title)}">'
        f'<title>{escape(title)}</title>'
    )
    parts.append(svg_text(x, y + 22, title, size=18, weight=600))
    parts.append(svg_text(x + width, y + 22, unit, size=12, fill=COLORS["muted"], anchor="end"))

    left = x + 168
    right = x + width - 20
    top = y + 54
    bottom = y + height - 38
    plot_width = right - left
    row_height = (bottom - top) / len(rows)
    bar_height = min(8.0, max(5.0, row_height / 3.5))

    for tick in range(5):
        value = maximum * tick / 4
        tick_x = left + plot_width * tick / 4
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{top - 8:.1f}" '
            f'x2="{tick_x:.1f}" y2="{bottom:.1f}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                tick_x,
                bottom + 19,
                format_value(value, unit),
                size=11,
                fill=COLORS["muted"],
                anchor="middle",
            )
        )

    for row_index, row in enumerate(rows):
        center = top + row_height * row_index + row_height / 2
        parts.append(svg_text(x + 8, center + 4, row["label"], size=11))
        total_height = len(series_names) * bar_height + (len(series_names) - 1) * 3
        first_bar_y = center - total_height / 2
        for series_index, series in enumerate(series_names):
            bar_y = first_bar_y + series_index * (bar_height + 3)
            value = float(row[series])
            bar_width = plot_width * value / maximum
            parts.append(
                f'<rect x="{left:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="2" fill="{COLORS[series]}">'
                f'<title>{escape(row["label"])} {series}: {format_value(value, unit)}'
                f' {escape(unit)}</title></rect>'
            )
            label_x = left + bar_width + 5
            anchor = "start"
            if label_x > right - 26:
                label_x = left + bar_width - 5
                anchor = "end"
            parts.append(
                svg_text(
                    label_x,
                    bar_y + bar_height - 0.5,
                    format_value(value, unit),
                    size=10,
                    fill=COLORS["text"],
                    anchor=anchor,
                )
            )

    parts.append(
        f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" '
        f'y2="{bottom:.1f}" stroke="{COLORS["rule"]}" stroke-width="1"/>'
    )
    parts.append("</g>")
    return "".join(parts)


def build_svg(data):
    width = 1400
    height = 1450
    margin = 42
    gap = 42
    panel_width = (width - margin * 2 - gap) / 2
    panel_height = 492
    lower_y = 610
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="chart-title chart-desc">',
        '<title id="chart-title">GPT-OSS unified performance matrix</title>',
        '<desc id="chart-desc">Cold and warm throughput, Runtime logical reads, '
        'sampled system physical reads, and short-text reference points for the '
        'GPT-OSS benchmark matrix.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        svg_text(margin, 44, "GPT-OSS unified performance matrix", size=27, weight=600),
        svg_text(
            margin,
            70,
            "Ryzen 7 9800X3D · 31.14 GiB RAM · RTX 5070 Ti 16 GiB · "
            "fixed 16-token prompt · three-run medians",
            size=13,
            fill=COLORS["muted"],
        ),
        f'<rect x="{width - 250}" y="31" width="14" height="10" rx="2" '
        f'fill="{COLORS["cold"]}"/>',
        svg_text(width - 229, 41, "cold", size=12),
        f'<rect x="{width - 168}" y="31" width="14" height="10" rx="2" '
        f'fill="{COLORS["warm"]}"/>',
        svg_text(width - 147, 41, "warm", size=12),
        bar_panel(
            margin,
            95,
            panel_width,
            panel_height,
            "Hybrid Runtime throughput",
            "token/s",
            data["hybrid"],
            20,
        ),
        bar_panel(
            margin + panel_width + gap,
            95,
            panel_width,
            panel_height,
            "CPU Expert storage control",
            "token/s",
            data["storage"],
            3.2,
        ),
        bar_panel(
            margin,
            lower_y,
            panel_width,
            panel_height,
            "Runtime logical Expert reads",
            "GB",
            data["logical_reads"],
            550,
        ),
        bar_panel(
            margin + panel_width + gap,
            lower_y,
            panel_width,
            panel_height,
            "Sampled system physical reads",
            "GB · total-disk estimate",
            data["physical_reads"],
            850,
        ),
        bar_panel(
            margin,
            1125,
            width - margin * 2,
            270,
            "Short-text reference points",
            "token/s · median",
            data["short_reference"],
            22,
            ("reference",),
        ),
        svg_text(
            margin,
            height - 17,
            "Matrix: short = 32 tokens, long = 256 tokens · reference points use "
            "their named workloads",
            size=12,
            fill=COLORS["muted"],
        ),
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main():
    arguments = parse_arguments()
    report_path = Path(arguments.report)
    data = from_report(report_path) or REFERENCE_DATA
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(data), encoding="utf-8")
    print(f"wrote {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
