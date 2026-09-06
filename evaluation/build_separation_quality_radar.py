#!/usr/bin/env python3
"""Build two category-wise separation-quality radars from canonical outputs."""

from __future__ import print_function

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_SPECS = (
    {
        "model": "DARS",
        # Muted, colour-blind-conscious academic palette. DARS is the anchor.
        "color": "#3E6584",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.70,
        # Keep the thicker solid trace underneath thin dashed baselines so that
        # coincident paths remain visible through their dash gaps.
        "zorder": 3,
    },
    {
        "model": "SPMamba",
        "color": "#A8875A",
        "linestyle": (0, (4.6, 2.3)),
        "marker": "D",
        "linewidth": 1.05,
        "zorder": 5,
    },
    {
        "model": "TDANet-Large",
        "color": "#668780",
        # Draw the restrained dash-dot trace above DARS.  The wider gaps keep
        # both curves visible when their values coincide without offsetting data.
        "linestyle": (0, (4.4, 2.6, 1.0, 2.6)),
        "marker": "^",
        "linewidth": 1.05,
        "zorder": 6,
    },
    {
        "model": "TF-Locoformer-M",
        "color": "#8B788D",
        "linestyle": (0, (1.7, 1.7)),
        "marker": "s",
        "linewidth": 1.05,
        "zorder": 4,
    },
)

CATEGORIES = (
    ("dialog+dialog", "D&D"),
    ("dialog+music", "D&M"),
    ("dialog+speech", "D&S"),
    ("dialog+tv", "D&T"),
    ("music+music", "M&M"),
    ("music+speech", "M&S"),
    ("music+tv", "M&T"),
    ("speech+speech", "S&S"),
    ("speech+tv", "S&T"),
    ("tv+tv", "T&T"),
)

METRIC_SPECS = (
    {
        "column": "sdr_i",
        "label": "SDRi",
        "title": "(a) Mean SDRi (dB)",
        "radial_min_db": -6.0,
        "radial_max_db": 18.0,
        "radial_ticks_db": np.arange(-6.0, 18.1, 4.0),
    },
    {
        "column": "sir",
        "label": "SIR",
        "title": "(b) Mean SIR (dB)",
        "radial_min_db": 0.0,
        "radial_max_db": 30.0,
        "radial_ticks_db": np.arange(0.0, 30.1, 5.0),
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="results/category_separation_metrics.csv",
        type=Path,
    )
    parser.add_argument(
        "--pdf",
        default="outputs/figures/separation_quality_radar.pdf",
        type=Path,
    )
    parser.add_argument(
        "--png",
        default="outputs/figures/separation_quality_radar.png",
        type=Path,
    )
    return parser.parse_args()


def load_summary(path):
    summary = pd.read_csv(path)
    required = {
        "model",
        "metric",
        "category",
        "axis_label",
        "n",
        "mean_db",
        "ci95_low_db",
        "ci95_high_db",
        "sample_rate_hz",
        "segment_seconds",
        "evaluation_n",
        "alignment",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, sorted(missing)))
    expected_models = {spec["model"] for spec in MODEL_SPECS}
    expected_metrics = {metric["label"] for metric in METRIC_SPECS}
    expected_categories = {category for category, _ in CATEGORIES}
    if set(summary["model"]) != expected_models:
        raise ValueError("Unexpected model set in {}".format(path))
    if set(summary["metric"]) != expected_metrics:
        raise ValueError("Unexpected metric set in {}".format(path))
    if set(summary["category"]) != expected_categories:
        raise ValueError("Unexpected category set in {}".format(path))
    if summary.duplicated(["model", "metric", "category"]).any():
        raise ValueError("Duplicate model/metric/category rows in {}".format(path))
    expected_rows = (
        len(expected_models) * len(expected_metrics) * len(expected_categories)
    )
    if len(summary) != expected_rows:
        raise ValueError("Expected {} rows in {}, got {}".format(expected_rows, path, len(summary)))
    numeric = summary[["mean_db", "ci95_low_db", "ci95_high_db"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite category metric in {}".format(path))
    if set(summary["alignment"]) != {"fixed identity"}:
        raise ValueError("Category metrics are not fixed-identity results")
    return summary


def _closed_xy(radii, angles):
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.r_[x, x[0]], np.r_[y, y[0]]


def _data_to_radius(values, radial_min_db, radial_max_db):
    values = np.asarray(values, dtype=float)
    return (values - radial_min_db) / (radial_max_db - radial_min_db)


def plot_panel(axis, summary, metric):
    labels = [axis_label for _, axis_label in CATEGORIES]
    categories = [category for category, _ in CATEGORIES]
    radial_min_db = metric["radial_min_db"]
    radial_max_db = metric["radial_max_db"]
    radial_ticks_db = metric["radial_ticks_db"]
    # Start at 12 o'clock and proceed clockwise.  A custom Cartesian radar
    # gives true decagonal grid rings, which align with the ten category axes.
    angles = np.pi / 2.0 - np.linspace(
        0.0, 2.0 * np.pi, len(labels), endpoint=False
    )

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.17, 1.17)
    axis.axis("off")

    spoke_color = "#E7E9EC"
    ring_color = "#DEE2E6"
    outer_color = "#C2C8CE"
    text_color = "#34383C"
    secondary_text = "#777D83"

    unit_radii = np.ones(len(angles), dtype=float)
    outer_x, outer_y = _closed_xy(unit_radii, angles)
    for angle in angles:
        axis.plot(
            [0.0, np.cos(angle)],
            [0.0, np.sin(angle)],
            color=spoke_color,
            linewidth=0.34,
            zorder=0,
        )

    for tick in radial_ticks_db:
        radius = float(
            _data_to_radius([tick], radial_min_db, radial_max_db)[0]
        )
        ring_x, ring_y = _closed_xy(np.full(len(angles), radius), angles)
        is_outer = np.isclose(tick, radial_max_db)
        axis.plot(
            ring_x,
            ring_y,
            color=outer_color if is_outer else ring_color,
            linewidth=0.52 if is_outer else 0.34,
            zorder=0,
        )

    # Place the radial values between two spokes, with a small white knockout
    # so the scale remains legible where model traces pass behind it.
    radial_label_angle = np.deg2rad(72.0)
    for tick in radial_ticks_db:
        radius = float(
            _data_to_radius([tick], radial_min_db, radial_max_db)[0]
        )
        label_radius = max(radius, 0.035)
        axis.text(
            label_radius * np.cos(radial_label_angle),
            label_radius * np.sin(radial_label_angle),
            "{:g}".format(tick),
            ha="center",
            va="center",
            fontsize=5.4,
            color=secondary_text,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12},
            zorder=8,
        )

    for label, angle in zip(labels, angles):
        x = 1.075 * np.cos(angle)
        y = 1.075 * np.sin(angle)
        if x > 0.16:
            horizontal = "left"
        elif x < -0.16:
            horizontal = "right"
        else:
            horizontal = "center"
        if y > 0.16:
            vertical = "bottom"
        elif y < -0.16:
            vertical = "top"
        else:
            vertical = "center"
        axis.text(
            x,
            y,
            label,
            ha=horizontal,
            va=vertical,
            fontsize=6.7,
            fontweight="normal",
            color=text_color,
        )

    axis.set_title(
        metric["title"],
        fontsize=8.2,
        fontweight="normal",
        pad=4,
        color=text_color,
    )

    metric_rows = summary.loc[summary["metric"] == metric["label"]]
    for spec in MODEL_SPECS:
        values = (
            metric_rows.loc[metric_rows["model"] == spec["model"]]
            .set_index("category")
            .loc[categories, "mean_db"]
            .to_numpy(dtype=float)
        )
        radii = _data_to_radius(values, radial_min_db, radial_max_db)
        closed_x, closed_y = _closed_xy(radii, angles)
        axis.plot(
            closed_x,
            closed_y,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=spec["linewidth"],
            marker=spec["marker"],
            markersize=2.45 if spec["model"] == "DARS" else 2.05,
            markerfacecolor="white",
            markeredgecolor=spec["color"],
            markeredgewidth=0.55,
            label=spec["model"],
            zorder=spec["zorder"],
        )


def plot_radar(summary, pdf_path, png_path):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.8,
            "axes.linewidth": 0.75,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.84),
    )
    figure.patch.set_facecolor("white")
    for axis, metric in zip(axes, METRIC_SPECS):
        axis.set_facecolor("white")
        plot_panel(axis, summary, metric)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=4,
        frameon=False,
        handlelength=2.15,
        handletextpad=0.38,
        columnspacing=1.10,
        markerscale=0.78,
        borderaxespad=0.0,
        fontsize=6.7,
    )
    figure.text(
        0.5,
        0.014,
        "D: dialogue,   M: music,   S: speech,   T: TV speech",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#7A8086",
    )
    figure.subplots_adjust(
        left=0.040, right=0.960, top=0.835, bottom=0.115, wspace=0.10
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.018)
    figure.savefig(png_path, bbox_inches="tight", pad_inches=0.018)
    plt.close(figure)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    input_csv = args.input_csv if args.input_csv.is_absolute() else repo_root / args.input_csv
    pdf_path = args.pdf if args.pdf.is_absolute() else repo_root / args.pdf
    png_path = args.png if args.png.is_absolute() else repo_root / args.png

    summary = load_summary(input_csv)
    plot_radar(summary, pdf_path, png_path)

    print("Loaded {} rows from {}".format(len(summary), input_csv))
    for metric in METRIC_SPECS:
        rows = summary.loc[summary["metric"] == metric["label"]]
        piv = rows.pivot(index="category", columns="model", values="mean_db")
        margins = piv["DARS"] - piv[[spec["model"] for spec in MODEL_SPECS[1:]]].max(axis=1)
        print(
            "{}: DARS best in {}/{} categories; margin range {:+.3f} to {:+.3f} dB".format(
                metric["label"],
                int((margins >= 0.0).sum()),
                len(margins),
                margins.min(),
                margins.max(),
            )
        )
    print("Wrote {}".format(pdf_path))
    print("Wrote {}".format(png_path))


if __name__ == "__main__":
    main()
