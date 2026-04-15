#!/usr/bin/env python3
"""Visual analysis for target generation outputs.

Usage examples:
  python scripts/analyze_target_outputs.py
  python scripts/analyze_target_outputs.py --run_id smoke
  python scripts/analyze_target_outputs.py --top_k 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


def with_run_id(base: Path, run_id: Optional[str]) -> Path:
    if not run_id:
        return base
    return base.with_name(f"{base.stem}_{run_id}{base.suffix}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_histograms(target_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, color in [
        (axes[0], "delta_map", "#1f77b4"),
        (axes[1], "delta_hota", "#ff7f0e"),
    ]:
        ax.hist(target_df[col].to_numpy(), bins=40, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "01_delta_histograms.png", dpi=160)
    plt.close(fig)


def save_scatter(target_df: pd.DataFrame, out_dir: Path) -> None:
    color_map = {
        "blur": "#1f77b4",
        "pixelate": "#2ca02c",
        "h264_local": "#d62728",
    }
    colors = target_df["degradation_type"].map(color_map).fillna("#7f7f7f")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        target_df["delta_map"].to_numpy(),
        target_df["delta_hota"].to_numpy(),
        s=10,
        c=colors.to_list(),
        alpha=0.45,
        linewidths=0,
    )
    ax.set_title("delta_map vs delta_hota")
    ax.set_xlabel("delta_map")
    ax.set_ylabel("delta_hota")
    ax.grid(alpha=0.2)
    handles = []
    for k, v in color_map.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=7))
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "02_delta_scatter.png", dpi=160)
    plt.close(fig)


def save_bar_charts(target_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grouped = (
        target_df.groupby("degradation_type", as_index=False)
        .agg(
            count=("clip_id", "count"),
            mean_delta_map=("delta_map", "mean"),
            mean_delta_hota=("delta_hota", "mean"),
            median_delta_map=("delta_map", "median"),
            median_delta_hota=("delta_hota", "median"),
            zero_delta_ratio=(
                "delta_map",
                lambda s: float(
                    (
                        (s.to_numpy() == 0.0)
                        & (target_df.loc[s.index, "delta_hota"].to_numpy() == 0.0)
                    ).mean()
                ),
            ),
        )
        .sort_values("mean_delta_map", ascending=False)
    )
    grouped.to_csv(out_dir / "summary_by_degradation.csv", index=False)

    x = np.arange(len(grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, grouped["mean_delta_map"], width, label="mean_delta_map", color="#1f77b4")
    ax.bar(x + width / 2, grouped["mean_delta_hota"], width, label="mean_delta_hota", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["degradation_type"].tolist())
    ax.set_ylabel("mean delta")
    ax.set_title("Average performance drop by degradation type")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "03_mean_delta_by_degradation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grouped["degradation_type"], grouped["zero_delta_ratio"], color="#9467bd", alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ratio")
    ax.set_title("Zero-delta ratio by degradation type")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "04_zero_delta_ratio_by_degradation.png", dpi=160)
    plt.close(fig)

    return grouped


def save_split_heatmaps(target_df: pd.DataFrame, out_dir: Path) -> None:
    for metric in ["delta_map", "delta_hota"]:
        pivot = (
            target_df.pivot_table(
                index="split",
                columns="degradation_type",
                values=metric,
                aggfunc="mean",
            )
            .sort_index()
            .sort_index(axis=1)
        )

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_title(f"Mean {metric} by split x degradation")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=20, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.iloc[r, c]
                if pd.isna(val):
                    text = "NA"
                else:
                    text = f"{val:.3f}"
                ax.text(c, r, text, ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"05_heatmap_{metric}.png", dpi=160)
        plt.close(fig)


def save_top_k_table(target_df: pd.DataFrame, out_dir: Path, top_k: int) -> pd.DataFrame:
    ranked = target_df.copy()
    ranked["combined_delta"] = ranked["delta_map"] + ranked["delta_hota"]
    top = ranked.sort_values("combined_delta", ascending=False).head(top_k)
    cols = [
        "clip_id",
        "original_clip_id",
        "split",
        "degradation_type",
        "degradation_param",
        "p_orig_map",
        "p_orig_hota",
        "p_anon_map",
        "p_anon_hota",
        "delta_map",
        "delta_hota",
        "combined_delta",
    ]
    top[cols].to_csv(out_dir / "top_k_hardest_clips.csv", index=False)
    return top[cols]


def build_intensity_summary(target_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        target_df.groupby(["degradation_type", "degradation_param"], as_index=False)
        .agg(
            count=("clip_id", "count"),
            mean_delta_map=("delta_map", "mean"),
            mean_delta_hota=("delta_hota", "mean"),
            median_delta_map=("delta_map", "median"),
            median_delta_hota=("delta_hota", "median"),
            zero_delta_ratio=(
                "delta_map",
                lambda s: float(
                    (
                        (s.to_numpy() == 0.0)
                        & (target_df.loc[s.index, "delta_hota"].to_numpy() == 0.0)
                    ).mean()
                ),
            ),
        )
        .sort_values(["degradation_type", "degradation_param"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return grouped


def save_intensity_curves(intensity_df: pd.DataFrame, out_dir: Path) -> None:
    color_map = {
        "blur": "#1f77b4",
        "pixelate": "#2ca02c",
        "h264_local": "#d62728",
    }
    marker_map = {
        "blur": "o",
        "pixelate": "s",
        "h264_local": "^",
    }

    def _plot(metric_col: str, file_name: str, title: str, y_label: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for deg_type, sub in intensity_df.groupby("degradation_type"):
            sub = sub.sort_values("degradation_param")
            ax.plot(
                sub["degradation_param"].to_numpy(),
                sub[metric_col].to_numpy(),
                label=deg_type,
                color=color_map.get(str(deg_type), "#7f7f7f"),
                marker=marker_map.get(str(deg_type), "o"),
                linewidth=2.0,
                markersize=6,
            )
        ax.set_title(title)
        ax.set_xlabel("degradation_param (intensity)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / file_name, dpi=160)
        plt.close(fig)

    _plot(
        metric_col="mean_delta_map",
        file_name="06_intensity_curve_delta_map.png",
        title="Intensity vs mean delta_map",
        y_label="mean delta_map",
    )
    _plot(
        metric_col="mean_delta_hota",
        file_name="07_intensity_curve_delta_hota.png",
        title="Intensity vs mean delta_hota",
        y_label="mean delta_hota",
    )

    intensity_df = intensity_df.copy()
    intensity_df["mean_delta_combined"] = intensity_df["mean_delta_map"] + intensity_df["mean_delta_hota"]
    _plot(
        metric_col="mean_delta_combined",
        file_name="08_intensity_curve_combined.png",
        title="Intensity vs mean (delta_map + delta_hota)",
        y_label="mean combined delta",
    )


def build_intensity_interpretation(intensity_df: pd.DataFrame) -> str:
    if intensity_df.empty:
        return "No intensity data available."

    slopes = []
    for deg_type, sub in intensity_df.groupby("degradation_type"):
        sub = sub.sort_values("degradation_param")
        x = sub["degradation_param"].to_numpy(dtype=float)
        y = (sub["mean_delta_map"] + sub["mean_delta_hota"]).to_numpy(dtype=float)
        if len(x) < 2 or np.allclose(x, x[0]):
            slope = 0.0
        else:
            slope = float(np.polyfit(x, y, 1)[0])
        slopes.append((str(deg_type), slope))
    if not slopes:
        return "No intensity data available."

    slopes_sorted = sorted(slopes, key=lambda t: t[1], reverse=True)
    steepest_type, steepest_slope = slopes_sorted[0]
    gentlest_type, gentlest_slope = slopes_sorted[-1]
    return (
        f"Steepest intensity sensitivity: {steepest_type} (slope={steepest_slope:.4f}), "
        f"gentlest: {gentlest_type} (slope={gentlest_slope:.4f}) "
        "based on linear trend of mean(delta_map + delta_hota) over degradation_param."
    )


def write_summary_files(
    out_dir: Path,
    target_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    top_df: pd.DataFrame,
    fail_df: pd.DataFrame,
    intensity_df: pd.DataFrame,
) -> None:
    summary = {
        "num_targets": int(len(target_df)),
        "num_unique_originals": int(target_df["original_clip_id"].nunique()),
        "num_failures": int(len(fail_df)),
        "delta_map_mean": float(target_df["delta_map"].mean()),
        "delta_hota_mean": float(target_df["delta_hota"].mean()),
        "delta_map_median": float(target_df["delta_map"].median()),
        "delta_hota_median": float(target_df["delta_hota"].median()),
        "zero_delta_ratio": float(((target_df["delta_map"] == 0.0) & (target_df["delta_hota"] == 0.0)).mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Target Output Analysis Summary")
    lines.append("")
    lines.append("## Headline Metrics")
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## By Degradation Type (mean delta)")
    lines.append("")
    lines.append("```")
    lines.append(grouped_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Top Hardest Clips (combined_delta)")
    lines.append("")
    lines.append("```")
    lines.append(top_df.head(10).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## By Intensity (degradation_param)")
    lines.append("")
    lines.append(f"- {build_intensity_interpretation(intensity_df)}")
    lines.append("")
    if intensity_df.empty:
        lines.append("- No intensity summary available.")
    else:
        for deg_type in sorted(intensity_df["degradation_type"].astype(str).unique().tolist()):
            lines.append(f"### {deg_type}")
            lines.append("")
            sub = intensity_df[intensity_df["degradation_type"] == deg_type].copy()
            cols = [
                "degradation_param",
                "count",
                "mean_delta_map",
                "mean_delta_hota",
                "median_delta_map",
                "median_delta_hota",
                "zero_delta_ratio",
            ]
            lines.append("```")
            lines.append(sub[cols].to_string(index=False))
            lines.append("```")
            lines.append("")
    lines.append("## Failure Stage Counts")
    if len(fail_df) == 0:
        lines.append("- {}")
    else:
        fail_counts = fail_df["stage"].value_counts(dropna=False).to_dict()
        for stage, count in fail_counts.items():
            lines.append(f"- {stage}: {count}")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual analysis for 03_generate_targets outputs.")
    parser.add_argument(
        "--manifests_dir",
        type=str,
        default="data/interim/manifests",
        help="Directory containing manifest outputs.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="data/interim/logs",
        help="Directory containing failure logs.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run-id suffix (e.g., smoke).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/reports/target_analysis",
        help="Where to save visualizations and summary tables.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-K hardest clips to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests_dir = Path(args.manifests_dir)
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    target_path = with_run_id(manifests_dir / "target_manifest.csv", args.run_id)
    stats_path = with_run_id(manifests_dir / "target_stats.csv", args.run_id)
    failure_path = with_run_id(logs_dir / "target_failures.csv", args.run_id)

    if not target_path.exists():
        raise FileNotFoundError(f"Missing target manifest: {target_path}")
    target_df = pd.read_csv(target_path)
    if target_df.empty:
        raise RuntimeError(f"Target manifest is empty: {target_path}")

    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        stats_df.to_csv(out_dir / "target_stats_copy.csv", index=False)
    if failure_path.exists():
        fail_df = pd.read_csv(failure_path)
    else:
        fail_df = pd.DataFrame(columns=["stage"])

    if "split" not in target_df.columns:
        target_df["split"] = "unknown"

    save_histograms(target_df, out_dir)
    save_scatter(target_df, out_dir)
    grouped_df = save_bar_charts(target_df, out_dir)
    save_split_heatmaps(target_df, out_dir)
    top_df = save_top_k_table(target_df, out_dir, args.top_k)
    intensity_df = build_intensity_summary(target_df)
    intensity_df.to_csv(out_dir / "intensity_summary_by_type.csv", index=False)
    save_intensity_curves(intensity_df, out_dir)
    write_summary_files(out_dir, target_df, grouped_df, top_df, fail_df, intensity_df)

    print(f"[DONE] Analysis outputs saved to: {out_dir}")
    print("[DONE] Generated files:")
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            print(f" - {p.name}")


if __name__ == "__main__":
    main()
