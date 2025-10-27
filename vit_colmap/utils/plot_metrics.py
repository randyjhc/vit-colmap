"""Plotting utilities for comparing SIFT and ViT SfM metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from vit_colmap.utils.export import MetricsExporter
from vit_colmap.utils.metrics import MetricsResult

logger = logging.getLogger(__name__)


class MetricsPlotter:
    """Class for creating comparative visualization plots of SIFT vs ViT metrics."""

    def __init__(
        self,
        results_dir: Path,
        color_sift: Optional[Tuple[float, ...]] = None,
        color_vit: Optional[Tuple[float, ...]] = None,
        enable_cache: bool = True,
    ):
        """Initialize the MetricsPlotter.

        Args:
            results_dir: Directory containing scan subdirectories with metrics JSON files
            color_sift: RGBA tuple for SIFT bars (defaults to tab10 color 0)
            color_vit: RGBA tuple for ViT bars (defaults to tab10 color 1)
            enable_cache: Whether to cache loaded metrics for performance
        """
        self.results_dir = results_dir.resolve()
        self.color_sift = color_sift or plt.get_cmap("tab10")(0)
        self.color_vit = color_vit or plt.get_cmap("tab10")(1)
        self.enable_cache = enable_cache
        self._metrics_cache: Dict[
            str, Tuple[Optional[MetricsResult], Optional[MetricsResult]]
        ] = {}

    def _load_metrics_pair(
        self, scan_name: str
    ) -> Tuple[Optional[MetricsResult], Optional[MetricsResult]]:
        """Load SIFT and ViT metrics for a scan, with optional caching.

        Args:
            scan_name: Name of the scan subdirectory

        Returns:
            Tuple of (sift_metrics, vit_metrics), where either can be None if missing
        """
        if self.enable_cache and scan_name in self._metrics_cache:
            return self._metrics_cache[scan_name]

        scan_dir = self.results_dir / scan_name
        sift_path = scan_dir / "sift.json"
        vit_path = scan_dir / "vit.json"

        sift_metrics: Optional[MetricsResult] = None
        vit_metrics: Optional[MetricsResult] = None

        if sift_path.exists():
            sift_metrics = MetricsExporter.load_json(sift_path)
        else:
            logger.warning("Missing SIFT metrics for %s at %s", scan_name, sift_path)

        if vit_path.exists():
            vit_metrics = MetricsExporter.load_json(vit_path)
        else:
            logger.warning("Missing ViT metrics for %s at %s", scan_name, vit_path)

        result = (sift_metrics, vit_metrics)
        if self.enable_cache:
            self._metrics_cache[scan_name] = result

        return result

    @staticmethod
    def _feature_value(metrics: Optional[MetricsResult], attr: str) -> float:
        """Safely extract a feature metric value, defaulting to zero."""
        if metrics is None:
            return 0.0
        return float(getattr(metrics.features, attr))

    @staticmethod
    def _matching_value(metrics: Optional[MetricsResult], attr: str) -> float:
        """Safely extract a matching metric value, defaulting to zero."""
        if metrics is None:
            return 0.0
        return float(getattr(metrics.matching, attr))

    @staticmethod
    def _reconstruction_value(metrics: Optional[MetricsResult], attr: str) -> float:
        """Safely extract a reconstruction metric value, defaulting to zero."""
        if metrics is None or metrics.reconstruction is None:
            return 0.0
        return float(getattr(metrics.reconstruction, attr))

    @staticmethod
    def _compute_ratio_pair(
        sift_value: float, vit_value: float, epsilon: float = 1e-9
    ) -> Tuple[float, float]:
        """Return ratios relative to the SIFT baseline (1.0 for SIFT).

        Args:
            sift_value: SIFT baseline value
            vit_value: ViT value to compare
            epsilon: Small value to avoid division by zero

        Returns:
            Tuple of (sift_ratio, vit_ratio) where sift_ratio is always 1.0
        """
        baseline = float(sift_value)
        vit_val = float(vit_value)

        if abs(baseline) <= epsilon:
            if abs(vit_val) <= epsilon:
                return 1.0, 1.0
            logger.warning(
                "SIFT baseline near zero; ratio undefined (sift=%.3f, vit=%.3f).",
                baseline,
                vit_val,
            )
            return 1.0, float("nan")

        return 1.0, vit_val / baseline

    @staticmethod
    def _format_bar_value(value: float) -> str:
        """Format bar labels with adaptive precision."""
        if value == 0 or abs(value) >= 1000:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def _annotate_bars_with_values(
        self, ax: plt.Axes, bars: Iterable, values: Iterable[float]
    ) -> None:
        """Place raw value labels above bars for the given axis."""
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                self._format_bar_value(float(value)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    def _plot_ratio_panel(
        self,
        ax: plt.Axes,
        labels: List[str],
        sift_raw: List[float],
        vit_raw: List[float],
        title: str,
    ) -> None:
        """Render a ratio comparison panel with bars annotated with raw values.

        Args:
            ax: Matplotlib axis to plot on
            labels: X-axis labels for each metric
            sift_raw: Raw SIFT values
            vit_raw: Raw ViT values
            title: Panel title
        """
        sift_ratios: List[float] = []
        vit_ratios: List[float] = []
        for s_val, v_val in zip(sift_raw, vit_raw):
            ratio_s, ratio_v = self._compute_ratio_pair(s_val, v_val)
            sift_ratios.append(ratio_s)
            vit_ratios.append(ratio_v)

        indices = np.arange(len(labels))
        width = 0.35
        sift_ratios_arr = np.nan_to_num(np.array(sift_ratios, dtype=float), nan=0.0)
        vit_ratios_arr = np.nan_to_num(np.array(vit_ratios, dtype=float), nan=0.0)

        bars_sift = ax.bar(
            indices - width / 2,
            sift_ratios_arr,
            width=width,
            color=self.color_sift,
            label="SIFT (baseline)",
        )
        bars_vit = ax.bar(
            indices + width / 2,
            vit_ratios_arr,
            width=width,
            color=self.color_vit,
            label="ViT",
        )

        ax.set_xticks(indices)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Ratio vs SIFT")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

        combined = (
            np.concatenate([sift_ratios_arr, vit_ratios_arr])
            if len(labels)
            else np.array([1.0])
        )
        max_ratio = float(np.nanmax(combined))
        max_ratio = max(1.05, max_ratio * 1.1)
        ax.set_ylim(0, max_ratio)

        ax.legend()
        self._annotate_bars_with_values(ax, bars_sift, sift_raw)
        self._annotate_bars_with_values(ax, bars_vit, vit_raw)

    def plot_single_scan(
        self,
        scan_name: str,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Create a detailed 3-panel comparison plot for a single scan.

        Args:
            scan_name: Name of the scan subdirectory
            output_path: Optional path to save the figure
            show: Whether to display the plot interactively

        Returns:
            The output path if specified, None otherwise
        """
        sift_metrics, vit_metrics = self._load_metrics_pair(scan_name)

        if sift_metrics is None and vit_metrics is None:
            logger.warning(
                "No metrics found for %s in %s; generating zero-valued plot.",
                scan_name,
                self.results_dir,
            )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Metrics Comparison for {scan_name}")

        # Feature metrics
        feature_labels = ["Total KP", "Avg KP/img", "Min KP", "Max KP"]
        feature_sift_raw = [
            self._feature_value(sift_metrics, "total_keypoints"),
            self._feature_value(sift_metrics, "avg_keypoints_per_image"),
            self._feature_value(sift_metrics, "min_keypoints"),
            self._feature_value(sift_metrics, "max_keypoints"),
        ]
        feature_vit_raw = [
            self._feature_value(vit_metrics, "total_keypoints"),
            self._feature_value(vit_metrics, "avg_keypoints_per_image"),
            self._feature_value(vit_metrics, "min_keypoints"),
            self._feature_value(vit_metrics, "max_keypoints"),
        ]
        self._plot_ratio_panel(
            axes[0],
            feature_labels,
            feature_sift_raw,
            feature_vit_raw,
            "Feature Extraction",
        )

        # Matching metrics
        matching_labels = [
            "Matched pairs",
            "Match rate (%)",
            "Avg raw matches",
            "Avg inlier matches",
            "Inlier ratio",
        ]
        matching_sift_raw = [
            self._matching_value(sift_metrics, "matched_pairs"),
            self._matching_value(sift_metrics, "match_rate"),
            self._matching_value(sift_metrics, "avg_raw_matches"),
            self._matching_value(sift_metrics, "avg_inlier_matches"),
            self._matching_value(sift_metrics, "inlier_ratio"),
        ]
        matching_vit_raw = [
            self._matching_value(vit_metrics, "matched_pairs"),
            self._matching_value(vit_metrics, "match_rate"),
            self._matching_value(vit_metrics, "avg_raw_matches"),
            self._matching_value(vit_metrics, "avg_inlier_matches"),
            self._matching_value(vit_metrics, "inlier_ratio"),
        ]
        self._plot_ratio_panel(
            axes[1],
            matching_labels,
            matching_sift_raw,
            matching_vit_raw,
            "Feature Matching",
        )

        # Reconstruction metrics
        reconstruction_labels = [
            "Registered cams",
            "Registration rate (%)",
            "3D points",
            "Avg track length",
            "Avg reproj error",
        ]
        reconstruction_sift_raw = [
            self._reconstruction_value(sift_metrics, "registered_images"),
            self._reconstruction_value(sift_metrics, "registration_rate"),
            self._reconstruction_value(sift_metrics, "total_3d_points"),
            self._reconstruction_value(sift_metrics, "avg_track_length"),
            self._reconstruction_value(sift_metrics, "avg_reprojection_error"),
        ]
        reconstruction_vit_raw = [
            self._reconstruction_value(vit_metrics, "registered_images"),
            self._reconstruction_value(vit_metrics, "registration_rate"),
            self._reconstruction_value(vit_metrics, "total_3d_points"),
            self._reconstruction_value(vit_metrics, "avg_track_length"),
            self._reconstruction_value(vit_metrics, "avg_reprojection_error"),
        ]
        self._plot_ratio_panel(
            axes[2],
            reconstruction_labels,
            reconstruction_sift_raw,
            reconstruction_vit_raw,
            "3D Reconstruction",
        )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            logger.info("Saved single scan comparison to %s", output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return output_path

    def plot_multiple_scans(
        self,
        scans: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Optional[Path]:
        """Plot concise metrics (3D points, inlier ratio, registered cams) across scans.

        Args:
            scans: List of scan names to plot. If None, auto-discovers from results_dir
            output_path: Optional path to save the figure
            show: Whether to display the plot interactively

        Returns:
            The output path if specified, None otherwise
        """
        if scans is None:
            if self.results_dir.exists():
                scans = sorted(
                    p.name
                    for p in self.results_dir.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                )
            else:
                scans = []

        series: Dict[str, Dict[str, List[float]]] = {
            "Total 3D Points": {
                "sift_raw": [],
                "vit_raw": [],
                "sift_ratio": [],
                "vit_ratio": [],
            },
            "Inlier Ratio (RANSAC)": {
                "sift_raw": [],
                "vit_raw": [],
                "sift_ratio": [],
                "vit_ratio": [],
            },
            "Registered Cameras": {
                "sift_raw": [],
                "vit_raw": [],
                "sift_ratio": [],
                "vit_ratio": [],
            },
        }
        display_scans: List[str] = []
        missing_scans: Set[str] = set()

        for scan in scans:
            display_scans.append(scan)
            sift_metrics, vit_metrics = self._load_metrics_pair(scan)

            missing = False
            if sift_metrics is None or vit_metrics is None:
                missing = True
                logger.warning(
                    "Using zero placeholders for %s due to missing extractor metrics.",
                    scan,
                )
            if (
                sift_metrics is None
                or vit_metrics is None
                or sift_metrics.reconstruction is None
                or vit_metrics.reconstruction is None
            ):
                missing = True
                if (
                    sift_metrics is not None
                    and vit_metrics is not None
                    and (
                        sift_metrics.reconstruction is None
                        or vit_metrics.reconstruction is None
                    )
                ):
                    logger.warning(
                        "Using zero placeholders for %s due to missing reconstruction metrics.",
                        scan,
                    )
            if missing:
                missing_scans.add(scan)

            # Total 3D points
            sift_points = self._reconstruction_value(sift_metrics, "total_3d_points")
            vit_points = self._reconstruction_value(vit_metrics, "total_3d_points")
            ratio_s, ratio_v = self._compute_ratio_pair(sift_points, vit_points)
            series["Total 3D Points"]["sift_raw"].append(sift_points)
            series["Total 3D Points"]["vit_raw"].append(vit_points)
            series["Total 3D Points"]["sift_ratio"].append(ratio_s)
            series["Total 3D Points"]["vit_ratio"].append(ratio_v)

            # Inlier ratio
            sift_inlier = self._matching_value(sift_metrics, "inlier_ratio")
            vit_inlier = self._matching_value(vit_metrics, "inlier_ratio")
            ratio_s, ratio_v = self._compute_ratio_pair(sift_inlier, vit_inlier)
            series["Inlier Ratio (RANSAC)"]["sift_raw"].append(sift_inlier)
            series["Inlier Ratio (RANSAC)"]["vit_raw"].append(vit_inlier)
            series["Inlier Ratio (RANSAC)"]["sift_ratio"].append(ratio_s)
            series["Inlier Ratio (RANSAC)"]["vit_ratio"].append(ratio_v)

            # Registered cameras
            sift_registered = self._reconstruction_value(
                sift_metrics, "registered_images"
            )
            vit_registered = self._reconstruction_value(
                vit_metrics, "registered_images"
            )
            ratio_s, ratio_v = self._compute_ratio_pair(sift_registered, vit_registered)
            series["Registered Cameras"]["sift_raw"].append(sift_registered)
            series["Registered Cameras"]["vit_raw"].append(vit_registered)
            series["Registered Cameras"]["sift_ratio"].append(ratio_s)
            series["Registered Cameras"]["vit_ratio"].append(ratio_v)

        if not display_scans:
            logger.warning(
                "No scans found in %s; generating placeholder zero plot.",
                self.results_dir,
            )
            display_scans = ["N/A"]
            for metric in series.values():
                metric["sift_raw"] = [0.0]
                metric["vit_raw"] = [0.0]
                metric["sift_ratio"] = [1.0]
                metric["vit_ratio"] = [1.0]
            missing_scans = {"N/A"}

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
        fig.suptitle("SIFT vs ViT Metrics Across Scans")

        indices = np.arange(len(display_scans))
        width = 0.35

        for ax, (metric_name, values) in zip(axes, series.items()):
            sift_ratios = np.nan_to_num(
                np.array(values["sift_ratio"], dtype=float), nan=0.0
            )
            vit_ratios = np.nan_to_num(
                np.array(values["vit_ratio"], dtype=float), nan=0.0
            )

            bars_sift = ax.bar(
                indices - width / 2,
                sift_ratios,
                width=width,
                label="SIFT (baseline)",
                color=self.color_sift,
            )
            bars_vit = ax.bar(
                indices + width / 2,
                vit_ratios,
                width=width,
                label="ViT",
                color=self.color_vit,
            )
            ax.set_title(metric_name)
            ax.set_xticks(indices)
            ax.set_xticklabels(display_scans, rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            combined = (
                np.concatenate([sift_ratios, vit_ratios])
                if display_scans
                else np.array([1.0])
            )
            max_ratio = float(np.nanmax(combined))
            max_ratio = max(1.05, max_ratio * 1.1)
            ax.set_ylim(0, max_ratio)
            self._annotate_bars_with_values(ax, bars_sift, values["sift_raw"])
            self._annotate_bars_with_values(ax, bars_vit, values["vit_raw"])

        axes[0].set_ylabel("Ratio vs SIFT")
        axes[0].legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if missing_scans:
            missing_list = ", ".join(sorted(missing_scans))
            note = (
                f"Zero-height ratio bars indicate missing metrics for: {missing_list}"
            )
            fig.text(0.5, 0.02, note, ha="center", fontsize=9)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
            logger.info("Saved scans summary plot to %s", output_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return output_path

    def clear_cache(self) -> None:
        """Clear the metrics cache to free memory."""
        self._metrics_cache.clear()


__all__ = ["MetricsPlotter"]
