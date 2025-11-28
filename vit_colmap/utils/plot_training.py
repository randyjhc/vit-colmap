"""Plotting utilities for training and validation loss visualization from log files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TrainingLossPlotter:
    """Class for visualizing training and validation losses from log files."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        color_train: Optional[Tuple[float, ...]] = None,
        color_val: Optional[Tuple[float, ...]] = None,
    ):
        """Initialize the TrainingLossPlotter.

        Args:
            log_file: Path to specific log file to parse
            log_dir: Directory containing log files (auto-selects latest if log_file not provided)
            color_train: RGBA tuple for training curves (defaults to tab10 color 0)
            color_val: RGBA tuple for validation curves (defaults to tab10 color 1)

        Note: Either log_file or log_dir must be provided. If both are given, log_file takes precedence.
        """
        if log_file is None and log_dir is None:
            raise ValueError("Either log_file or log_dir must be provided")

        if log_file is not None:
            self.log_file = log_file.resolve()
        else:
            # Auto-detect latest log file from directory
            self.log_file = self._find_latest_log(log_dir)

        self.color_train = color_train or plt.get_cmap("tab10")(0)
        self.color_val = color_val or plt.get_cmap("tab10")(1)
        self._history_cache: Optional[Dict] = None

        # Regex patterns for parsing log lines
        # Updated to capture detector components: Det: X.XX (Score: Y.YY, Orient: Z.ZZ)
        self.train_pattern = re.compile(
            r"Epoch (\d+) completed.*?"
            r"Avg Loss: ([\d.]+).*?"
            r"Det: ([\d.]+)\s+\(Score: ([\d.]+), Orient: ([\d.]+)\).*?"
            r"Desc: ([\d.]+)"
        )
        self.val_pattern = re.compile(
            r"Validation \| "
            r"Loss: ([\d.]+).*?"
            r"Det: ([\d.]+)\s+\(Score: ([\d.]+), Orient: ([\d.]+)\).*?"
            r"Desc: ([\d.]+)"
        )

    @staticmethod
    def _find_latest_log(log_dir: Path) -> Path:
        """Find the latest log file in the directory.

        Args:
            log_dir: Directory to search

        Returns:
            Path to the latest log file

        Raises:
            FileNotFoundError: If no log files found
        """
        log_dir = log_dir.resolve()
        if not log_dir.exists():
            raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

        # Find all .log files
        log_files = list(log_dir.glob("*.log"))
        if not log_files:
            raise FileNotFoundError(f"No log files found in {log_dir}")

        # Sort by modification time, return latest
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        logger.info("Auto-selected latest log file: %s", latest_log.name)
        return latest_log

    def load_training_history(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load training history from log file.

        Args:
            force_reload: If True, reload from disk even if cached

        Returns:
            Dictionary with structure:
            {
                'epochs': [1, 2, 3, ...],
                'train': {
                    'total': [...],
                    'detector': [...],
                    'detector_score': [...],
                    'detector_orient': [...],
                    'descriptor': [...]
                },
                'val': {
                    'epochs': [1, 2, 3, ...],
                    'total': [...],
                    'detector': [...],
                    'detector_score': [...],
                    'detector_orient': [...],
                    'descriptor': [...]
                }
            }
        """
        if self._history_cache is not None and not force_reload:
            return self._history_cache

        if not self.log_file.exists():
            logger.warning("Log file does not exist: %s", self.log_file)
            return {
                "epochs": [],
                "train": {
                    "total": [],
                    "detector": [],
                    "detector_score": [],
                    "detector_orient": [],
                    "descriptor": [],
                },
                "val": None,
            }

        epochs: List[int] = []
        train_losses: Dict[str, List[float]] = {
            "total": [],
            "detector": [],
            "detector_score": [],
            "detector_orient": [],
            "descriptor": [],
        }
        val_epochs: List[int] = []
        val_losses_data: Dict[str, List[float]] = {
            "total": [],
            "detector": [],
            "detector_score": [],
            "detector_orient": [],
            "descriptor": [],
        }

        logger.info("Parsing log file: %s", self.log_file)

        try:
            with open(self.log_file, "r") as f:
                current_epoch = None

                for line in f:
                    # Try to match training loss line
                    train_match = self.train_pattern.search(line)
                    if train_match:
                        epoch = int(train_match.group(1))
                        total_loss = float(train_match.group(2))
                        det_loss = float(train_match.group(3))
                        det_score_loss = float(train_match.group(4))
                        det_orient_loss = float(train_match.group(5))
                        desc_loss = float(train_match.group(6))

                        epochs.append(epoch)
                        train_losses["total"].append(total_loss)
                        train_losses["detector"].append(det_loss)
                        train_losses["detector_score"].append(det_score_loss)
                        train_losses["detector_orient"].append(det_orient_loss)
                        train_losses["descriptor"].append(desc_loss)

                        current_epoch = epoch
                        continue

                    # Try to match validation loss line
                    val_match = self.val_pattern.search(line)
                    if val_match and current_epoch is not None:
                        total_loss = float(val_match.group(1))
                        det_loss = float(val_match.group(2))
                        det_score_loss = float(val_match.group(3))
                        det_orient_loss = float(val_match.group(4))
                        desc_loss = float(val_match.group(5))

                        val_epochs.append(current_epoch)
                        val_losses_data["total"].append(total_loss)
                        val_losses_data["detector"].append(det_loss)
                        val_losses_data["detector_score"].append(det_score_loss)
                        val_losses_data["detector_orient"].append(det_orient_loss)
                        val_losses_data["descriptor"].append(desc_loss)
                        continue

        except Exception as e:
            logger.error("Error parsing log file %s: %s", self.log_file, e)
            return {
                "epochs": [],
                "train": {
                    "total": [],
                    "detector": [],
                    "detector_score": [],
                    "detector_orient": [],
                    "descriptor": [],
                },
                "val": None,
            }

        # Prepare validation losses with separate epoch tracking
        val_losses = None
        if val_losses_data["total"]:
            val_losses = {
                "epochs": val_epochs,
                "total": val_losses_data["total"],
                "detector": val_losses_data["detector"],
                "detector_score": val_losses_data["detector_score"],
                "detector_orient": val_losses_data["detector_orient"],
                "descriptor": val_losses_data["descriptor"],
            }
            logger.info(
                "Loaded training data for %d epochs, validation data for %d epochs",
                len(epochs),
                len(val_epochs),
            )
        else:
            logger.info(
                "Loaded training data for %d epochs (no validation data)", len(epochs)
            )

        history = {
            "epochs": epochs,
            "train": train_losses,
            "val": val_losses,
        }

        self._history_cache = history
        return history

    def plot_loss_curves(
        self,
        output_path: Optional[Path] = None,
        show: bool = False,
        title: str = "Training and Validation Loss",
    ) -> Optional[Path]:
        """Plot total loss curves for training and validation.

        Args:
            output_path: Optional path to save the figure
            show: Whether to display the plot interactively
            title: Plot title

        Returns:
            The output path if specified, None otherwise
        """
        history = self.load_training_history()

        if not history["epochs"]:
            logger.warning("No training history available to plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = history["epochs"]
        train_total = history["train"]["total"]

        ax.plot(
            epochs,
            train_total,
            color=self.color_train,
            linewidth=2,
            marker="o",
            markersize=4,
            label="Training Loss",
        )

        # Plot validation loss if available
        if history["val"] is not None:
            val_epochs = history["val"]["epochs"]
            val_total = history["val"]["total"]
            ax.plot(
                val_epochs,
                val_total,
                color=self.color_val,
                linewidth=2,
                marker="s",
                markersize=4,
                label="Validation Loss",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

        fig.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved loss curve to %s", output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def plot_loss_components(
        self,
        output_path: Optional[Path] = None,
        show: bool = False,
        title: str = "Loss Components",
    ) -> Optional[Path]:
        """Plot individual loss components including detector breakdown.

        Args:
            output_path: Optional path to save the figure
            show: Whether to display the plot interactively
            title: Plot title

        Returns:
            The output path if specified, None otherwise
        """
        history = self.load_training_history()

        if not history["epochs"]:
            logger.warning("No training history available to plot")
            return None

        # 2x3 grid: top row (total, detector, descriptor), bottom row (score, orient, empty)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        epochs = history["epochs"]

        # Define plots: (row, col, loss_type, title)
        plots = [
            (0, 0, "total", "Total Loss"),
            (0, 1, "detector", "Detector Loss (Combined)"),
            (0, 2, "descriptor", "Descriptor Loss"),
            (1, 0, "detector_score", "Detector: Score Component"),
            (1, 1, "detector_orient", "Detector: Orientation Component"),
        ]

        for row, col, loss_type, subplot_title in plots:
            ax = axes[row, col]

            train_losses = history["train"][loss_type]

            ax.plot(
                epochs,
                train_losses,
                color=self.color_train,
                linewidth=2,
                marker="o",
                markersize=3,
                label="Training",
            )

            # Plot validation loss if available
            if history["val"] is not None:
                val_epochs = history["val"]["epochs"]
                val_losses = history["val"][loss_type]
                ax.plot(
                    val_epochs,
                    val_losses,
                    color=self.color_val,
                    linewidth=2,
                    marker="s",
                    markersize=3,
                    label="Validation",
                )

            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.set_title(subplot_title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))

        # Hide the unused subplot (bottom right)
        axes[1, 2].axis("off")

        fig.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Saved loss components plot to %s", output_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def plot_all(
        self,
        output_dir: Optional[Path] = None,
        show: bool = False,
    ) -> Dict[str, Optional[Path]]:
        """Generate all available plots.

        Args:
            output_dir: Optional directory to save plots
            show: Whether to display plots interactively

        Returns:
            Dictionary mapping plot names to their output paths
        """
        results = {}

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            loss_curve_path = output_dir / "loss_curve.png"
            loss_components_path = output_dir / "loss_components.png"
        else:
            loss_curve_path = None
            loss_components_path = None

        results["loss_curve"] = self.plot_loss_curves(
            output_path=loss_curve_path,
            show=show,
        )

        results["loss_components"] = self.plot_loss_components(
            output_path=loss_components_path,
            show=show,
        )

        return results

    def clear_cache(self) -> None:
        """Clear the history cache to force reload on next access."""
        self._history_cache = None


__all__ = ["TrainingLossPlotter"]
