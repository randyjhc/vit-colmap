#!/usr/bin/env python3
"""Standalone script to visualize training and validation losses from log files.

This script loads training log files and generates loss curve plots.
Can be run anytime after training to visualize or re-generate plots.

Usage:
    python scripts/plot_training_losses.py --log-file logs/vit_features_20251116_232027.log
    python scripts/plot_training_losses.py --log-dir logs
    python scripts/plot_training_losses.py --log-dir logs --output-dir plots --show
    python scripts/plot_training_losses.py --log-dir logs --plot-type components
"""

import argparse
import logging
from pathlib import Path

from vit_colmap.utils.plot_training import TrainingLossPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training and validation losses from log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot losses from a specific log file
  python scripts/plot_training_losses.py --log-file logs/vit_features_20251116_232027.log

  # Auto-detect latest log file from directory
  python scripts/plot_training_losses.py --log-dir logs

  # Save plots to a specific directory
  python scripts/plot_training_losses.py --log-dir logs --output-dir plots

  # Show plots interactively
  python scripts/plot_training_losses.py --log-dir logs --show

  # Plot only specific type
  python scripts/plot_training_losses.py --log-dir logs --plot-type curve
  python scripts/plot_training_losses.py --log-dir logs --plot-type components

  # Use single row layout for components
  python scripts/plot_training_losses.py --log-dir logs --plot-type components --layout single_row
        """,
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Specific log file to parse",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing log files (auto-selects latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: same as log file directory)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["all", "curve", "components"],
        default="all",
        help="Type of plot to generate (default: all)",
    )
    parser.add_argument(
        "--layout",
        choices=["grid", "single_row"],
        default="grid",
        help="Layout for loss components plot: 'grid' (2x3) or 'single_row' (1x4) (default: grid)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (blocks until windows are closed)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload log data (ignore cache)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.log_file is None and args.log_dir is None:
        logger.error("Either --log-file or --log-dir must be specified")
        return 1

    if args.log_file and not args.log_file.exists():
        logger.error("Log file does not exist: %s", args.log_file)
        return 1

    if args.log_dir and not args.log_dir.exists():
        logger.error("Log directory does not exist: %s", args.log_dir)
        return 1

    # Initialize plotter
    try:
        plotter = TrainingLossPlotter(log_file=args.log_file, log_dir=args.log_dir)
    except Exception as e:
        logger.error("Failed to initialize plotter: %s", e)
        return 1

    logger.info("Using log file: %s", plotter.log_file)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = plotter.log_file.parent / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Plots will be saved to: %s", output_dir)

    # Load history to check if data exists
    history = plotter.load_training_history(force_reload=args.force_reload)

    if not history["epochs"]:
        logger.error("No training history found in log file")
        logger.error("Please ensure the log file contains training/validation data")
        return 1

    num_epochs = len(history["epochs"])
    has_val = history["val"] is not None

    logger.info("Found training data for %d epochs", num_epochs)
    if has_val:
        num_val_epochs = len(history["val"]["epochs"])
        logger.info("Validation data available for %d epochs", num_val_epochs)
    else:
        logger.warning("No validation data found in log file")

    # Generate requested plots
    results = {}

    if args.plot_type in ["all", "curve"]:
        logger.info("Generating loss curve plot...")
        loss_curve_path = output_dir / "loss_curve.png"
        result = plotter.plot_loss_curves(
            output_path=loss_curve_path,
            show=args.show,
        )
        if result:
            results["loss_curve"] = result
            logger.info("Saved loss curve to: %s", result)

    if args.plot_type in ["all", "components"]:
        logger.info("Generating loss components plot...")
        loss_components_path = output_dir / "loss_components.png"
        result = plotter.plot_loss_components(
            output_path=loss_components_path,
            show=args.show,
            layout=args.layout,
        )
        if result:
            results["loss_components"] = result
            logger.info("Saved loss components to: %s", result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Log file: %s", plotter.log_file)
    logger.info("Epochs processed: %d", num_epochs)
    logger.info("Plots generated: %d", len(results))
    for plot_name, plot_path in results.items():
        logger.info("  - %s: %s", plot_name, plot_path)
    logger.info("=" * 60)

    if not results:
        logger.warning("No plots were generated")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
