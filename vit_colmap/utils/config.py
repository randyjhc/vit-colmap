"""Configuration and settings management for vit-colmap."""

import logging
from dataclasses import dataclass, field
from typing import Optional
import pycolmap


@dataclass
class LogConfig:
    """Logging configuration settings."""

    level: int = logging.INFO
    format: str = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    datefmt: str = "%H:%M:%S"

    def apply(self):
        """Apply logging configuration."""
        # Configure Python logging
        logging.basicConfig(
            level=self.level,
            format=self.format,
            datefmt=self.datefmt,
            force=True,  # Override any existing configuration
        )


@dataclass
class CameraConfig:
    """Camera model configuration."""

    model: str = "SIMPLE_PINHOLE"
    width: Optional[int] = None
    height: Optional[int] = None
    params: Optional[list[float]] = None

    def get_default_params(self, width: int, height: int) -> list[float]:
        """Get default camera parameters based on model and image size."""
        if self.params is not None:
            return self.params

        if self.model == "SIMPLE_PINHOLE":
            # f, cx, cy
            f = max(width, height)
            return [f, width / 2.0, height / 2.0]
        elif self.model == "PINHOLE":
            # fx, fy, cx, cy
            f = max(width, height)
            return [f, f, width / 2.0, height / 2.0]
        else:
            raise ValueError(f"Unsupported camera model: {self.model}")


@dataclass
class MatchingConfig:
    """Feature matching configuration."""

    use_gpu: bool = False
    max_ratio: float = 0.8
    max_distance: float = 0.7
    cross_check: bool = True
    num_threads: int = 1  # -1 means auto-detect

    def to_sift_options(self):
        """Convert to pycolmap SiftMatchingOptions."""

        opts = pycolmap.SiftMatchingOptions()
        opts.use_gpu = self.use_gpu
        opts.max_ratio = self.max_ratio
        opts.max_distance = self.max_distance
        opts.cross_check = self.cross_check
        opts.num_threads = self.num_threads
        return opts


@dataclass
class ReconstructionConfig:
    """3D reconstruction configuration."""

    min_num_matches: int = 15
    multiple_models: bool = False

    def to_mapper_options(self):
        """Convert to pycolmap IncrementalPipelineOptions."""

        opts = pycolmap.IncrementalPipelineOptions()
        opts.min_num_matches = self.min_num_matches
        opts.multiple_models = self.multiple_models
        return opts


@dataclass
class Config:
    """Main configuration class for vit-colmap pipeline."""

    # Logging
    log: LogConfig = field(default_factory=LogConfig)

    # Camera
    camera: CameraConfig = field(default_factory=CameraConfig)

    # Matching
    matching: MatchingConfig = field(default_factory=MatchingConfig)

    # Reconstruction
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    # Pipeline flags
    do_matching: bool = True
    do_reconstruction: bool = True

    def __post_init__(self):
        """Apply configuration after initialization."""
        # Apply logging configuration immediately
        self.log.apply()

    @classmethod
    def from_args(cls, args):
        """Create Config from argparse arguments."""
        config = cls()

        # Update camera config
        if hasattr(args, "camera_model"):
            config.camera.model = args.camera_model

        # Update matching config
        if hasattr(args, "use_gpu"):
            config.matching.use_gpu = args.use_gpu

        # Update pipeline flags
        if hasattr(args, "skip_matching"):
            config.do_matching = not args.skip_matching
        if hasattr(args, "skip_reconstruction"):
            config.do_reconstruction = not args.skip_reconstruction

        # Update logging
        if hasattr(args, "verbose") and args.verbose:
            config.log.level = logging.DEBUG
            config.log.apply()  # Reapply with new level

        return config

    def summary(self) -> str:
        """Get configuration summary."""
        lines = [
            "Configuration:",
            f"  Camera model: {self.camera.model}",
            f"  Matching: {'enabled' if self.do_matching else 'disabled'}",
            f"  Reconstruction: {'enabled' if self.do_reconstruction else 'disabled'}",
            f"  GPU matching: {'enabled' if self.matching.use_gpu else 'disabled'}",
            f"  Min matches: {self.reconstruction.min_num_matches}",
        ]
        return "\n".join(lines)
