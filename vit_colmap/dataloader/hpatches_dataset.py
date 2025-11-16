"""
HPatches dataset loader for training feature extractors.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Optional, Tuple, List


class HPatchesDataset(Dataset):
    """
    PyTorch Dataset for HPatches image pairs with ground truth homographies.

    HPatches dataset structure:
        HPatches/
        ├── i_ajuntament/     # Illumination changes (i_*)
        │   ├── 1.ppm         # Reference image
        │   ├── 2.ppm         # Target image
        │   ├── H_1_2         # Homography from 1 to 2
        │   └── ...
        ├── v_adam/           # Viewpoint changes (v_*)
        │   └── ...
        └── ...

    Each sample returns:
        - img1: Reference image tensor (3, H, W)
        - img2: Target image tensor (3, H, W)
        - H: Homography matrix (3, 3) mapping img1 -> img2
        - seq_name: Sequence name string
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "all",
        target_size: Tuple[int, int] = (1200, 1600),
        patch_size: int = 14,
        transform: Optional[transforms.Compose] = None,
        max_pairs_per_sequence: int = 5,
    ):
        """
        Initialize HPatches dataset.

        Args:
            root_dir: Path to HPatches root directory
            split: Dataset split - "all", "illumination" (i_*), "viewpoint" (v_*), or "train"/"test"
            target_size: Target image size (H, W) - will be adjusted to be divisible by patch_size
            patch_size: ViT patch size (default 14 for DINOv2)
            transform: Optional additional transforms
            max_pairs_per_sequence: Maximum number of image pairs per sequence (2-6)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.max_pairs_per_sequence = min(
            max_pairs_per_sequence, 5
        )  # HPatches has images 1-6

        # Adjust target size to be divisible by patch_size
        self.target_h = (target_size[0] // patch_size) * patch_size
        self.target_w = (target_size[1] // patch_size) * patch_size

        # Default transform: normalize to ImageNet stats
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Discover sequences
        self.sequences = self._discover_sequences()

        # Build list of all image pairs
        self.pairs = self._build_pair_list()

        print("HPatchesDataset initialized:")
        print(f"  Root: {self.root_dir}")
        print(f"  Split: {self.split}")
        print(f"  Sequences: {len(self.sequences)}")
        print(f"  Image pairs: {len(self.pairs)}")
        print(f"  Target size: {self.target_h}x{self.target_w}")

    def _discover_sequences(self) -> List[Path]:
        """Discover all valid sequences in the dataset."""
        if not self.root_dir.exists():
            raise ValueError(f"HPatches root directory not found: {self.root_dir}")

        sequences = []
        for seq_dir in sorted(self.root_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            # Filter by split
            seq_name = seq_dir.name
            if self.split == "illumination" and not seq_name.startswith("i_"):
                continue
            if self.split == "viewpoint" and not seq_name.startswith("v_"):
                continue
            if self.split == "train":
                # Use illumination sequences for training
                if not seq_name.startswith("i_"):
                    continue
            if self.split == "test":
                # Use viewpoint sequences for testing
                if not seq_name.startswith("v_"):
                    continue

            # Check if sequence has required files
            ref_img = self._find_image(seq_dir, "1")
            if ref_img is None:
                continue

            sequences.append(seq_dir)

        return sequences

    def _find_image(self, seq_dir: Path, img_num: str) -> Optional[Path]:
        """Find image file with given number (supports .ppm, .png, .jpg)."""
        for ext in [".ppm", ".png", ".jpg", ".jpeg"]:
            img_path = seq_dir / f"{img_num}{ext}"
            if img_path.exists():
                return img_path
        return None

    def _build_pair_list(self) -> List[Tuple[Path, int]]:
        """Build list of (sequence_path, target_img_num) pairs."""
        pairs = []
        for seq_dir in self.sequences:
            # For each sequence, pair image 1 with images 2-6
            for img_num in range(2, 2 + self.max_pairs_per_sequence):
                img_path = self._find_image(seq_dir, str(img_num))
                h_path = seq_dir / f"H_1_{img_num}"

                if img_path is not None and h_path.exists():
                    pairs.append((seq_dir, img_num))

        return pairs

    def _load_image(self, img_path: Path) -> np.ndarray:
        """Load and resize image to target size."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(
            img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
        )

        return img

    def _load_homography(self, h_path: Path, orig_size: Tuple[int, int]) -> np.ndarray:
        """
        Load homography matrix and adjust for image resizing.

        Args:
            h_path: Path to homography file
            orig_size: Original image size (H, W)

        Returns:
            Adjusted homography matrix (3, 3)
        """
        # Load original homography
        H = np.loadtxt(str(h_path)).reshape(3, 3).astype(np.float32)

        # Adjust homography for resizing
        # H' = S2 @ H @ S1_inv
        # where S1 scales original to target, S2 scales target back
        orig_h, orig_w = orig_size
        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h

        # Scaling matrix
        S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)
        S_inv = np.array(
            [[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1]], dtype=np.float32
        )

        # Adjusted homography: scale -> apply H -> scale back
        H_adjusted = S @ H @ S_inv

        return H_adjusted

    def _get_original_size(self, seq_dir: Path) -> Tuple[int, int]:
        """Get original image size from reference image."""
        ref_path = self._find_image(seq_dir, "1")
        if ref_path is None:
            raise ValueError(f"Reference image not found in {seq_dir}")

        img = cv2.imread(str(ref_path))
        if img is None:
            raise ValueError(f"Failed to load reference image: {ref_path}")

        return img.shape[:2]  # (H, W)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Get image pair with homography.

        Returns:
            Dictionary containing:
                - img1: Reference image tensor (3, H, W)
                - img2: Target image tensor (3, H, W)
                - H: Homography matrix (3, 3)
                - seq_name: Sequence name
                - pair_idx: Target image index (2-6)
        """
        seq_dir, target_idx = self.pairs[idx]

        # Load images
        ref_path = self._find_image(seq_dir, "1")
        target_path = self._find_image(seq_dir, str(target_idx))

        if ref_path is None or target_path is None:
            raise ValueError(f"Image files not found for {seq_dir}")

        # Get original size for homography adjustment
        orig_size = self._get_original_size(seq_dir)

        # Load and process images
        img1 = self._load_image(ref_path)
        img2 = self._load_image(target_path)

        # Load and adjust homography
        h_path = seq_dir / f"H_1_{target_idx}"
        H = self._load_homography(h_path, orig_size)

        # Apply transforms
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        H_tensor = torch.from_numpy(H).float()

        return {
            "img1": img1_tensor,
            "img2": img2_tensor,
            "H": H_tensor,
            "seq_name": seq_dir.name,
            "pair_idx": target_idx,
        }

    def get_sequence_info(self) -> dict:
        """Get information about loaded sequences."""
        illumination_seqs = [s for s in self.sequences if s.name.startswith("i_")]
        viewpoint_seqs = [s for s in self.sequences if s.name.startswith("v_")]

        return {
            "total_sequences": len(self.sequences),
            "illumination_sequences": len(illumination_seqs),
            "viewpoint_sequences": len(viewpoint_seqs),
            "total_pairs": len(self.pairs),
            "target_size": (self.target_h, self.target_w),
        }
