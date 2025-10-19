from pathlib import Path
import numpy as np
import cv2
import pycolmap

from vit_colmap.pipeline.run_pipeline import Pipeline
from vit_colmap.utils.config import Config


def _make_checkerboard(w=640, h=480, tile=40):
    """Create a checkerboard pattern image for testing."""
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                img[y : y + tile, x : x + tile] = 255
    return img


def test_pipeline_integration(tmp_path: Path):
    """Test the Pipeline class that orchestrates the full workflow."""
    # 1) Create test images
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    base = _make_checkerboard()

    # Create 3 images with translations
    for i, shift in enumerate([(0, 0), (50, 30), (100, 60)]):
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        moved = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]))
        img_path = img_dir / f"image_{i:03d}.png"
        cv2.imwrite(str(img_path), moved)

    # 2) Configure and run the pipeline
    output_dir = tmp_path / "output"
    db_path = tmp_path / "database.db"

    # Create config with test settings - use DummyExtractor via config
    config = Config()
    config.camera.model = "PINHOLE"
    config.extractor.extractor_type = "dummy"  # Use dummy extractor for testing
    config.do_matching = True
    config.do_reconstruction = False  # Skip reconstruction for dummy features

    pipeline = Pipeline(config=config)
    result = pipeline.run(
        image_dir=img_dir,
        output_dir=output_dir,
        db_path=db_path,
    )

    # 3) Verify outputs
    assert db_path.exists(), "Database file should be created"
    assert output_dir.exists(), "Output directory should be created"
    assert result is None, "Should return None when do_reconstruction=False"

    # 4) Check database contents
    db = pycolmap.Database(str(db_path))
    assert db.num_cameras >= 1, "Should have at least one camera"
    assert db.num_images == 3, f"Expected 3 images, got {db.num_images}"
    assert db.num_matched_image_pairs >= 1, "Should have matched image pairs"

    # Verify all images have features
    for img_id in range(1, 4):
        assert db.exists_keypoints(img_id), f"Image {img_id} should have keypoints"
        assert db.exists_descriptors(img_id), f"Image {img_id} should have descriptors"

    print(
        f"✓ Pipeline test passed: {db.num_images} images processed, {db.num_matched_image_pairs} pairs matched"
    )
