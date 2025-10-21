"""
Test script for ViT extractor integration.
Tests the complete pipeline: image loading → feature extraction → database writing
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_vit_extractor_basic():
    """Test 1: Basic ViT extractor initialization and inference."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic ViT Extractor Initialization")
    print("=" * 60)

    from vit_colmap.features.vit_extractor import ViTExtractor

    try:
        # Initialize extractor
        print("\n1. Initializing ViT extractor...")
        extractor = ViTExtractor(
            model_name="dinov2_vitb14",
            num_keypoints=1024,
            descriptor_dim=128,
            device="cpu",  # Use CPU for testing
        )
        print("   ✓ Extractor initialized successfully")

        # Create a test image
        print("\n2. Creating test image...")
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"   ✓ Test image shape: {test_img.shape}")

        # Run inference
        print("\n3. Running inference...")
        keypoints, descriptors = extractor._run_inference(test_img)

        # Verify outputs
        print("\n4. Results:")
        print(f"   - Keypoints shape: {keypoints.shape}")
        print(f"   - Keypoints dtype: {keypoints.dtype}")
        print(f"   - Descriptors shape: {descriptors.shape}")
        print(f"   - Descriptors dtype: {descriptors.dtype}")

        # Validate
        assert keypoints.dtype == np.float32, "Keypoints should be float32"
        assert descriptors.dtype == np.uint8, "Descriptors should be uint8"
        assert keypoints.shape[1] == 2, "Keypoints should have 2 columns (x, y)"
        assert descriptors.shape[1] == 128, "Descriptors should be 128-dimensional"
        assert len(keypoints) == len(
            descriptors
        ), "Keypoints and descriptors must match"

        print("\n✓ TEST 1 PASSED")
        return True

    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vit_extractor_full_pipeline():
    """Test 2: Full pipeline with directory processing and database writing."""
    print("\n" + "=" * 60)
    print("TEST 2: Full Pipeline (Directory → Database)")
    print("=" * 60)

    from vit_colmap.features.vit_extractor import ViTExtractor
    import cv2
    import tempfile
    import shutil

    try:
        # Create temporary directories
        temp_dir = Path(tempfile.mkdtemp())
        image_dir = temp_dir / "images"
        image_dir.mkdir()
        db_path = temp_dir / "test.db"

        print("\n1. Setup:")
        print(f"   - Temp directory: {temp_dir}")
        print(f"   - Image directory: {image_dir}")
        print(f"   - Database path: {db_path}")

        # Generate test images
        print("\n2. Generating 3 test images...")
        for i in range(3):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = image_dir / f"test_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)
            print(f"   ✓ Created {img_path.name}")

        # Initialize extractor
        print("\n3. Initializing ViT extractor...")
        extractor = ViTExtractor(
            model_name="dinov2_vitb14",
            num_keypoints=512,  # Fewer for testing speed
            descriptor_dim=128,
            device="cpu",
        )

        # Run extraction
        print("\n4. Running feature extraction pipeline...")
        extractor.extract(
            image_dir=image_dir,
            db_path=db_path,
            camera_model="SIMPLE_PINHOLE",
            camera_params=None,  # Will use defaults
        )

        # Verify database
        print("\n5. Verifying database...")
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM images")
        n_images = cursor.fetchone()[0]
        print(f"   - Images in database: {n_images}")

        cursor.execute("SELECT COUNT(*) FROM keypoints")
        n_keypoints = cursor.fetchone()[0]
        print(f"   - Keypoint entries: {n_keypoints}")

        cursor.execute("SELECT COUNT(*) FROM descriptors")
        n_descriptors = cursor.fetchone()[0]
        print(f"   - Descriptor entries: {n_descriptors}")

        conn.close()

        # Validate
        assert n_images == 3, f"Expected 3 images, got {n_images}"
        assert n_keypoints == 3, f"Expected 3 keypoint entries, got {n_keypoints}"
        assert n_descriptors == 3, f"Expected 3 descriptor entries, got {n_descriptors}"

        # Cleanup
        shutil.rmtree(temp_dir)
        print("\n6. Cleaned up temporary files")

        print("\n✓ TEST 2 PASSED")
        return True

    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        # Cleanup on failure
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def test_compare_with_dummy():
    """Test 3: Compare ViT extractor output format with DummyExtractor."""
    print("\n" + "=" * 60)
    print("TEST 3: Compare with DummyExtractor (Format Validation)")
    print("=" * 60)

    from vit_colmap.features.vit_extractor import ViTExtractor
    from vit_colmap.features.dummy_extractor import DummyExtractor
    import cv2
    import tempfile
    import shutil

    try:
        # Create temporary directories
        temp_dir = Path(tempfile.mkdtemp())
        image_dir = temp_dir / "images"
        image_dir.mkdir()

        # Create single test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = image_dir / "test.jpg"
        cv2.imwrite(str(img_path), img)

        # Test with DummyExtractor
        print("\n1. Running DummyExtractor...")
        db_dummy = temp_dir / "dummy.db"
        dummy = DummyExtractor()
        dummy.extract(image_dir, db_dummy, "SIMPLE_PINHOLE")

        # Test with ViTExtractor
        print("\n2. Running ViTExtractor...")
        db_vit = temp_dir / "vit.db"
        vit = ViTExtractor(model_name="dinov2_vitb14", num_keypoints=512, device="cpu")
        vit.extract(image_dir, db_vit, "SIMPLE_PINHOLE")

        # Compare database structures
        print("\n3. Comparing database structures...")
        import sqlite3

        def get_db_info(db_path):
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT rows, cols FROM keypoints WHERE image_id=1")
            kp_info = cursor.fetchone()

            cursor.execute("SELECT rows, cols FROM descriptors WHERE image_id=1")
            desc_info = cursor.fetchone()

            conn.close()
            return kp_info, desc_info

        dummy_kp, dummy_desc = get_db_info(db_dummy)
        vit_kp, vit_desc = get_db_info(db_vit)

        print("\n   DummyExtractor:")
        print(f"     - Keypoints: ({dummy_kp[0]}, {dummy_kp[1]})")
        print(f"     - Descriptors: ({dummy_desc[0]}, {dummy_desc[1]})")

        print("\n   ViTExtractor:")
        print(f"     - Keypoints: ({vit_kp[0]}, {vit_kp[1]})")
        print(f"     - Descriptors: ({vit_desc[0]}, {vit_desc[1]})")

        # Validate format compatibility
        assert vit_kp[1] == dummy_kp[1], "Keypoint columns must match"
        assert vit_desc[1] == dummy_desc[1], "Descriptor dimensions must match"

        print("\n✓ Formats are compatible!")

        # Cleanup
        shutil.rmtree(temp_dir)

        print("\n✓ TEST 3 PASSED")
        return True

    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ViT Extractor Integration Test Suite")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Basic Initialization", test_vit_extractor_basic()))
    results.append(("Full Pipeline", test_vit_extractor_full_pipeline()))
    results.append(("Format Compatibility", test_compare_with_dummy()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")

    all_passed = all(result[1] for result in results)

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Test on real images from your dataset")
        print("2. Run matching pipeline")
        print("3. Compare with COLMAP SIFT baseline")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nCheck the error messages above for details")
        return 1


if __name__ == "__main__":
    exit(main())
