#!/usr/bin/env python3
"""Test script for simplified BEiT extractor."""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vit_colmap.features.beit_extractor import BEiTExtractor


def main():
    print("=" * 60)
    print("Testing Simplified BEiT Extractor")
    print("=" * 60)
    print()

    # Configuration
    test_image = Path("data/raw/test_image.png")
    output_dir = Path("outputs/beit_simplified")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        print("Please provide a valid test image")
        return

    print(f"Test image: {test_image}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize extractor
    print("Initializing BEiT extractor...")
    extractor = BEiTExtractor(
        model_name="microsoft/beit-base-patch16-224-pt22k-ft22k",
        layer_idx=9,  # Middle layer
    )

    # Test 1: RGB Reconstruction
    print("\n" + "=" * 60)
    print("Test 1: RGB Reconstruction Visualization")
    print("=" * 60)
    print()

    try:
        output_rgb = output_dir / "layer9_rgb_channels_012.png"
        extractor.visualize_layer_rgb(
            test_image,
            output_rgb,
            channels=[0, 1, 2],  # First 3 channels
        )
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: RGB with different channels
    print("\n" + "=" * 60)
    print("Test 2: RGB with Different Channels")
    print("=" * 60)
    print()

    try:
        output_rgb2 = output_dir / "layer9_rgb_channels_345.png"
        extractor.visualize_layer_rgb(
            test_image,
            output_rgb2,
            channels=[3, 4, 5],  # Channels 3, 4, 5
        )
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Heatmap Visualization
    print("\n" + "=" * 60)
    print("Test 3: Heatmap Visualization")
    print("=" * 60)
    print()

    try:
        output_heatmap = output_dir / "layer9_heatmap.png"
        extractor.visualize_layer_heatmap(test_image, output_heatmap)
        print()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    # Test different layers
    print("\n" + "=" * 60)
    print("Test 4: Different Layers")
    print("=" * 60)
    print()

    for layer_idx in [0, 6, 11]:
        try:
            print(f"\nTesting layer {layer_idx}...")
            extractor_layer = BEiTExtractor(
                model_name="microsoft/beit-base-patch16-224-pt22k-ft22k",
                layer_idx=layer_idx,
            )

            output_layer = output_dir / f"layer{layer_idx}_heatmap.png"
            extractor_layer.visualize_layer_heatmap(test_image, output_layer)
            print()
        except Exception as e:
            print(f"✗ Error for layer {layer_idx}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    expected_files = [
        "layer9_rgb_channels_012.png",
        "layer9_rgb_channels_345.png",
        "layer9_heatmap.png",
        "layer0_heatmap.png",
        "layer6_heatmap.png",
        "layer11_heatmap.png",
    ]

    print("\nGenerated files:")
    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename:<35} ({size:.2f} MB)")
        else:
            print(f"  ✗ {filename:<35} (missing)")
            all_exist = False

    if all_exist:
        print("\n✓ All tests passed!")
        print(f"\nView results at: {output_dir}")
    else:
        print("\n✗ Some tests failed")

    print("=" * 60)


if __name__ == "__main__":
    main()
