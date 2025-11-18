# Visualize Matches - User Guide

A tool for visualizing matched keypoints between two images from a COLMAP database with color-coded inlier/outlier distinction.

## Features

- **Color-coded match visualization**: Green lines for geometrically verified inliers, red lines for outliers
- **Flexible image selection**: Select images by name or index
- **Comprehensive statistics**: Display total keypoints, matched pairs, inlier ratio, and outlier count
- **Filtering options**: Show all matches, only inliers, or only outliers
- **Customizable appearance**: Adjust colors, line width, keypoint size, and more
- **Export capability**: Save high-resolution visualizations to file

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy opencv-python matplotlib pycolmap
```

## Basic Usage

### View matches between two images by name:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg
```

### View matches between images by index:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 0 \
    --image2 1
```

## Advanced Options

### Limit number of displayed matches:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --max-matches 100
```

### Show only inliers (geometrically verified matches):

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --filter inliers
```

### Show only outliers (rejected by geometric verification):

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --filter outliers
```

### Save visualization to file:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --output visualizations/matches_001_002.png \
    --dpi 300
```

### Show all keypoints (including unmatched):

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --show-all-keypoints
```

### Customize colors and appearance:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --inlier-color blue \
    --outlier-color orange \
    --line-width 1.0 \
    --keypoint-size 5
```

## Command-Line Arguments

### Required Arguments

- `--database`: Path to COLMAP database file
- `--image-dir`: Directory containing the images
- `--image1`: First image (name or 0-based index)
- `--image2`: Second image (name or 0-based index)

### Optional Arguments

- `--output`: Output file path for saving visualization (default: display only)
- `--max-matches`: Maximum number of matches to display (default: all)
- `--filter`: Filter matches to display - `all`, `inliers`, or `outliers` (default: `all`)
- `--show-all-keypoints`: Show all keypoints, not just matched ones
- `--inlier-color`: Color for inlier matches (default: `green`)
- `--outlier-color`: Color for outlier matches (default: `red`)
- `--keypoint-size`: Size of keypoint markers in pixels (default: `3`)
- `--line-width`: Width of match lines (default: `0.5`)
- `--dpi`: DPI for saved figure (default: `150`)
- `--seed`: Random seed for sampling matches (default: `42`)

## Output Statistics

The visualization displays the following statistics:

1. **Keypoints**: Total number of keypoints detected in each image
2. **Total Matches**: Total number of matched keypoint pairs between the two images
3. **Inliers**: Number and percentage of geometrically verified matches
4. **Outliers**: Number of matches rejected by geometric verification

### Example Output:

```
Match Visualization: IMG_001.jpg <-> IMG_002.jpg
Keypoints: 2048 / 1987 | Total Matches: 342 | Inliers: 287 (83.9%) | Outliers: 55
```

## Understanding the Visualization

### Color Coding

- **Green lines**: Inlier matches (passed geometric verification, e.g., fundamental matrix or homography estimation)
- **Red lines**: Outlier matches (failed geometric verification, likely false matches)
- **Yellow dots**: Matched keypoint locations
- **Cyan dots** (if `--show-all-keypoints` used): All detected keypoints, including unmatched

### Layout

The visualization shows both images side-by-side with lines connecting corresponding keypoint pairs. The left image is the first image specified, and the right image is the second image.

## Examples

### Quick check of match quality:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 0 \
    --image2 1 \
    --max-matches 50
```

### Detailed analysis with all matches:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --show-all-keypoints \
    --output detailed_matches.png \
    --dpi 300
```

### Debug outlier matches:

```bash
python scripts/visualize_matches.py \
    --database outputs/database.db \
    --image-dir data/images/ \
    --image1 IMG_001.jpg \
    --image2 IMG_002.jpg \
    --filter outliers \
    --line-width 1.5
```

## Troubleshooting

### No matches found

If you see "No matches found", ensure:
1. Feature matching has been performed on the database
2. The two images have overlapping content
3. The image names/indices are correct

### No geometric verification data

If all matches appear as inliers with the message "No geometric verification data available":
- The database doesn't contain two-view geometry (geometric verification hasn't been run)
- Run `pycolmap.match_exhaustive()` or equivalent with geometric verification enabled

### Image not found

Ensure:
1. The `--image-dir` path is correct
2. Image names in the database match actual filenames
3. Images exist in the specified directory

## Integration with ViT-COLMAP Pipeline

This tool works with any COLMAP database created by the ViT-COLMAP pipeline:

```bash
# Run the pipeline
python scripts/run_pipeline.py --config configs/vit_config.yaml

# Visualize matches
python scripts/visualize_matches.py \
    --database outputs/vit_features/database.db \
    --image-dir data/images/ \
    --image1 0 \
    --image2 1
```

## Technical Details

### Data Sources

- **Keypoints**: Read from `keypoints` table using `pycolmap.Database.read_keypoints()`
- **Raw matches**: Read from `matches` table using `pycolmap.Database.read_matches()`
- **Inliers**: Read from `two_view_geometries` table using `pycolmap.Database.read_two_view_geometry()`

### Inlier/Outlier Classification

Matches are classified as inliers if they appear in the `inlier_matches` field of the two-view geometry. All other raw matches are classified as outliers. If no geometric verification data is available, all matches are treated as inliers.

## License

Part of the ViT-COLMAP project.
