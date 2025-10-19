# vit-colmap

A Vision Transformer (ViT) based feature extraction pipeline integrated with COLMAP for 3D reconstruction.

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

### Environment Setup with uv

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/vit-colmap.git
   cd vit-colmap
   ```

3. **Create and activate a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

5. **Install development dependencies** (for testing and pre-commit hooks):
   ```bash
   uv sync --group dev
   ```

## Development Workflow

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency.

1. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

2. **Run pre-commit manually** (before committing):
   ```bash
   pre-commit run --all-files
   ```

The pre-commit hooks include:
- **Ruff**: Fast Python linter and formatter
- **MyPy**: Static type checking
- **General checks**: Trailing whitespace, end-of-file fixes, YAML/TOML validation

After installation, these hooks run automatically on every `git commit`.

### Running Tests

#### Sanity Check (Smoke Test)

Run the end-to-end integration test to verify the pipeline works correctly:

```bash
pytest tests/test_smoke_e2e.py -v
```

This test:
- Creates synthetic checkerboard images
- Extracts features using DummyExtractor
- Builds a COLMAP database
- Performs feature matching
- Verifies database integrity

#### Run All Tests

```bash
pytest
```

## Usage

### Command-Line Interface

Run the pipeline from the command line:

```bash
# Using COLMAP's built-in SIFT (recommended)
python -m vit_colmap.pipeline.run_pipeline \
    --images data/raw/my-dataset/images \
    --output data/outputs/my-dataset \
    --db data/intermediate/my-dataset/database.db \
    --camera-model PINHOLE \
    --use-colmap-sift \
    --verbose

# Using ViT extractor (when implemented)
python -m vit_colmap.pipeline.run_pipeline \
    --images data/raw/my-dataset/images \
    --output data/outputs/my-dataset \
    --db data/intermediate/my-dataset/database.db \
    --camera-model PINHOLE \
    --model path/to/vit-weights.pth
```

### Python API Example

```python
from pathlib import Path
from vit_colmap.pipeline import Pipeline
from vit_colmap.utils.config import Config

# Configure the pipeline
config = Config()
config.camera.model = "PINHOLE"
config.extractor.extractor_type = "colmap_sift"  # or "vit" or "dummy"
config.do_matching = True
config.do_reconstruction = True

# Initialize and run pipeline
pipeline = Pipeline(config=config)
result = pipeline.run(
    image_dir=Path("data/raw/my-dataset/images"),
    output_dir=Path("data/outputs/my-dataset"),
    db_path=Path("data/intermediate/my-dataset/database.db"),
)
```

### DTU Dataset Example

A convenience script is provided for running on DTU datasets:

```bash
# Run on DTU scan1
./scripts/run_DTU_colmap.sh scan1

# Run on DTU scan21
./scripts/run_DTU_colmap.sh scan21
```

See [data/README.md](data/README.md) for more information on data organization.

## Configuration

The pipeline supports various configuration options through the `Config` class:

- **Camera**: Model type (SIMPLE_PINHOLE, PINHOLE), parameters
- **Extractor**: Type (colmap_sift, vit, dummy), model weights path
- **Matching**: GPU usage, matching thresholds, number of threads
- **Reconstruction**: Minimum matches, multiple models
- **Logging**: Log level, format

See [vit_colmap/utils/config.py](vit_colmap/utils/config.py) for all available options.

## Project Structure

```
vit-colmap/
├── data/                          # Data directory (see data/README.md)
│   ├── raw/                      # Raw input datasets
│   ├── intermediate/             # Intermediate files (databases, temp images)
│   └── outputs/                  # Reconstruction outputs
├── vit_colmap/                   # Main package
│   ├── features/                 # Feature extractors
│   │   ├── base_extractor.py    # Base extractor interface
│   │   ├── vit_extractor.py     # ViT-based extractor (to be implemented)
│   │   ├── colmap_sift_extractor.py  # COLMAP SIFT wrapper
│   │   └── dummy_extractor.py   # Testing/dummy extractor
│   ├── pipeline/                 # Pipeline orchestration
│   ├── database/                 # COLMAP database utilities
│   └── utils/                    # Configuration and utilities
├── scripts/                      # Utility scripts
│   └── run_DTU_colmap.sh        # DTU dataset runner
└── tests/                        # Tests
    └── test_smoke_e2e.py        # End-to-end integration test
```
