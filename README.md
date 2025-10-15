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

### Basic Pipeline Example

```python
from pathlib import Path
from vit_colmap.pipeline import Pipeline
from vit_colmap.features.dummy_extractor import DummyExtractor
from vit_colmap.utils import Config

# Configure the pipeline
config = Config()
config.camera.model = "PINHOLE"
config.do_matching = True
config.do_reconstruction = True

# Initialize pipeline and extractor
pipeline = Pipeline(config=config)
extractor = DummyExtractor()

# Run the pipeline
result = pipeline.run(
    image_dir=Path("path/to/images"),
    output_dir=Path("path/to/output"),
    db_path=Path("path/to/database.db"),
    extractor=extractor
)
```

## Configuration

The pipeline supports various configuration options through the `Config` class:

- **Camera**: Model type (SIMPLE_PINHOLE, PINHOLE), parameters
- **Matching**: GPU usage, matching thresholds, number of threads
- **Reconstruction**: Minimum matches, multiple models
- **Logging**: Log level, format

See [vit_colmap/utils/config.py](vit_colmap/utils/config.py) for all available options.
