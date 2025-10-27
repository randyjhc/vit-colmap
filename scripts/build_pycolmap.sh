#!/bin/bash

# Script to build pycolmap from source with CUDA support
# This script automates the process of building pycolmap with GPU acceleration
# Prerequisites: COLMAP must already be built and installed in third_party/colmap/install

set -e  # Exit on error

echo "========================================"
echo "pycolmap CUDA Build Script"
echo "========================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Check if running from project root directory
if [ ! -f "vit_colmap/__init__.py" ] && [ ! -f "pyproject.toml" ]; then
    print_error "This script must be run from the project root directory (vit-colmap/)"
    echo "Current directory: $(pwd)"
    echo "Expected directory structure:"
    echo "  vit-colmap/"
    echo "  ├── vit_colmap/"
    echo "  ├── scripts/"
    echo "  ├── third_party/"
    echo "  └── pyproject.toml"
    echo ""
    echo "Please cd to the project root and run: ./scripts/build_pycolmap.sh"
    exit 1
fi

# Check if third_party/colmap exists
if [ ! -d "third_party/colmap" ]; then
    print_error "COLMAP submodule not found at third_party/colmap"
    echo "Please initialize the git submodule first:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

echo "Step 1: Checking prerequisites..."
echo "-----------------------------------"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA Toolkit not found (nvcc command not available)"
    echo "Please install CUDA Toolkit (version 11.0 or higher)"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
else
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    print_success "CUDA Toolkit found (version $CUDA_VERSION)"
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found"
    echo "Please install CMake (version 3.15 or higher)"
    echo "Ubuntu/Debian: sudo apt-get install cmake"
    exit 1
else
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_success "CMake found (version $CMAKE_VERSION)"
fi

# Check for Ninja
if ! command -v ninja &> /dev/null; then
    print_error "Ninja build system not found"
    echo "Please install Ninja:"
    echo "Ubuntu/Debian: sudo apt-get install ninja-build"
    exit 1
else
    print_success "Ninja found"
fi

# Check for build tools (gcc/g++)
if ! command -v gcc &> /dev/null || ! command -v g++ &> /dev/null; then
    print_error "Build tools (gcc/g++) not found"
    echo "Please install build essential tools:"
    echo "Ubuntu/Debian: sudo apt-get install build-essential"
    exit 1
else
    GCC_VERSION=$(gcc --version | head -n1 | cut -d' ' -f4)
    print_success "Build tools found (gcc version $GCC_VERSION)"
fi

# Check for uv (Python package manager)
if ! command -v uv &> /dev/null; then
    print_error "uv not found"
    echo "Please install uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
else
    print_success "uv found"
fi

echo ""
echo "Step 2: Checking for COLMAP installation..."
echo "--------------------------------------------"

# Check if COLMAP is already built
if [ ! -d "third_party/colmap/build" ]; then
    print_error "COLMAP build not found at third_party/colmap/install"
    echo "Please build COLMAP first before building pycolmap."
    echo "You can build COLMAP manually with:"
    echo "  cd third_party/colmap"
    echo "  mkdir -p build && cd build"
    echo "  cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON"
    echo "  ninja"
    echo "  cmake --install . --prefix ../install"
    exit 1
else
    print_success "COLMAP build found at third_party/colmap/install"
fi

echo ""
echo "Step 3: Building and installing pycolmap with CUDA..."
echo "------------------------------------------------------"

cd third_party/colmap

# Set CMAKE_PREFIX_PATH to help find the locally built COLMAP
export CMAKE_PREFIX_PATH=$(pwd)/install

print_info "Installing pycolmap with uv pip..."
uv pip install .

cd ../..  # Return to project root
print_success "pycolmap installed successfully"

echo ""
echo "Step 4: Verifying CUDA support..."
echo "----------------------------------"

# Verify CUDA support
print_info "Checking if pycolmap has CUDA support..."
CUDA_ENABLED=$(python -c "import pycolmap; print(pycolmap.has_cuda)" 2>&1)

if [ "$CUDA_ENABLED" == "True" ]; then
    print_success "CUDA support is enabled!"

    # Check number of GPUs
    NUM_GPUS=$(python -c "import pycolmap; print(pycolmap.get_num_cuda_devices())" 2>&1)
    print_info "Number of CUDA devices available: $NUM_GPUS"
else
    print_error "CUDA support is NOT enabled"
    echo "pycolmap was built but CUDA support was not enabled."
    echo "This might be due to:"
    echo "  - CUDA libraries not found during build"
    echo "  - Incompatible CUDA version"
    echo "  - Missing CUDA development headers"
    exit 1
fi

# Show pycolmap version
PYCOLMAP_VERSION=$(python -c "import pycolmap; print(pycolmap.__version__)" 2>&1)
print_info "pycolmap version: $PYCOLMAP_VERSION"

echo ""
echo "========================================"
print_success "Build completed successfully!"
echo "========================================"
echo ""
echo "Summary:"
echo "  - pycolmap installed with CUDA: Yes"
echo "  - CUDA devices available: $NUM_GPUS"
echo "  - pycolmap version: $PYCOLMAP_VERSION"
echo ""
echo "You can now use GPU-accelerated COLMAP features for:"
echo "  - SIFT feature extraction"
echo "  - Feature matching"
echo "  - Dense reconstruction (MVS)"
echo ""
