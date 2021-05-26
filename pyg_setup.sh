
# Works on OS X, with conda installed.

# Create conda environment for PyTorch Geometric
echo "Creating pyg environment"
conda create -n pyg python=3.6

echo "Activate pyg Env"
source activate pyg

# PyTorch Conda Installation
echo "Installing PyTorch"
conda install pytorch torchvision -c pytorch

# Change of Compilers
echo "Compiler Changing on OS X"
conda install -y clang_osx-64 clangxx_osx-64 gfortran_osx-64
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++

# Install dependencies
echo "Installing PyG Dependencies"
pip install torch_scatter
pip install torch_sparse
pip install torch_cluster
pip install torch_geometric
