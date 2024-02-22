./install-tool.sh m4
./install-tool.sh gmp
./install-tool.sh mpfr
./install-tool.sh mpc
./install-tool.sh cmake
./install-tool.sh gcc
./install-tool.sh llvm-clang 
./install-tool.sh python-packages
./install-tool.sh cm
./install-tool.sh git-lfs
./install-tool.sh re2c
./install-tool.sh ninja
./install-tool.sh oneapi-esimd
./install-tool.sh oneapi-support

# Add tools
TOOLS_PATH=$(pwd)/install
GCC_PATH=$TOOLS_PATH/gcc-7.5.0
export PATH=$TOOLS_PATH/bin:$GCC_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TOOLS_PATH/lib:$GCC_PATH/lib64:$LD_LIBRARY_PATH

# Checkout all files with git-lfs installed
git checkout -f HEAD

# Compile
cd Halide
make -j
