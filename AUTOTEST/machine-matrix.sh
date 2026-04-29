#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

testname=`basename $0 .sh`

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   **** Only run this script on LC's matrix cluster ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory

   This script runs a number of tests suitable for the matrix cluster.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
cmake_version=3.30
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=`cd $1; pwd`
root_dir=`cd $src_dir/..; pwd`
shift

# Basic build and run tests
aco="LIBS=-ldl"
cco="-DMPIEXEC_EXECUTABLE=\"srun\" -DMPIEXEC_NUMPROC_FLAG=\"n\""
cmo="--parallel"
mo="-j test"
eo=""
rtol="0.0"
atol="3e-15"
save="matrix"

# 1C) GCC 13.3.1 + CUDA 12.9.1 with UM and memory tracker OFF in debug mode [error, ij, ams, struct, sstruct]
module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
co="${cco} -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_MEMORY_TRACKER=OFF -DHYPRE_ENABLE_PERSISTENT_COMM=ON -DHYPRE_ENABLE_PRINT_ERRORS=ON"
ro="-error -ij -ams -struct -sstruct -rt -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo -ro: $ro
./renametest.sh cmake $output_dir/cmake-cuda-um-dbg

# 2C) GCC 13.3.1 + CUDA 12.9.1 with mixed integers and UM in debug mode [ij-mixed, ams, struct, sstruct-mixed]
module reset && module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
co="${cco} -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_MIXEDINT=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_PRINT_ERRORS=ON"
ro="-error -ij-mixed -ams -struct -sstruct-mixed -rt -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo -ro: $ro
./renametest.sh cmake $output_dir/cmake-cuda-um-mixedint

# 3C) GCC 13.3.1 + CUDA 12.9.1 with OMP and shared library in release mode
module reset && module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
co="${cco} -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_OPENMP=ON -DHYPRE_ENABLE_HOPSCOTCH=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_PRINT_ERRORS=ON"
ro="-gpumemcheck -rt -cudasan -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo -ro: $ro
./renametest.sh cmake $output_dir/cmake-cuda-um-shared

# 4C) GCC 13.3.1 + CUDA 12.9.1 with UM and single precision in debug mode
module reset && module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
co="${cco} -DHYPRE_ENABLE_SINGLE=ON -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_CUSOLVER=ON -DHYPRE_ENABLE_OPENMP=ON -DHYPRE_ENABLE_HOPSCOTCH=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_PRINT_ERRORS=ON"
ro="-single -rt -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo -ro: $ro
./renametest.sh cmake $output_dir/cmake-cuda-um-single

# 5C) GCC 13.3.1 + CUDA 12.9.1 without MPI [no run]
module reset && module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
co="${cco} -DHYPRE_ENABLE_MPI=OFF -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_PRINT_ERRORS=ON"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo
./renametest.sh cmake $output_dir/cmake-cuda-um-without-MPI

# 6C) GCC 13.3.1 + CUDA 12.9.1 with Umpire [benchmark]
module reset && module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
UMPIRE_DIR=/usr/workspace/hypre/ext-libs/Umpire/install-umpire_2025.09.0-cuda_12.9_sm90-gcc_13.3
co="${cco} -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_PRINT_ERRORS=ON -Dumpire_DIR=${UMPIRE_DIR}/lib/cmake"
ro="-bench -rt -mpibind -save ${save}"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo -ro: $ro
./renametest.sh cmake $output_dir/cmake-cuda-bench

# 7C) GCC 13.3.1 + CUDA 12.9.1 with host execution
module reset && module -q load cmake/${cmake_version} cuda/12.9.1 gcc/13.3.1
co="${cco} -DHYPRE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DHYPRE_ENABLE_UMPIRE=OFF -DHYPRE_ENABLE_TEST_USING_HOST=ON -DHYPRE_ENABLE_MEMORY_TRACKER=OFF -DHYPRE_ENABLE_PRINT_ERRORS=ON -DCMAKE_BUILD_TYPE=Debug"
ro="-ij-noilu -ams -struct -sstruct"
./test.sh cmake.sh $root_dir -co: $co -mo: $cmo -ro: $ro
./renametest.sh cmake $output_dir/cmake-cuda-cpu

##################################
## Autotools (build only) tests ##
##################################

# 1A) GCC 13.3.1 + CUDA 12.9.1 with UM and memory tracker in debug mode
module -q load cuda/12.9.1 gcc/13.3.1
co="${aco} --with-cuda --without-umpire --enable-unified-memory --enable-debug --with-gpu-arch=90 --with-memory-tracker --enable-persistent --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-dbg

# 2A) GCC 13.3.1 + CUDA 12.9.1 with mixed integers and UM in debug mode
module reset && module -q load cuda/12.9.1 gcc/13.3.1
co="${aco} --with-cuda --without-umpire --enable-unified-memory --enable-debug --with-gpu-arch=90 --enable-mixedint --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-mixedint

# 3A) GCC 13.3.1 + CUDA 12.9.1 with OMP and shared library in release mode
module reset && module -q load cuda/12.9.1 gcc/13.3.1
co="${aco} --with-cuda --without-umpire --enable-unified-memory --with-openmp --enable-hopscotch --enable-shared --with-gpu-arch=90 --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-shared

# 4A) GCC 13.3.1 + CUDA 12.9.1 with UM and single precision in debug mode
module reset && module -q load cuda/12.9.1 gcc/13.3.1
co="${aco} --enable-single --without-umpire --enable-unified-memory --with-cuda --enable-cusolver --with-openmp --enable-hopscotch --enable-shared --with-gpu-arch=90 --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-single

# 5A) GCC 13.3.1 + CUDA 12.9.1 without MPI
module reset && module -q load cuda/12.9.1 gcc/13.3.1
co="${aco} --without-MPI --without-umpire --enable-unified-memory --with-cuda --with-gpu-arch=90 --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-without-MPI

# 6A) GCC 13.3.1 + CUDA 12.9.1 with Umpire
module reset && module -q load cuda/12.9.1 gcc/13.3.1
UMPIRE_DIR=/usr/workspace/hypre/ext-libs/Umpire/install-umpire_2025.09.0-cuda_12.9_sm90-gcc_13.3
co="${aco} --with-cuda --with-gpu-arch=90 --with-umpire --with-umpire-include=${UMPIRE_DIR}/include --with-umpire-lib-dirs=${UMPIRE_DIR}/lib --with-umpire-libs=\\"umpire camp\\" --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-bench

# 7A) GCC 13.3.1 + CUDA 12.9.1 with host execution
module reset && module -q load cuda/12.9.1 gcc/13.3.1
co="${aco} --with-cuda --with-gpu-arch=90 --without-umpire --with-test-using-host --with-memory-tracker --enable-debug --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-cpu

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
