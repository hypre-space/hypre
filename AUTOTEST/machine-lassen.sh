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

   **** Only run this script on the lassen cluster ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory

   This script runs a number of tests suitable for the lassen cluster.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=`cd $1; pwd`
shift

# Basic build and run tests
mo="-j test"
eo=""

rtol="0.0"
atol="3e-15"

#save=`echo $(hostname) | sed 's/[0-9]\+$//'`
save="lassen"

######################
##   DEFAULT CUDA   ##
##  (cuda/10.1.243) ##
######################

module -q load cuda
module -q load xl

# CUDA with UM in debug mode [ij, ams, struct, sstruct]
co="--with-cuda --enable-unified-memory --enable-persistent --enable-debug --with-gpu-arch=70 --with-memory-tracker --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-ij-gpu -ams -struct -sstruct -rt -mpibind -save ${save} -rtol ${rtol} -atol ${atol}"
eo="-gpu -rt -mpibind -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -eo: $eo
./renametest.sh basic $output_dir/basic-cuda-um

# CUDA with UM in debug mode [ij, ams, struct, sstruct]
co="--with-cuda --enable-unified-memory --enable-persistent --enable-debug --with-print-errors --with-gpu-arch=70 --with-memory-tracker --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-error -rt -mpibind -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-um-with-errors

# CUDA with UM and mixed-int
co="--with-cuda --enable-unified-memory --enable-mixedint --enable-debug --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-ij-mixed -ams -struct -sstruct-mixed -rt -mpibind -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-um-mixedint

# CUDA with UM with shared library
co="--with-cuda --enable-unified-memory --with-openmp --enable-hopscotch --enable-shared --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-gpumemcheck -rt -mpibind -cudamemcheck -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-um-shared

# CUDA with UM and single precision
co="--with-cuda --enable-unified-memory --enable-single --enable-debug --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-single -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: ${ro}
./renametest.sh basic $output_dir/basic-cuda-um-single

# CUDA with UM without MPI [no run]
#co="--with-cuda --enable-unified-memory --without-MPI --with-gpu-arch=70 --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo
#./renametest.sh basic $output_dir/basic-cuda-um-without-MPI

# CUDA without UM with device memory pool [struct]
co="--with-cuda --enable-device-memory-pool --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-struct -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-nonum

# CUDA without UM with umpire [benchmark]
UMPIRE_DIR=/usr/workspace/hypre/ext-libs/Umpire/2022.03.1-nvcc10.1.243-sm_70-xl2021.09.22
co="--with-cuda --with-gpu-arch=70 --with-umpire --with-umpire-include=${UMPIRE_DIR}/include --with-umpire-lib-dirs=${UMPIRE_DIR}/lib --with-umpire-libs=umpire --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-bench -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-bench

# run on CPU
co="--with-cuda --with-test-using-host --with-memory-tracker --enable-debug --with-gpu-arch=70"
ro="-ij-noilu -ams -struct -sstruct -rt -mpibind -save lassen_cpu"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-cpu

############
## OMP4.5 ##
############

# OMP 4.5 with UM with shared library [no run]
#co="--with-device-openmp --enable-unified-memory --enable-shared --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029:1500-030:1501-308\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029:1500-030:1501-308\\'"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo
#./renametest.sh basic $output_dir/basic-deviceomp-um-shared

# OMP 4.5 without UM in debug mode [struct]
co="--with-device-openmp --enable-debug --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-struct -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-deviceomp-nonum-debug-struct

#####################################
## CUDA + CMake build (only) tests ##
#####################################

mo="-j"

# CUDA with UM + CMake
co="-DCMAKE_C_COMPILER=$(which xlc) -DCMAKE_CXX_COMPILER=$(which xlc++) -DCMAKE_CUDA_COMPILER=$(which nvcc) -DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx) -DHYPRE_WITH_CUDA=ON -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DCMAKE_BUILD_TYPE=Debug -DHYPRE_ENABLE_PERSISTENT_COMM=ON -DHYPRE_ENABLE_DEVICE_POOL=ON -DHYPRE_WITH_EXTRA_CFLAGS="\'"-qmaxmem=-1 -qsuppress=1500-029"\'" -DHYPRE_WITH_EXTRA_CXXFLAGS="\'"-qmaxmem=-1 -qsuppress=1500-029"\'" -DHYPRE_CUDA_SM=70"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo
./renametest.sh cmake $output_dir/cmake-cuda-um-ij

# CUDA with UM [shared library] + CMake
co="-DCMAKE_C_COMPILER=$(which xlc) -DCMAKE_CXX_COMPILER=$(which xlc++) -DCMAKE_CUDA_COMPILER=$(which nvcc) -DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx) -DHYPRE_WITH_CUDA=ON -DHYPRE_ENABLE_UNIFIED_MEMORY=ON -DCMAKE_BUILD_TYPE=Debug -DHYPRE_WITH_OPENMP=ON -DHYPRE_ENABLE_HOPSCOTCH=ON -DHYPRE_ENABLE_SHARED=ON -DHYPRE_WITH_EXTRA_CFLAGS="\'"-qmaxmem=-1 -qsuppress=1500-029"\'" -DHYPRE_WITH_EXTRA_CXXFLAGS="\'"-qmaxmem=-1 -qsuppress=1500-029 "\'" -DHYPRE_CUDA_SM=70"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo
./renametest.sh cmake $output_dir/cmake-cuda-um-shared

# CUDA w.o UM + CMake
co="-DCMAKE_C_COMPILER=$(which xlc) -DCMAKE_CXX_COMPILER=$(which xlc++) -DCMAKE_CUDA_COMPILER=$(which nvcc) -DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx) -DHYPRE_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Debug -DHYPRE_WITH_EXTRA_CFLAGS="\'"-qmaxmem=-1 -qsuppress=1500-029"\'" -DHYPRE_WITH_EXTRA_CXXFLAGS="\'"-qmaxmem=-1 -qsuppress=1500-029"\'" -DHYPRE_CUDA_SM=70"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo
./renametest.sh cmake $output_dir/cmake-cuda-nonum-struct

#######################
## latest CUDA tests ##
##    cuda/12.0.0    ##
#######################

rtol="1e-2"
atol="1e-6"
mo="-j test"

# CUDA 12.0 + GCC with UM in debug mode, thrust nosync [ij, ams, struct, sstruct]
module -q load gcc/8.3.1
module -q load cuda/12.0
module list cuda/12.0 |& grep "None found"
co="--with-cuda --enable-unified-memory --enable-thrust-nosync --enable-debug --with-gpu-arch=70 CC=mpicc CXX=mpicxx"
ro="-ij-gpu -ams -struct -sstruct -rt -mpibind -save ${save} -rtol ${rtol} -atol ${atol}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda12_0

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

