#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

   This script runs a number of tests suitable for the syrah cluster.

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

##########
## CUDA ##
##########

# CUDA without UM with Umpire [benchmark]
# build Umpire
# cmake3 -DCMAKE_INSTALL_PREFIX=../install_xlC_lassen -DENABLE_CUDA=On -DENABLE_OPENMP=Off -DCMAKE_CXX_COMPILER=xlC -DCMAKE_C_COMPILER=xlc -DENABLE_CUDA=On -DCMAKE_CUDA_FLAGS="-arch sm_70" -DENABLE_C=On ../
# make
# make install

UMPIRE_DIR=/usr/workspace/li50/Umpire-git/Umpire/install_xlC_lassen
co="--with-cuda --enable-debug --with-umpire --with-umpire-include=${UMPIRE_DIR}/include --with-umpire-lib-dirs=${UMPIRE_DIR}/lib --with-umpire-libs=umpire --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-bench -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo #-ro: $ro
./renametest.sh basic $output_dir/basic-cuda-nonum-umpire

# CUDA without UM with RAJA [struct]
# build RAJA
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(which xlC) -DENABLE_OPENMP=Off -DENABLE_CUDA=On -DCUDA_ARCH=sm_70 -DCMAKE_INSTALL_PREFIX=../install_lassen ../

RAJA_DIR=/usr/workspace/li50/RAJA-git/raja/install_lassen
co="--with-cuda --enable-debug --with-raja --with-raja-include=${RAJA_DIR}/include --with-raja-lib-dirs=${RAJA_DIR}/lib --with-raja-libs=RAJA --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-struct -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-nonum-raja

# CUDA without UM with Kokkos [struct]
# build Kokkos use [2]
# [1] cmake -D CMAKE_CXX_COMPILER=${PWD}/../bin/nvcc_wrapper -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${PWD}/../install_lassen_cuda -D Kokkos_ARCH_POWER9=ON -D Kokkos_ARCH_VOLTA70=ON -D Kokkos_ENABLE_DEBUG=OFF -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_CUDA_DIR=${CUDA_HOME} -D Kokkos_ENABLE_CUDA_LAMBDA=ON -D Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON ..
# [2] ../generate_makefile.bash --with-cuda=${CUDA_HOME} --prefix=${PWD}/../install_lassen_cuda --arch=Volta70 --compiler=${PWD}/../bin/nvcc_wrapper --with-cuda-options=enable_lambda,rdc

module load gcc/8.3.1
module load cmake/3.16
KOKKOS_DIR=/usr/workspace/li50/kokkos-git/kokkos/install_lassen_cuda
co="--with-cuda --enable-debug --with-kokkos --with-kokkos-include=${KOKKOS_DIR}/include --with-kokkos-lib-dirs=${KOKKOS_DIR}/lib64 --with-kokkos-libs=kokkoscore --with-cxxstandard=14 --with-gpu-arch=70 CC=mpicc CXX=mpicxx"
ro="-struct -rt -mpibind -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-nonum-kokkos

############
## OMP4.5 ##
############

