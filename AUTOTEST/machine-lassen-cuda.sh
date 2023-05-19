#!/bin/sh
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

############################################
## Various CUDA verion build (only) tests ##
############################################

# CUDA 9.0 with UM [no run]
module -q load cuda/9.0
module list cuda/9.0 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda9_0

# CUDA 9.1 with UM [no run]
module -q load cuda/9.1
module list cuda/9.1 |& grep "None found"
module -q load gcc
co="--with-cuda --enable-unified-memory --with-gpu-arch=70 --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\' CC=mpicc CXX=mpicxx"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda9_1

# CUDA 9.2 with UM [no run]
module -q load cuda/9.2
module list cuda/9.2 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda9_2

# CUDA 10.2 with UM [no run]
module -q load cuda/10.2
module list cuda/10.2 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda10_2

# CUDA 11.0 with UM [no run]
module -q load cuda/11.0
module list cuda/11.0 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_0

# CUDA 11.1 with UM [no run]
module -q load cuda/11.1
module list cuda/11.1 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_1

# CUDA 11.2 with UM with async malloc [no run]
module -q load cuda/11.2
module list cuda/11.2 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_2

# CUDA 11.3 with UM with async malloc [no run]
module -q load cuda/11.3
module list cuda/11.3 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_3

# CUDA 11.4 with UM with async malloc [no run]
module -q load cuda/11.4
module list cuda/11.4 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_4

# CUDA 11.5 with UM with async malloc [no run]
module -q load cuda/11.5
module list cuda/11.5 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_5

# CUDA 11.6 with UM with async malloc [no run]
module -q load cuda/11.6
module list cuda/11.6 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_6

# CUDA 11.7 with UM with async malloc [no run]
module -q load cuda/11.7
module list cuda/11.7 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_7

# CUDA 11.8 with UM with async malloc [no run]
module -q load cuda/11.8
module list cuda/11.8 |& grep "None found"
module -q load xl
co="--with-cuda --enable-unified-memory --enable-device-malloc-async --with-gpu-arch=70 --with-extra-CFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qsuppress=1500-029\\' --with-extra-CUFLAGS=\\'--Wno-deprecated-declarations\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda11_8

