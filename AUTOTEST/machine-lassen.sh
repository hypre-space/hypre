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

   **** Only run this script on the lassen/ray cluster ****

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

##########
## CUDA ##
##########

# CUDA with UM in debug mode [ij, ams, struct, sstruct]
co="--with-cuda --enable-unified-memory --enable-persistent --enable-debug --with-cuda-arch=\\'60 70\\' --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-ij-gpu -ams -struct -sstruct -rt -mpibind -save cuda"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-um

# CUDA with UM with shared library [no run]
co="--with-cuda --enable-unified-memory --with-openmp --enable-hopscotch --enable-shared --with-cuda-arch=\\'60 70\\' --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-shared

# CUDA with UM without MPI [no run]
#co="--with-cuda --enable-unified-memory --without-MPI --with-cuda-arch=\\'60 70\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo
#./renametest.sh basic $output_dir/basic-cuda-um-without-MPI

# CUDA without UM with device memory pool [benchmark, struct]
co="--with-cuda --enable-device-memory-pool --with-cuda-arch=\\'60 70\\' --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-bench -struct -rt -mpibind -save cuda"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda-nonum

############
## OMP4.5 ##
############

# OMP 4.5 with UM with shared library [no run]
#co="--with-device-openmp --enable-unified-memory --enable-shared --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029:1500-030:1501-308\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029:1500-030:1501-308\\'"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo
#./renametest.sh basic $output_dir/basic-deviceomp-um-shared

# OMP 4.5 without UM in debug mode [struct]
co="--with-device-openmp --enable-debug --with-cuda-arch=\\'60 70\\' --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
ro="-struct -rt -mpibind -save cuda"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-deviceomp-nonum-debug-struct

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

