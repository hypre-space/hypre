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

   **** This script requires certain external libraries. ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory
          

   This script runs a number of external library tests.

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
ro="-superlu"
eo=""

co="--enable-debug --with-blas-lib=\\'-L/home/falgout2/codes/blas/BLAS-3.7.1 -lblas -lgfortran\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-blas

co="--enable-debug --with-mli --with-superlu --with-superlu-include=/home/falgout2/codes/superlu/SuperLU_5.2.1/SRC"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-superlu

co="--enable-debug --with-mli --with-superlu --with-superlu-include=/home/falgout2/codes/superlu/SuperLU_5.2.1/SRC --with-dsuperlu --with-dsuperlu-include=/home/falgout2/codes/superlu/superlu_dist-6.3.1/SRC --with-blas-lib=\\'-L/home/falgout2/codes/blas/BLAS-3.7.1 -lblas -lgfortran\\' --with-dsuperlu-lib=\\'-L/home/falgout2/codes/superlu/superlu_dist-6.3.1/lib -lsuperlu_dist -L/home/falgout2/codes/parmetis/parmetis-4.0.3/build/Linux-x86_64/libparmetis -lparmetis -L/home/falgout2/codes/parmetis/parmetis-4.0.3/build/Linux-x86_64/libmetis -lmetis -lstdc++\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-dsuperlu

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
