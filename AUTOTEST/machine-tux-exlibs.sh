#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

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

co="--enable-debug --with-mli --with-superlu --with-superlu-include=/home/falgout2/codes/superlu/SuperLU_5.2.1/SRC --with-dsuperlu --with-dsuperlu-include=/home/falgout2/codes/superlu/SuperLU_DIST_5.2.1/SRC --with-blas-lib=\\'-L/home/falgout2/codes/blas/BLAS-3.7.1 -lblas -lgfortran\\' --with-dsuperlu-lib=\\'-L/home/falgout2/codes/superlu/SuperLU_DIST_5.2.1/lib -lsuperlu_dist -L/home/falgout2/codes/parmetis/parmetis-4.0.3/build/Linux-x86_64/libparmetis -lparmetis -L/home/falgout2/codes/parmetis/parmetis-4.0.3/build/Linux-x86_64/libmetis -lmetis\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-dsuperlu

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
