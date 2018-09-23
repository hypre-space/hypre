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

   **** Only run this script on the ray cluster ****

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
#ro="-ams -ij -sstruct -struct -rt -mpibind"
eo=""

# CUDA
#co="--with-cuda --enable-unified-memory"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
#./renametest.sh basic $output_dir/basic-cuda-um

#co="--with-cuda --enable-unified-memory --enable-shared"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
#./renametest.sh basic $output_dir/basic-cuda-um-shared

# OMP 4.5
#co="--with-device-openmp --enable-unified-memory"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
#./renametest.sh basic $output_dir/basic-deviceomp-um

#co="--with-device-openmp --enable-unified-memory --enable-shared"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
#./renametest.sh basic $output_dir/basic-deviceomp-um-shared

#co="--with-device-openmp --enable-unified-memory -enable-debug"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
#./renametest.sh basic $output_dir/basic-deviceomp-um-debug

# Without UM only struct
ro="-struct -rt -mpibind"
co="--with-cuda --with-extra-CXXFLAGS=\"-qmaxmem=-1 -qsuppress=1500-029\""
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-cuda

#co="--with-device-openmp"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
#./renametest.sh basic $output_dir/basic-deviceomp

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
