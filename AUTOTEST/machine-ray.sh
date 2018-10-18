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
eo=""
roij="-ij -rt -mpibind -rtol 1e-3 -atol 1e-3"
ross="-struct -sstruct -rt -mpibind -rtol 1e-6 -atol 1e-6"
rost="-struct -rt -mpibind -rtol 1e-8 -atol 1e-8"

# CUDA with UM
co="--with-cuda --enable-unified-memory --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $roij
./renametest.sh basic $output_dir/basic-cuda-um-ij
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ross
./renametest.sh basic $output_dir/basic-cuda-um-struct-sstruct

# CUDA with UM [shared library, no run]
co="--with-cuda --enable-unified-memory --enable-shared --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-cuda-um-shared
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $roij
#./renametest.sh basic $output_dir/basic-cuda-um-shared-ij
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ross
#./renametest.sh basic $output_dir/basic-cuda-um-shared-struct-sstruct

# OMP 4.5 with UM
co="--with-device-openmp --enable-unified-memory --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $roij
./renametest.sh basic $output_dir/basic-deviceomp-um-ij
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ross
./renametest.sh basic $output_dir/basic-deviceomp-um-struct-sstruct

# OMP 4.5 with UM [shared library, no run]
co="--with-device-openmp --enable-unified-memory --enable-shared --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029:1500-030:1501-308\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029:1500-030:1501-308\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-deviceomp-um-shared
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $roij
#./renametest.sh basic $output_dir/basic-deviceomp-um-shared-ij
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ross
#./renametest.sh basic $output_dir/basic-deviceomp-um-shared-struct-sstruct

# OMP 4.5 with UM [in debug mode]
#co="--with-device-openmp --enable-unified-memory --enable-debug --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $roij
#./renametest.sh basic $output_dir/basic-deviceomp-um-debug-ij
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ross
#./renametest.sh basic $output_dir/basic-deviceomp-um-debug-struct-sstruct

# CUDA w.o UM, only struct
#co="--with-cuda --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
#./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $rost
#./renametest.sh basic $output_dir/basic-cuda-nonum-struct

# OMP4.5 w.o UM, only struct [in debug mode]
co="--with-device-openmp --enable-debug --with-extra-CFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\' --with-extra-CXXFLAGS=\\'-qmaxmem=-1 -qsuppress=1500-029\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $rost
./renametest.sh basic $output_dir/basic-deviceomp-nonum-debug-struct

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done


