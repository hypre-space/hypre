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

   **** Only run this script on one of the tux machines. ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory


   This script runs a number of tests suitable for the tux machines.

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
root_dir=`cd $src_dir/..; pwd`
shift

# Basic build and run tests
mo="-j test"
ro="-ams -ij -sstruct -sstructmat -struct -structmat -lobpcg"
ronolob="-ams -ij -sstruct -sstructmat -struct -structmat"
eo=""
# From tux master: ro="-ams -ij -sstruct -struct -lobpcg"

co="--enable-debug --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ronolob -error -rt -valgrind
./renametest.sh basic $output_dir/basic--valgrind

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
