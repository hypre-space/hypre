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

   **** Only run this script on a Mac ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory

   This script runs a number of tests suitable for a Mac.

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

# This is needed for some reason
export CXX="mpicxx"

# Basic build and run tests
# Make sure that we don't check for a working Fortran compiler
mo="test"
ro="-ams -ij -sstruct -struct -rt -D HYPRE_NO_SAVED"

co="--disable-fortran"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-default

co="--enable-debug --disable-fortran"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic--enable-debug

co="--enable-bigint --disable-fortran"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic--enable-bigint

co="--enable-mixedint --disable-fortran"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic--enable-mixedint

# Test linking for different languages
link_opts="all++"
for opt in $link_opts
do
   output_subdir=$output_dir/link$opt
   mkdir -p $output_subdir
   ./test.sh link.sh $src_dir $opt
   mv -f link.??? $output_subdir
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
