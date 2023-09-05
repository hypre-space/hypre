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
shift

# Organizing the tests from "fast" to "slow"

# Check license header info
#( cd $src_dir; make distclean )
./test.sh check-license.sh $src_dir/..
mv -f check-license.??? $output_dir

# Check usage of int, double, MPI, memory, headers
./test.sh check-int.sh $src_dir
mv -f check-int.??? $output_dir
./test.sh check-double.sh $src_dir
mv -f check-double.??? $output_dir
./test.sh check-mpi.sh $src_dir
mv -f check-mpi.??? $output_dir
./test.sh check-mem.sh $src_dir
mv -f check-mem.??? $output_dir
./test.sh check-headers.sh $src_dir
mv -f check-headers.??? $output_dir

# Check for case-insensitive filename matches
./test.sh check-case.sh $src_dir/..
mv -f check-case.??? $output_dir

# Basic build and run tests
mo="-j test"
ro="-ams -ij -sstruct -struct -lobpcg"
eo=""

co=""
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-default

# Test linking for different languages
co=""
./test.sh configure.sh $src_dir $co
./test.sh make.sh $src_dir $mo
link_opts="all++ all77"
for opt in $link_opts
do
   output_subdir=$output_dir/link$opt
   mkdir -p $output_subdir
   cp -r configure.??? make.??? $output_subdir
   ./test.sh link.sh $src_dir $opt
   mv -f link.??? $output_subdir
done
rm -rf configure.??? make.???
( cd $src_dir; make distclean )

co="--without-MPI"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic--without-MPI

co="--with-strict-checking"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic--with-strict-checking

co="--enable-shared"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic--enable-shared

co="--enable-debug --with-extra-CFLAGS=\\'-Wstrict-prototypes\\'"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -eo: $eo
./renametest.sh basic $output_dir/basic-debug1

co="--enable-maxdim=4 --enable-debug"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -eo: -maxdim
./renametest.sh basic $output_dir/basic--enable-maxdim=4

co="--enable-complex --enable-maxdim=4 --enable-debug"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -eo: -complex
# ignore complex compiler output for now
rm -fr basic.dir/make.???
grep -v make.err basic.err > basic.tmp
mv basic.tmp basic.err
./renametest.sh basic $output_dir/basic--enable-complex

co="--with-openmp"
RO="-ams -ij -sstruct -struct -lobpcg -rt -D HYPRE_NO_SAVED -nthreads 2"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $RO
./renametest.sh basic $output_dir/basic--with-openmp

co="--with-openmp --enable-hopscotch"
RO="-ij -sstruct -struct -lobpcg -rt -D HYPRE_NO_SAVED -nthreads 2"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $RO
./renametest.sh basic $output_dir/basic--with-concurrent-hopscotch

co="--enable-single --enable-debug"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: -single
./renametest.sh basic $output_dir/basic--enable-single

co="--enable-longdouble --enable-debug"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: -longdouble
./renametest.sh basic $output_dir/basic--enable-longdouble

co="--enable-debug CC=mpiCC"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -eo: $eo
./renametest.sh basic $output_dir/basic-debug-cpp

co="--enable-bigint --enable-debug"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -eo: -bigint
./renametest.sh basic $output_dir/basic--enable-bigint

co="--enable-mixedint --enable-debug"
RO="-ams -ij-mixed -sstruct-mixed -struct -lobpcg-mixed"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $RO
./renametest.sh basic $output_dir/basic--enable-mixedint

co="--enable-debug --with-print-errors"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -error -rt -valgrind
./renametest.sh basic $output_dir/basic--valgrind

# CMake build and run tests
mo="-j"
ro="-ams -ij -sstruct -struct -lobpcg"
eo=""

co=""
./test.sh cmake.sh $src_dir -co: $co -mo: $mo
./renametest.sh cmake $output_dir/cmake-default

co="-DHYPRE_SEQUENTIAL=ON"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo
./renametest.sh cmake $output_dir/cmake-sequential

co="-DHYPRE_SHARED=ON"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo
./renametest.sh cmake $output_dir/cmake-shared

co="-DHYPRE_SINGLE=ON"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo -ro: -single
./renametest.sh cmake $output_dir/cmake-single

co="-DHYPRE_LONG_DOUBLE=ON"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo -ro: -longdouble
./renametest.sh cmake $output_dir/cmake-longdouble

co="-DCMAKE_BUILD_TYPE=Debug"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh cmake $output_dir/cmake-debug

co="-DHYPRE_BIGINT=ON"
./test.sh cmake.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh cmake $output_dir/cmake-bigint

# cmake build doesn't currently support maxdim
# cmake build doesn't currently support complex

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
