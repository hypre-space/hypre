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
odr=$output_dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=`cd $1; pwd`
root_dir=`cd $src_dir/..; pwd`
shift

# Organizing the tests from "fast" to "slow"

# Check license header info
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
npt=${NPT:-4}
mo="-j test"
ro="-ams -ij -sstruct -sstructmat -struct -structmat -lobpcg -rt -j $npt -countjobs"
eo=""

co=""
./test.sh -od $odr -tn default basic.sh $src_dir -co: $co -mo: $mo

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
( cd $src_dir; make distclean 1>&2 > /dev/null )

co="--without-MPI"
./test.sh -od $odr -tn without-MPI basic.sh $src_dir -co: $co -mo: $mo

co="--with-strict-checking"
./test.sh -od $odr -tn strict-checking basic.sh $src_dir -co: $co -mo: $mo

co="--enable-shared"
./test.sh -od $odr -tn shared basic.sh $src_dir -co: $co -mo: $mo

co="--enable-debug --with-extra-CFLAGS=\\'-Wstrict-prototypes\\'"
./test.sh -od $odr -tn debug1 basic.sh $src_dir -co: $co -mo: $mo -eo: $eo

co="--enable-maxdim=4 --enable-debug"
./test.sh -od $odr -tn maxdim_4 basic.sh $src_dir -co: $co -mo: $mo -eo: -maxdim

# co="--enable-complex --enable-maxdim=4 --enable-debug"
# ./test.sh basic.sh $src_dir -co: $co -mo: $mo -eo: -complex
# # ignore complex compiler output for now
# rm -fr basic.dir/make.???
# grep -v make.err basic.err > basic.tmp
# mv basic.tmp basic.err
# ./renametest.sh basic $output_dir/basic--enable-complex

#co="--with-openmp"
#RO="-ams -ij -sstruct -sstructmat -struct -structmat -lobpcg -rt -D HYPRE_NO_SAVED -nthreads 2"
#./test.sh -od $odr -tn openmp basic.sh $src_dir -co: $co -mo: $mo -ro: $RO
#
#co="--with-openmp --enable-hopscotch"
#RO="-ij -sstruct -struct -lobpcg -rt -D HYPRE_NO_SAVED -nthreads 2"
#./test.sh -od $odr -tn concurrent-hopscotch basic.sh $src_dir -co: $co -mo: $mo -ro: $RO

co="--enable-single --enable-debug"
./test.sh -od $odr -tn single basic.sh $src_dir -co: $co -mo: $mo -ro: "-single -rt -j $npt"

co="--enable-longdouble --enable-debug"
./test.sh -od $odr -tn longdouble basic.sh $src_dir -co: $co -mo: $mo -ro: "-longdouble -rt -j $npt"

co="--enable-debug CC=mpiCC"
./test.sh -od $odr -tn debug-cpp basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -eo: $eo

co="--enable-bigint --enable-debug"
./test.sh -od $odr -tn bigint basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -eo: -bigint

co="--enable-debug --enable-mixed-precision"
./test.sh -od $odr -tn mixed-precision basic.sh $src_dir -co: $co -mo: $mo -ro: $ro

co="--enable-debug --enable-mixedint"
RO="-ams -ij-mixed -sstruct-mixed -struct -lobpcg-mixed -rt -j $npt"
./test.sh -od $odr -tn mixedint basic.sh $src_dir -co: $co -mo: $mo -ro: $RO

# RDF: This is currently in 'machine-tux-valgrind.sh'.
# co="--enable-debug --with-print-errors"
# ./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro -error -rt -valgrind
# ./renametest.sh basic $output_dir/basic--valgrind

# CMake build and run tests
mo="-j"
ro="-ams -ij -sstruct -sstructmat -struct -structmat -lobpcg -rt -j $npt"
eo=""

co=""
./test.sh -od $odr -tn default cmake.sh $root_dir -co: $co -mo: $mo

co="-DHYPRE_ENABLE_MPI=OFF"
./test.sh -od $odr -tn sequential cmake.sh $root_dir -co: $co -mo: $mo

co="-DBUILD_SHARED_LIBS=ON"
./test.sh -od $odr -tn shared cmake.sh $root_dir -co: $co -mo: $mo

co="-DHYPRE_ENABLE_SINGLE=ON"
./test.sh -od $odr -tn single cmake.sh $root_dir -co: $co -mo: $mo -ro: "-single -rt -j $npt"

co="-DHYPRE_ENABLE_LONG_DOUBLE=ON"
./test.sh -od $odr -tn longdouble cmake.sh $root_dir -co: $co -mo: $mo -ro: "-longdouble -rt -j $npt"

co="-DHYPRE_ENABLE_COMPLEX=ON"
./test.sh -od $odr -tn complex cmake.sh $root_dir -co: $co -mo: $mo

co="-DHYPRE_ENABLE_BIGINT=ON"
./test.sh -od $odr -tn bigint cmake.sh $root_dir -co: $co -mo: $mo -ro: $ro

co="-DHYPRE_ENABLE_MIXEDINT=ON"
./test.sh -od $odr -tn mixedint cmake.sh $root_dir -co: $co -mo: $mo -ro: $ro

co="-DCMAKE_BUILD_TYPE=Debug"
./test.sh -od $odr -tn debug cmake.sh $root_dir -co: $co -mo: $mo -ro: $ro

co="-DCMAKE_BUILD_TYPE=Debug -DHYPRE_ENABLE_MIXED_PRECISION=ON"
./test.sh -od $odr -tn mixed-precision cmake.sh $root_dir -co: $co -mo: $mo

# cmake build doesn't currently support maxdim

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
