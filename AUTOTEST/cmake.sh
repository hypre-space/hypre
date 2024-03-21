#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

testname=`basename $0 .sh`

drivers="ij sstruct struct ams_driver maxwell_unscaled sstruct_fac ij_mv struct_migrate ij_assembly"

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   $0 [-h] {src_dir} [options]

   where: {src_dir}     is the hypre source directory
          -co: <opts>   configuration options
          -mo: <opts>   make options
          -ro: <opts>   call the run script with these options
          -eo: <opts>   call the examples script with these options
          -h|-help      prints this usage information and exits

   This script uses cmake to configure and compile the source in {src_dir}, then
   optionally runs driver and example tests.

   Example usage: $0 ../src -co -DCMAKE_BUILD_TYPE=Debug -ro: -ij

EOF
      exit
      ;;
esac

# Set src_dir
src_dir=`cd $1; pwd`
shift

# Parse the rest of the command line
copts="-DHYPRE_BUILD_TESTS=ON"
mopts=""
ropts=""
eopts=""
while [ "$*" ]
do
   case $1 in
      -co:)
         opvar="copts"; shift
         ;;
      -mo:)
         opvar="mopts"; shift
         ;;
      -ro:)
         opvar="ropts"; rset="yes"; shift
         ;;
      -eo:)
         opvar="eopts"; eset="yes"; shift
         ;;
      *)
         eval $opvar=\"\$$opvar $1\"
         shift
         ;;
   esac
done

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
cd $src_dir
src_dir=`pwd`

# Clean up the cmbuild directories (do it from src_dir as a precaution)
cd $src_dir
rm -fr `echo cmbuild/* | sed 's/[^ ]*README.txt//g'`

# Clean up the previous install
cd $src_dir
rm -fr hypre

# Configure
cd $src_dir/cmbuild
eval cmake $copts ..
make $mopts install

# Make
cd $src_dir/cmbuild/test
make $mopts
mv -f $drivers ../../test

cd $test_dir

# Run
if [ -n "$rset" ]; then
   ./test.sh run.sh $src_dir $ropts
   mv -f run.??? $output_dir
fi

# Examples
if [ -n "$eset" ]; then
   ./test.sh examples.sh $src_dir $eopts
   mv -f examples.??? $output_dir
fi

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

# Clean up
cd $src_dir
rm -fr `echo cmbuild/* | sed 's/[^ ]*README.txt//g'`
rm -fr hypre
( cd $src_dir/test; rm -f $drivers; ./cleantest.sh )

