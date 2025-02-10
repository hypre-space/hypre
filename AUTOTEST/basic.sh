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

   $0 [-h] {src_dir} [options]

   where: {src_dir}     is the hypre source directory
          -co: <opts>   options for configure script
          -mo: <opts>   options for make script
          -ro: <opts>   call the run script with these options
          -eo: <opts>   call the examples script with these options
          -h|-help      prints this usage information and exits

   This script configures and compiles the source in {src_dir}, then optionally
   runs driver and example tests.

   Example usage: $0 ../src -ro: -ij -sstruct

EOF
      exit
      ;;
esac

# Set src_dir
src_dir=`cd $1; pwd`
shift

# Parse the rest of the command line
copts=""
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

# Configure
# NOTE: The use of 'eval' is needed to deal properly with nested quotes in argument lists
eval ./test.sh configure.sh $src_dir $copts
mv -f configure.??? $output_dir

# Make
./test.sh make.sh $src_dir $mopts
mv -f make.??? $output_dir

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
( cd $src_dir; make distclean )

