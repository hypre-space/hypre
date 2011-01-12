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

   $0 [-h] {ex_dir} [options for examples make]

   where: {ex_dir}  is the hypre examples directory
          -h|-help  prints this usage information and exits

   This script builds the hypre example codes in {ex_dir} and runs the example
   regression tests in test/TEST_examples.

   Example usage: $0 ../examples

EOF
   exit
   ;;
esac

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
ex_dir=$1
shift

# Run make in the examples directory
cd $ex_dir
make clean
make $@

# Run the examples regression test
cd ../test
./runtest.sh $@ TEST_examples/default.sh

# Collect all error files from the tests
for errfile in $( find TEST_examples -name "*.err" -o -name "*.fil" -o -name "*.out*" )
do
   mkdir -p $output_dir/`dirname $errfile`
   mv -f $errfile $output_dir/$errfile
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
