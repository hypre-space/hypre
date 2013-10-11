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
tests=""

# Echo usage information
while [ "$*" ]
do
   case $1 in
      -h|-help)
         cat <<EOF

   $0 [options] {ex_dir} [options for examples]

   where: {ex_dir}  is the hypre examples directory
          -<test>   run <test>  (test = default, bigint, maxdim, complex)
          -h|-help  prints this usage information and exits

   This script builds the hypre example codes in {ex_dir} and runs the example
   regression tests in test/TEST_examples.

   Example usage: $0 -maxdim ../examples

EOF
         exit
         ;;

      -*)
         tname=`echo $1 | sed 's/-//'`
         tests="$tests $tname"
         shift
         ;;
      *)
         break
         ;;
   esac
done

# If no tests were specified, run default
if [ "$tests" = "" ]; then
   tests="default"
fi

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
ex_dir=$1
shift

# Run make in the examples directory
cd $ex_dir
make clean
for tname in $tests
do
   make $@ $tname
done

# Run the examples regression test
cd ../test
for tname in $tests
do
   ./runtest.sh $@ TEST_examples/$tname.sh
done

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
