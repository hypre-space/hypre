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

   $0 [options] {src_dir} [options for 'runtest.sh']

   where: {src_dir}  is the hypre source directory

   with options (more than one set of tests can be specified):

      -all       run ams, fac, ij, sstruct, struct (default behavior)
      -<test>    run <test>
      -h|-help   prints this usage information and exits

   This script runs runtest.sh in {src_dir}/test with optional parameters.

   Example usage: $0 .. -D HYPRE_NO_SAVED

EOF
         exit
         ;;

      -all)
         shift
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

# If no tests were specified, run all tests
if [ "$tests" = "" ]; then
   tests="ams fac ij sstruct struct"
fi

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=$1
shift

# Run the test drivers
cd $src_dir/test
./cleantest.sh
for tname in $tests
do
   ./runtest.sh $@ TEST_$tname/*.sh
done

# Collect all error files from the tests
for errfile in $( find . -name "*.err" -o -name "*.fil" -o -name "*.out*" )
do
   mkdir -p $output_dir/`dirname $errfile`
   mv -f $errfile $output_dir/$errfile
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
