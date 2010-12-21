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

   **** Only run this script on the Hera machine ****

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs a number of tests suitable for the Hera machine.

   Example usage: $0 ..

EOF
      exit
      ;;
esac

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=$1
shift

# Test runtest tests
./test.sh default.sh $src_dir
mv -f default.??? $output_dir

# Test linking for different languages
link_opts="all++ all77"
for opt in $link_opts
do
   output_subdir=$output_dir/link$opt
   mkdir -p $output_subdir
   ./test.sh link.sh $src_dir $opt
   mv -f link.??? $output_subdir
done

# Test other builds
# temporarily change word delimeter in order to have spaces in options
tmpIFS=$IFS
IFS=:
configure_opts="--enable-debug:--with-openmp:--enable-bigint"
for opt in $configure_opts
do
    # only use first part of $opt for subdir name
    output_subdir=$output_dir/build`echo $opt | awk '{print $1}'`
    mkdir -p $output_subdir
    ./test.sh configure.sh $src_dir $opt 
    mv -f configure.??? $output_subdir
    ./test.sh make.sh $src_dir test
    mv -f make.??? $output_subdir
done
IFS=$tmpIFS

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
