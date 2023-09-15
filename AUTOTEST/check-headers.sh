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

   $0 [-h] {src_dir}

   where: {src_dir}     is the hypre source directory
          -h|-help      prints this usage information and exits

   This script checks hypre header usage.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

# Configure and make library
cd $src_dir
./configure --enable-debug
make clean
make -j test

# Make examples and check header usage
cd examples
make clean
make COPTS="-H -g -Wall" |& grep "hypre/include" | grep -v "HYPRE" >&2
