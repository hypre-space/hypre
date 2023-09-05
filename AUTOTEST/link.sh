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

   $0 [-h] {src_dir} [options for make in the test directory]

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs make clean; make [options] in {src_dir}/test.

   Example usage: $0 ../src all++

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

# Run make
cd $src_dir/test
make clean
make $@
