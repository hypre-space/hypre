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

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script tests the documentation build (on the tux machines).

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

# Make sure Makefile.config is generated
cd $src_dir
./configure > /dev/null 2>&1

# Test documentation build in docs/
cd docs
make clean
make

# Test documentation build in examples/docs/
cd ../examples/docs
make distclean
make
