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

   $0 [-h] {src_dir} [options for make]

   where: {src_dir}     is the hypre source directory
          -spack <dir>  compile and link drivers to spack build
          -h|-help      prints this usage information and exits

   This script runs make clean; make [options] in {src_dir}.

   Example usage: $0 ../src test

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

# Parse the rest of the command line
mopts=""
while [ "$*" ]
do
   case $1 in
      -spack)
         shift; spackdir="$1"; shift
         ;;
      *)
         mopts="$mopts $1"; shift
         ;;
   esac
done

# Run make
cd $src_dir
make clean
if [ -n "$spackdir" ]; then
   cd $src_dir/test
   make HYPRE_BUILD_DIR="$spackdir" $mopts
else
   make $mopts
fi
