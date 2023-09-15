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

   $0 [-h|-help] {top_dir}

   where: {top_dir}  is the top-level hypre release directory
          -h|-help   prints this usage information and exits

   This script checks for case-insensitive filename matches.

   Example usage: $0 ..

EOF
      exit
      ;;
esac

# Setup
top_dir=`cd $1; pwd`
shift

cd $top_dir

find -type f | sort -f | uniq -i -d >&2

