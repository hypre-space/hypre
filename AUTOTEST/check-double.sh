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

   This script checks for 'double' in sections of hypre.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

cd $src_dir

find . -type f -print | egrep '[.]*[.](c|cc|cpp|cxx|C|h|hpp|hxx|H)$' |
  egrep -v '/cmbuild' |
  egrep -v '/docs' |
  egrep -v '/examples' |
  egrep -v '/FEI_mv' |
  egrep -v '/hypre/include' > check-double.files

egrep '(^|[^[:alnum:]_-]+)double([^[:alnum:]_-]+|$)' `cat check-double.files` >&2

rm -f check-double.files
