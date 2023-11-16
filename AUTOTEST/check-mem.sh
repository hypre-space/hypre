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

   This script checks memory manager usage in hypre.

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
  egrep -v '/hypre/include' |
  egrep -v '/utilities/memory_tracker.c' |
  egrep -v '/utilities/memory.c' > check-mem.files

egrep '(^|[^[:alnum:]_]+)malloc[[:space:]]*\('  `cat check-mem.files` >&2
egrep '(^|[^[:alnum:]_]+)calloc[[:space:]]*\('  `cat check-mem.files` >&2
egrep '(^|[^[:alnum:]_]+)realloc[[:space:]]*\(' `cat check-mem.files` >&2
egrep '(^|[^[:alnum:]_]+)free[[:space:]]*\('    `cat check-mem.files` >&2

rm -f check-mem.files
