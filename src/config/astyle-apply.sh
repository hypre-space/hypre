#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

scriptname=`basename $0 .sh`

# Check number of arguments
if [ $# -lt 1 ]; then
    echo "Need at least one argument"
    exit
fi

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script applies indentation style to hypre source files.
   It also generates the internal '_hypre*.h' header files.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

cd $src_dir

# Check for correct version of astyle
astyle_version="Artistic Style Version 3.1"
if [ ! -x "$(command -v astyle)" ]; then
  echo "$astyle_version not found"
  exit
elif [ "$(astyle --version)" != "$astyle_version" ]; then
  echo "Please use $astyle_version"
fi

# Generate list of source files to indent
find . -type f -print | egrep '[.]*[.](c|cc|cpp|cxx|C|h|hpp|hxx|H)$' |
  egrep -v '/cmbuild' |
  egrep -v '/docs' |
  egrep -v '/FEI_mv' |
  egrep -v '/blas' |
  egrep -v '/lapack' |
  egrep -v '/distributed' |
  egrep -v '/hypre/include' |
  egrep -v '/HYPREf[.]h' |
  egrep -v '/utilities/HYPRE_error_f[.]h' |
  egrep -v '/utilities/cub_allocator[.]h' |
  egrep -v '/_hypre_.*[.]h' > $scriptname.files

# Apply indentation style to source files
astyle_result=$(astyle --options=config/astylerc $(cat $scriptname.files))

if [ -n "$astyle_result" ]; then
  echo "Please make sure changes are committed"
else
  echo "No source files were changed"
fi

# Run headers scripts
for i in $(find . -name 'headers')
do
  dir=$(dirname $i)
  (cd $dir; ./headers)
done

rm -f $scriptname.files
