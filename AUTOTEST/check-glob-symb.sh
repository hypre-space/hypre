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

   This script checks if there are any globally defined symbols in libHYPRE.a
   without the appropriate namespace protection.

   Example usage: $0 ../src

EOF
   exit
   ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

cd $src_dir

# find global symbols
if [ -f lib/libHYPRE.a ]; then
  nm -o --extern-only --defined-only lib/libHYPRE.a |
    grep -vi hypre_ |
    grep -vi mli_ |
    grep -vi fei_ |
    grep -vi Euclid |
    grep -vi ParaSails |
    grep -v " _Z" > check-glob-symb.temp
else
   echo "check-glob-symb.sh can't find lib/libHYPRE.a"
fi

# find the '.o' file directories and add them to the output for filtering
while read line
do
  sym=`echo $line | awk -F: '{print $2}'`
  for dir in `find . -name $sym`
  do
    echo $line | awk -v dir=$dir -F: 'BEGIN {OFS=FS} {print $1,dir,$3}'
  done
done < check-glob-symb.temp >&2

rm -f check-glob-symb.temp
