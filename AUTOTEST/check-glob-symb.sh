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

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script checks if there are any globally defined symbols in libHYPRE.a
   without the appropriate namespace protection.

   Example usage: $0 ..

EOF
      exit
      ;;
esac

# Setup
src_dir=$1
shift

cd $src_dir

if [ -f lib/libHYPRE.a ]; then
   nm -o --extern-only --defined-only lib/libHYPRE.a | grep -vi hypre_ | grep -vi mli_ | grep -vi fei_ | grep -vi Euclid | grep -vi ParaSails | grep -v " _Z" >&2
else
   echo "check-glob-symb.sh can't find lib/libHYPRE.a"
fi

