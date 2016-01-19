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

   $0 [-h] {src_dir} [options for make]

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs make clean; make [options] in {src_dir}.

   Example usage: $0 ../src test

EOF
      exit
      ;;
esac

# Setup
src_dir=`cd $1; pwd`
shift

# Run make
cd $src_dir
make clean
make $@
