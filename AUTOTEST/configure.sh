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

   $0 [-h] {src_dir} [options for configure]

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs configure in {src_dir} with optional parameters.

   Example usage: $0 ../src --enable-debug

EOF
      exit
      ;;
esac

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=`cd $1; pwd`
shift

# Run configure
cd $src_dir
if [ "`uname -s`" = "AIX" ]
then
   nopoe ./configure $@
else
   ./configure $@
fi

# Save config.log, HYPRE_config.h and Makefile.config
cp config.log HYPRE_config.h config/Makefile.config $output_dir

# Save the environment variables
set > $output_dir/sh.env

