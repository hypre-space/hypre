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
# NOTE: The use of 'eval' is needed to deal properly with nested quotes in argument lists
cd $src_dir
if [ "`uname -s`" = "AIX" ]
then
   eval nopoe ./configure $@
else
   eval ./configure $@
fi

# Save config.log, HYPRE_config.h and Makefile.config
cp config.log HYPRE_config.h config/Makefile.config $output_dir

# Save the environment variables
set > $output_dir/sh.env

