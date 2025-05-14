#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

## Call from directory where multiprecision functions are.
# Generate file containing function names in current folder.
# This will be used to generate a header file for transforming 
# multiprecision function names. It will also be useful for 
# regression testing for the multiprecision build. 
#
# NOTE: It must be run on symbols generated from the 
# non-multiprecision build of hypre.

# Assumes hypre has been built (non-multiprecision) to generate object files
###
#  * make distclean
#  * ./configure –enable-debug
#  * make -s 
###

# extract directory rootname
rootdir=$PWD
rootdir="${rootdir%/}"
rootname="${rootdir##*/}"

## extract function names and **remove** leading and trailing underscores, if any
## To include local functions (static functions) test on $2=="t" as well (use: $2=="t|T"{print $3})
#if [ ${rootname} == "utilities" ]
#then
#   nm -A --defined-only *.o* | awk -F'[ ]' '$2=="T"{print $3}' | sed 's/_*//;s/_*$//' > ${rootname}_functions.new
#else
## NOTE: This will exclude functions beginning with HYPRE_
#   nm -A --defined-only *.o* | awk -F'[ ]' '$2=="T"{print $3}' | sed 's/_*//;s/_*$//' | sed -n '/^HYPRE_/ !p' > ${rootname}_functions.new
#fi

## extract function names and **remove** trailing underscores, if any
## To include local functions (static functions) test on $2=="t" as well (use: $2=="t|T"{print $3})
#if [ ${rootname} == "utilities" ]
#then
   nm -A --defined-only *.o* | awk -F'[ ]' '$2=="T"{print $3}' | sed 's/_*$//' > ${rootname}_functions.new
#else
## NOTE: This will exclude functions beginning with HYPRE_
#   nm -A --defined-only *.o* | awk -F'[ ]' '$2=="T"{print $3}' | sed 's/_*$//' | sed -n '/^HYPRE_/ !p' > ${rootname}_functions.new
#fi
