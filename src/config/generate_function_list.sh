#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# generate file containing function names in current folder.
# This will be used to generate a header file for transforming 
# multiprecision function names. It will also be useful for 
# regression testing for the multiprecision build. 
#
# NOTE: It must be run on symbols generated from the 
# non-multiprecision build of hypre.

rootdir=$PWD
rootdir="${rootdir%/}"
rootname="${rootdir##*/}"

## extract function names and remove leading and trailing underscores, if any
## To include local functions (static functions) test on $2=="t" as well
nm -A --defined-only *.o* | awk -F'[ ]' '$2=="T"{print $3}' | sed 's/_*//;s/_*$//' > ${rootname}_functions.out
