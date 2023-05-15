#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Check that current multiprecision functions are up to date. 
# It will also be useful for regression testing for the multiprecision build. 
#
# NOTE: It must be run on symbols generated from the 
# non-multiprecision build of hypre.

# Assumes hypre has been built (non-multiprecision) to generate object files
###
#  * make distclean
#  * ./configure –enable-debug
#  * make -s 
###

rootdir=$PWD
rootdir="${rootdir%/}"
rootname="${rootdir##*/}"

# 1. Generate current list of function names
../config/generate_function_list.sh
# 2. Compare against saved multiprecision object names
#diff -wc ${rootname}_functions.saved ${rootname}_functions.out > ${rootname}_functions.err
## diff on sorted list 
bash -c 'diff -wc <(sort '${rootname}'_functions.saved) <(sort '${rootname}'_functions.out) > '${rootname}'_functions.err'
