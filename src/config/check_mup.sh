#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

## Call from directory where multiprecision functions are.
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

# extract directory rootname
rootdir=$PWD
rootdir="${rootdir%/}"
rootname="${rootdir##*/}"

# this directory
BASEDIR=$(dirname $0)

# 1. Generate current list of function names
$BASEDIR/generate_function_list.sh
# 2. Compare against saved multiprecision object names
#diff -wc ${rootname}_functions.saved ${rootname}_functions.new > ${rootname}_functions.err
## diff on sorted lists 
bash -c 'diff -wc <(sort '${rootname}'_functions.saved) <(sort '${rootname}'_functions.new) > '${rootname}'_functions.err'

## overwrite saved file with new file
#cp ${rootname}'_functions.new' ${rootname}'_functions.saved'
