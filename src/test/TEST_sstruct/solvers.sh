#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
 ${TNAME}.out.7\
 ${TNAME}.out.8\
 ${TNAME}.out.9\
 ${TNAME}.out.10\
 ${TNAME}.out.11\
 ${TNAME}.out.14\
 ${TNAME}.out.15\
 ${TNAME}.out.16\
 ${TNAME}.out.17\
 ${TNAME}.out.18\
 ${TNAME}.out.19\
 ${TNAME}.out.20\
 ${TNAME}.out.21\
 ${TNAME}.out.22\
 ${TNAME}.out.23\
 ${TNAME}.out.24\
 ${TNAME}.out.25\
 ${TNAME}.out.26\
 ${TNAME}.out.27\
 ${TNAME}.out.28\
 ${TNAME}.out.29\
 ${TNAME}.out.30\
 ${TNAME}.out.31\
 ${TNAME}.out.32\
 ${TNAME}.out.33\
 ${TNAME}.out.34\
 ${TNAME}.out.35\
 ${TNAME}.out.36\
 ${TNAME}.out.37\
 ${TNAME}.out.38\
 ${TNAME}.out.39\
 ${TNAME}.out.40\
 ${TNAME}.out.41\
 ${TNAME}.out.42\
 ${TNAME}.out.43\
 ${TNAME}.out.44\
 ${TNAME}.out.45\
 ${TNAME}.out.46\
 ${TNAME}.out.47\
 ${TNAME}.out.48\
 ${TNAME}.out.49\
 ${TNAME}.out.50\
 ${TNAME}.out.52\
 ${TNAME}.out.53\
 ${TNAME}.out.56\
 ${TNAME}.out.57\
 ${TNAME}.out.58\
 ${TNAME}.out.59\
 ${TNAME}.out.121\
 ${TNAME}.out.122\
 ${TNAME}.out.123\
 ${TNAME}.out.124\
 ${TNAME}.out.125\
 ${TNAME}.out.126\
 ${TNAME}.out.127\
 ${TNAME}.out.128\
 ${TNAME}.out.129\
 ${TNAME}.out.130\
 ${TNAME}.out.131\
 ${TNAME}.out.132\
 ${TNAME}.out.133\
 ${TNAME}.out.134\
 ${TNAME}.out.135\
 ${TNAME}.out.136\
 ${TNAME}.out.137\
 ${TNAME}.out.138\
 ${TNAME}.out.139\
 ${TNAME}.out.140\
 ${TNAME}.out.141\
 ${TNAME}.out.142\
 ${TNAME}.out.143\
"
# ${TNAME}.out.5\
# ${TNAME}.out.6\
# ${TNAME}.out.12\
# ${TNAME}.out.13\
# ${TNAME}.out.14\
# ${TNAME}.out.51\
# ${TNAME}.out.54\
# ${TNAME}.out.55\

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

# rm -f ${TNAME}.testdata*
