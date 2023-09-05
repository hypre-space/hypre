#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

FILES="\
 ${TNAME}.out.300\
 ${TNAME}.out.301\
 ${TNAME}.out.302\
 ${TNAME}.out.303\
 ${TNAME}.out.304\
 ${TNAME}.out.305\
 ${TNAME}.out.306\
 ${TNAME}.out.307\
 ${TNAME}.out.308\
 ${TNAME}.out.309\
 ${TNAME}.out.310\
 ${TNAME}.out.311\
 ${TNAME}.out.312\
 ${TNAME}.out.313\
 ${TNAME}.out.314\
 ${TNAME}.out.315\
 ${TNAME}.out.316\
 ${TNAME}.out.317\
 ${TNAME}.out.318\
 ${TNAME}.out.319\
 ${TNAME}.out.320\
 ${TNAME}.out.321\
 ${TNAME}.out.322\
 ${TNAME}.out.323\
 ${TNAME}.out.324\
 ${TNAME}.out.325\
"

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

