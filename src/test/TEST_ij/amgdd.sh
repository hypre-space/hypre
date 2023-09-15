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
 ${TNAME}.out.900\
 ${TNAME}.out.901\
 ${TNAME}.out.902\
 ${TNAME}.out.903\
 ${TNAME}.out.904\
 ${TNAME}.out.905\
 ${TNAME}.out.906\
 ${TNAME}.out.910\
 ${TNAME}.out.911\
 ${TNAME}.out.912\
 ${TNAME}.out.913\
 ${TNAME}.out.914\
 ${TNAME}.out.915\
 ${TNAME}.out.916\
 ${TNAME}.out.917\
 ${TNAME}.out.918\
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

