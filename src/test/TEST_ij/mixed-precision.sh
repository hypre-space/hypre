#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

FILES="\
 ${TNAME}.out.600\
 ${TNAME}.out.601\
 ${TNAME}.out.602\
 ${TNAME}.out.603\
 ${TNAME}.out.604\
 ${TNAME}.out.605\
 ${TNAME}.out.606\
 ${TNAME}.out.607\
 ${TNAME}.out.608\
 ${TNAME}.out.609\
 ${TNAME}.out.610\
 ${TNAME}.out.611\
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

