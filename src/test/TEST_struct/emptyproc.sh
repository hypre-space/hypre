#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

FILES="\
 ${TNAME}.out.00a\
 ${TNAME}.out.01a\
 ${TNAME}.out.01b\
 ${TNAME}.out.03a\
 ${TNAME}.out.03b\
 ${TNAME}.out.04a\
 ${TNAME}.out.04b\
 ${TNAME}.out.11a\
 ${TNAME}.out.13a\
 ${TNAME}.out.14a\
 ${TNAME}.out.17a\
 ${TNAME}.out.18a\
 ${TNAME}.out.21a\
 ${TNAME}.out.31a\
 ${TNAME}.out.41a\
 ${TNAME}.out.51a\
 ${TNAME}.out.61a\
"

#=============================================================================
# check results when there are processors with no data
#=============================================================================

for i in $FILES
do
  tail -3 $i > ${TNAME}.testdata
  for j in $i.*
  do
    tail -3 $j > ${TNAME}.testdata.temp
    diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
  done
done

#=============================================================================
# compare with baseline case
#=============================================================================

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

rm -f ${TNAME}.testdata*
