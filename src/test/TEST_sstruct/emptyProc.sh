#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# sstruct: Test various empty proc problems
#=============================================================================

TNUMS="\
 00 01 02 03 04 05       08 09\
 10 11 12       15 16 17 18   \
    21 22 23 24 25 26 27 28 29\
 30 31 32 33 34 35 36 37 38   \
"
# 06 14 20

for i in $TNUMS
do
  tail -3 ${TNAME}.out.${i}  > ${TNAME}.testdata
  tail -3 ${TNAME}.out.1${i} > ${TNAME}.testdata.temp
  (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
done

#=============================================================================
# compare with baseline case
#=============================================================================

for i in $TNUMS
do
  echo "# Output file: ${TNAME}.out.${i}"
  tail -3 ${TNAME}.out.${i}
done > ${TNAME}.out

# Make sure that the output file is reasonable
RUNCOUNT=`echo $TNUMS | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
