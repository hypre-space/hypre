#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 00 01 02 03 04 05 06    08 09\
 10 11 12    14 15 16 17 18   \
 20 21 22 23 24 25 26 27 28 29\
 30 31 32 33 34 35 36 37 38   \
"

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

# Make sure that the output files are reasonable
CHECK_LINE="Iterations"
OUT_COUNT=`grep "$CHECK_LINE" ${TNAME}.out | wc -l`
SAVED_COUNT=`grep "$CHECK_LINE" ${TNAME}.saved | wc -l`
if [ "$OUT_COUNT" != "$SAVED_COUNT" ]; then
   echo "Incorrect number of \"$CHECK_LINE\" lines in ${TNAME}.out" >&2
fi

if [ -z $HYPRE_NO_SAVED ]; then
   (../runcheck.sh ${TNAME}.out ${TNAME}.saved $RTOL $ATOL) >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
