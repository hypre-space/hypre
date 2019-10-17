#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# Test SetNeighborPart by comparing one-part problem against
# equivalent multi-part problems
#=============================================================================

tail -3 ${TNAME}.out.0 > ${TNAME}.testdata

for i in 1 2 3
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
done

#=============================================================================
# Test SetNeighborPart by comparing multi-part problems
#=============================================================================

tail -3 ${TNAME}.out.10 > ${TNAME}.testdata
tail -3 ${TNAME}.out.11 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

tail -3 ${TNAME}.out.20 > ${TNAME}.testdata
tail -3 ${TNAME}.out.21 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

tail -3 ${TNAME}.out.30 > ${TNAME}.testdata
for i in 31 32
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
done

tail -3 ${TNAME}.out.40 > ${TNAME}.testdata
tail -3 ${TNAME}.out.41 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

tail -3 ${TNAME}.out.50 > ${TNAME}.testdata
for i in 51 52
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
done

tail -3 ${TNAME}.out.60 > ${TNAME}.testdata
for i in 61 62
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
done

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.10\
 ${TNAME}.out.11\
 ${TNAME}.out.20\
 ${TNAME}.out.21\
 ${TNAME}.out.30\
 ${TNAME}.out.31\
 ${TNAME}.out.32\
 ${TNAME}.out.40\
 ${TNAME}.out.41\
 ${TNAME}.out.50\
 ${TNAME}.out.51\
 ${TNAME}.out.52\
 ${TNAME}.out.60\
 ${TNAME}.out.61\
 ${TNAME}.out.62\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
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
