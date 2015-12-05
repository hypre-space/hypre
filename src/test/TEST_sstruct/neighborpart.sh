#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.3 $
#EHEADER**********************************************************************


TNAME=`basename $0 .sh`

#=============================================================================
# Test SetNeighborPart by comparing one-part problem against
# equivalent multi-part problems
#=============================================================================

tail -3 ${TNAME}.out.0 > ${TNAME}.testdata

for i in 1 2 3
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

#=============================================================================
# Test SetNeighborPart by comparing multi-part problems
#=============================================================================

tail -3 ${TNAME}.out.10 > ${TNAME}.testdata
tail -3 ${TNAME}.out.11 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.20 > ${TNAME}.testdata
tail -3 ${TNAME}.out.21 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.30 > ${TNAME}.testdata
for i in 31 32
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

tail -3 ${TNAME}.out.40 > ${TNAME}.testdata
tail -3 ${TNAME}.out.41 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.50 > ${TNAME}.testdata
for i in 51 52
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

tail -3 ${TNAME}.out.60 > ${TNAME}.testdata
for i in 61 62
do
   tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
   diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
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
   diff -U3 -bI"time" ${TNAME}.saved ${TNAME}.out >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
