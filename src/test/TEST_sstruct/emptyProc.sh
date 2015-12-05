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
# $Revision: 1.10 $
#EHEADER**********************************************************************





TNAME=`basename $0 .sh`

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
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
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
   diff -U3 -bI"time" ${TNAME}.saved ${TNAME}.out >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
