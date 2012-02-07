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
# $Revision$
#EHEADER**********************************************************************


TNAME=`basename $0 .sh`

#=============================================================================
# Check SetNeighborBox for ${TNAME} problems (2D)
#=============================================================================

tail -3 ${TNAME}.out.20 > ${TNAME}.testdata
tail -3 ${TNAME}.out.21 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# Check SetNeighborBox for ${TNAME} problems (3D)
#=============================================================================

tail -3 ${TNAME}.out.30 > ${TNAME}.testdata
tail -3 ${TNAME}.out.31 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# Check SysPFMG for power-of-two and non-power-of-two systems
#=============================================================================

tail -3 ${TNAME}.out.40 > ${TNAME}.testdata
tail -3 ${TNAME}.out.41 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.42 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.50 > ${TNAME}.testdata
tail -3 ${TNAME}.out.51 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.52 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# Check PFMG, SMG, and SysPFMG for problems with period larger than the grid
#=============================================================================

# First check that sstruct and struct are the same here
tail -3 ${TNAME}.out.60 > ${TNAME}.testdata
tail -3 ${TNAME}.out.61 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.62 > ${TNAME}.testdata
tail -3 ${TNAME}.out.63 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

# Also check that PFMG and SysPFMG are the same
tail -3 ${TNAME}.out.66 > ${TNAME}.testdata
tail -3 ${TNAME}.out.67 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# Check that reverse communication used to AddValues still works
#=============================================================================

#=============================================================================
# Check SetPeriodic for node/cell problems and STRUCT, SSTRUCT, PARCSR types
#=============================================================================

tail -3 ${TNAME}.out.80 > ${TNAME}.testdata
TNUM="81 82 83 84 85"
for i in $TNUM
do
  tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

tail -3 ${TNAME}.out.90 > ${TNAME}.testdata
TNUM="91 92 93 94 95"
for i in $TNUM
do
  tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

#=============================================================================
# compare with baseline case
#=============================================================================

TNUM="\
 20 21\
 30 31\
 40 41 42\
 50 51 52\
 60 61 62 63 65 66 67\
 70\
 80 81 82 83 85\
 90 91 92 93 95\
"

for i in $TNUM
do
  FILE="${TNAME}.out.$i"
  echo "# Output file: ${FILE}"
  tail -3 ${FILE}
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
