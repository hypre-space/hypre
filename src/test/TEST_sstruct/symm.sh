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
# $Revision: 1.11 $
#EHEADER**********************************************************************





TNAME=`basename $0 .sh`

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_SSTRUCT data type (2D)
#=============================================================================

tail -3 ${TNAME}.out.20 > ${TNAME}.testdata
tail -3 ${TNAME}.out.21 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.22 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.23 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_PARCSR data type (2D)
#=============================================================================

tail -3 ${TNAME}.out.24 > ${TNAME}.testdata
tail -3 ${TNAME}.out.25 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_SSTRUCT data type (3D)
#=============================================================================

tail -3 ${TNAME}.out.30 > ${TNAME}.testdata
tail -3 ${TNAME}.out.31 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.32 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.33 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_PARCSR data type (3D)
#=============================================================================

tail -3 ${TNAME}.out.34 > ${TNAME}.testdata
tail -3 ${TNAME}.out.35 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.20\
 ${TNAME}.out.21\
 ${TNAME}.out.22\
 ${TNAME}.out.23\
 ${TNAME}.out.24\
 ${TNAME}.out.25\
 ${TNAME}.out.30\
 ${TNAME}.out.31\
 ${TNAME}.out.32\
 ${TNAME}.out.33\
 ${TNAME}.out.34\
 ${TNAME}.out.35\
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
