#!/bin/ksh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.15 $
#EHEADER**********************************************************************





TNAME=`basename $0 .sh`

#=============================================================================
# The outputs below should differ only in timings.
#=============================================================================

diff -bI"time" solvers.out.0 solvers.out.1 >&2
diff -bI"time" solvers.out.2 solvers.out.3 >&2
diff -bI"time" solvers.out.4 solvers.out.5 >&2
diff -bI"time" solvers.out.6 solvers.out.7 >&2
diff -bI"time" solvers.out.8 solvers.out.9 >&2
diff -bI"time" solvers.out.10 solvers.out.11 >&2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
"
for i in $FILES
do
  echo "# Output file: $i"
  tail -17 $i | head -8
done > ${TNAME}.out

FILES="\
 ${TNAME}.out.4\
 ${TNAME}.out.5\
 ${TNAME}.out.6\
 ${TNAME}.out.7\
 ${TNAME}.out.12\
"
for i in $FILES
do
  echo "# Output file: $i"
  tail -4 $i
done >> ${TNAME}.out

FILES="\
 ${TNAME}.out.8\
 ${TNAME}.out.9\
 ${TNAME}.out.10\
 ${TNAME}.out.11\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -22 $i | head -13
done >> ${TNAME}.out

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

# rm -f ${TNAME}.testdata*
