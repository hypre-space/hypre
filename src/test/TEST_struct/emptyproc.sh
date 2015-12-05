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
# $Revision: 1.1 $
#EHEADER**********************************************************************

TNAME=`basename $0 .sh`

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
