#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# A. set/get: multi-component vector tests
#=============================================================================

FILES="\
 ${TNAME}.out.A0\
 ${TNAME}.out.A1\
 ${TNAME}.out.A2\
 ${TNAME}.out.A3\
 ${TNAME}.out.A4\
 ${TNAME}.out.A5\
 ${TNAME}.out.A6\
 ${TNAME}.out.A7\
 ${TNAME}.out.A8\
 ${TNAME}.out.A9\
 ${TNAME}.out.A10
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -2 $i
done > ${TNAME}.out.A

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Vector/Multivector error" ${TNAME}.out.A | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out.A" >&2
fi

#=============================================================================
# B. krylov solvers: multi-component vector tests
#=============================================================================

FILES="\
 ${TNAME}.out.B0\
 ${TNAME}.out.B1\
 ${TNAME}.out.B2\
 ${TNAME}.out.B3\
 ${TNAME}.out.B4\
 ${TNAME}.out.B5\
 ${TNAME}.out.B6\
 ${TNAME}.out.B7\
 ${TNAME}.out.B8\
 ${TNAME}.out.B9\
 ${TNAME}.out.B10\
 ${TNAME}.out.B100\
 ${TNAME}.out.B101\
 ${TNAME}.out.B102\
 ${TNAME}.out.B103\
 ${TNAME}.out.B104\
 ${TNAME}.out.B105\
 ${TNAME}.out.B106\
 ${TNAME}.out.B107\
 ${TNAME}.out.B108\
 ${TNAME}.out.B109\
 ${TNAME}.out.B110
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out.B

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out.B | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out.B" >&2
fi

# put all of the output files together
cat ${TNAME}.out.[A-Z] > ${TNAME}.out
