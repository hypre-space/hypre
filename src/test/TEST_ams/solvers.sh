#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

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
done > ${TNAME}.out.a

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Convergence" ${TNAME}.out.a | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

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
done > ${TNAME}.out.b

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out.b | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

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
done > ${TNAME}.out.c

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "iterations" ${TNAME}.out.c | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

# put all of the output files together
cat ${TNAME}.out.[a-z] > ${TNAME}.out

#=============================================================================
# remove temporary files
#=============================================================================

# rm -f ${TNAME}.testdata*
