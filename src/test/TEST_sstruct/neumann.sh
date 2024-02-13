#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# sstruct: Test various blockings and distributions of default problem
#=============================================================================

tail -3 ${TNAME}.out.0 > ${TNAME}.testdata
tail -3 ${TNAME}.out.2 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.10 > ${TNAME}.testdata
tail -3 ${TNAME}.out.12 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.20 > ${TNAME}.testdata
tail -3 ${TNAME}.out.22 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.1 > ${TNAME}.testdata
tail -3 ${TNAME}.out.3 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.11 > ${TNAME}.testdata
tail -3 ${TNAME}.out.13 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.21 > ${TNAME}.testdata
tail -3 ${TNAME}.out.23 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

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
 ${TNAME}.out.12\
 ${TNAME}.out.13\
 ${TNAME}.out.20\
 ${TNAME}.out.21\
 ${TNAME}.out.22\
 ${TNAME}.out.23\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
