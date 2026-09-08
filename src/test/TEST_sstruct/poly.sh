#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# Compare MatPrec-Jacobi PCG to Jacobi PCG
#=============================================================================

tail -3 ${TNAME}.out.1m > ${TNAME}.testdata.m
tail -3 ${TNAME}.out.1j > ${TNAME}.testdata.j
diff ${TNAME}.testdata.m ${TNAME}.testdata.j >&2

tail -3 ${TNAME}.out.2m > ${TNAME}.testdata.m
tail -3 ${TNAME}.out.2j > ${TNAME}.testdata.j
diff ${TNAME}.testdata.m ${TNAME}.testdata.j >&2

tail -3 ${TNAME}.out.3m > ${TNAME}.testdata.m
tail -3 ${TNAME}.out.3j > ${TNAME}.testdata.j
diff ${TNAME}.testdata.m ${TNAME}.testdata.j >&2

tail -3 ${TNAME}.out.4m > ${TNAME}.testdata.m
tail -3 ${TNAME}.out.4j > ${TNAME}.testdata.j
diff ${TNAME}.testdata.m ${TNAME}.testdata.j >&2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.1m\
 ${TNAME}.out.1j\
 ${TNAME}.out.2m\
 ${TNAME}.out.2j\
 ${TNAME}.out.3m\
 ${TNAME}.out.3j\
 ${TNAME}.out.4m\
 ${TNAME}.out.4j\
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
