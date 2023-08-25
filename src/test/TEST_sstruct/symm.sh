#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

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
