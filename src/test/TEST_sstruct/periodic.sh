#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# Check SetNeighborBox for ${TNAME} problems (2D)
#=============================================================================

tail -3 ${TNAME}.out.20 > ${TNAME}.testdata
tail -3 ${TNAME}.out.21 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

#=============================================================================
# Check SetNeighborBox for ${TNAME} problems (3D)
#=============================================================================

tail -3 ${TNAME}.out.30 > ${TNAME}.testdata
tail -3 ${TNAME}.out.31 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

#=============================================================================
# Check SysPFMG for power-of-two and non-power-of-two systems
#=============================================================================

tail -3 ${TNAME}.out.40 > ${TNAME}.testdata
tail -3 ${TNAME}.out.41 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
tail -3 ${TNAME}.out.42 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

tail -3 ${TNAME}.out.50 > ${TNAME}.testdata
tail -3 ${TNAME}.out.51 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
tail -3 ${TNAME}.out.52 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

#=============================================================================
# Check PFMG, SMG, and SysPFMG for problems with period larger than the grid
#=============================================================================

# First check that sstruct and struct are the same here
tail -3 ${TNAME}.out.60 > ${TNAME}.testdata
tail -3 ${TNAME}.out.61 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
tail -3 ${TNAME}.out.62 > ${TNAME}.testdata
tail -3 ${TNAME}.out.63 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

# Also check that PFMG and SysPFMG are the same
tail -3 ${TNAME}.out.66 > ${TNAME}.testdata
tail -3 ${TNAME}.out.67 > ${TNAME}.testdata.temp
(../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2

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
  (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
done

tail -3 ${TNAME}.out.90 > ${TNAME}.testdata
TNUM="91 92 93 94 95"
for i in $TNUM
do
  tail -3 ${TNAME}.out.$i > ${TNAME}.testdata.temp
  (../runcheck.sh ${TNAME}.testdata ${TNAME}.testdata.temp $RTOL $ATOL) >&2
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
 80 81 82 83 84 85\
 90 91 92 93 94 95\
"

for i in $TNUM
do
  FILE="${TNAME}.out.$i"
  echo "# Output file: ${FILE}"
  tail -3 ${FILE}
done > ${TNAME}.out

# Make sure that the output file is reasonable
RUNCOUNT=`echo $TNUM | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
