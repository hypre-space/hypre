#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# IJ: MGR case nlevels < 1 and bsize < 2 should be the same
#                    compare results
#=============================================================================

tail -17 ${TNAME}.out.200 | head -6 > ${TNAME}.mgr_testdata

#=============================================================================

tail -17 ${TNAME}.out.202 | head -6 > ${TNAME}.mgr_testdata.temp
diff ${TNAME}.mgr_testdata ${TNAME}.mgr_testdata.temp >&2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.200\
 ${TNAME}.out.201\
 ${TNAME}.out.202\
 ${TNAME}.out.203\
 ${TNAME}.out.204\
 ${TNAME}.out.205\
 ${TNAME}.out.206\
 ${TNAME}.out.207\
 ${TNAME}.out.208\
 ${TNAME}.out.209\
 ${TNAME}.out.210\
 ${TNAME}.out.211\
 ${TNAME}.out.212\
 ${TNAME}.out.213\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out.e

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out.e | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

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

rm -r ${TNAME}.mgr_testdata*
