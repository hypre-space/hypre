#!/bin/sh 
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


TNAME=`basename $0 .sh`

#=============================================================================
# sstruct_fac: Tests the sstruct_fac solver
#=============================================================================

tail -3 ${TNAME}.out.0 > ${TNAME}.testdata
tail -3 ${TNAME}.out.1 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

# tail -3 ${TNAME}.out.2 > ${TNAME}.testdata
# tail -3 ${TNAME}.out.3 > ${TNAME}.testdata.temp
# diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.4 > ${TNAME}.testdata
tail -3 ${TNAME}.out.5 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.6 > ${TNAME}.testdata
tail -3 ${TNAME}.out.7 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

#tail -3 ${TNAME}.out.8 > ${TNAME}.testdata
#tail -3 ${TNAME}.out.9 > ${TNAME}.testdata.temp
#diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================

tail -3 ${TNAME}.out.10 > ${TNAME}.testdata
tail -3 ${TNAME}.out.11 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.4\
 ${TNAME}.out.5\
 ${TNAME}.out.6\
 ${TNAME}.out.7\
 ${TNAME}.out.10\
 ${TNAME}.out.11\
"
#  ${TNAME}.out.2\
#  ${TNAME}.out.3\
# ${TNAME}.out.8\
# ${TNAME}.out.9\
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
#   remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
