#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# IJ: Run multiplicative and mult_additive cycle and compare results 
#                    should be the same
#=============================================================================

tail -17 ${TNAME}.out.109 | head -6 > ${TNAME}.testdata

#=============================================================================

tail -17 ${TNAME}.out.110 | head -6 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

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
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
 ${TNAME}.out.5\
 ${TNAME}.out.6\
 ${TNAME}.out.7\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out

FILES="\
 ${TNAME}.out.8\
 ${TNAME}.out.9\
 ${TNAME}.out.10\
 ${TNAME}.out.11\
 ${TNAME}.out.12\
 ${TNAME}.out.13\
 ${TNAME}.out.14\
 ${TNAME}.out.15\
 ${TNAME}.out.16\
 ${TNAME}.out.17\
 ${TNAME}.out.18\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -5 $i
done >> ${TNAME}.out

FILES="\
 ${TNAME}.out.sysh\
 ${TNAME}.out.sysn\
 ${TNAME}.out.sysu\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -21 $i | head -6
done >> ${TNAME}.out

FILES="\
 ${TNAME}.out.101\
 ${TNAME}.out.102\
 ${TNAME}.out.103\
 ${TNAME}.out.104\
 ${TNAME}.out.105\
 ${TNAME}.out.106\
 ${TNAME}.out.107\
 ${TNAME}.out.108\
 ${TNAME}.out.109\
 ${TNAME}.out.110\
 ${TNAME}.out.111\
 ${TNAME}.out.112\
 ${TNAME}.out.113\
 ${TNAME}.out.114\
 ${TNAME}.out.115\
 ${TNAME}.out.116\
 ${TNAME}.out.117\
 ${TNAME}.out.118\
 ${TNAME}.out.119\
 ${TNAME}.out.120\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done >> ${TNAME}.out

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
done >> ${TNAME}.out

# Make sure that the output files are reasonable
CHECK_LINE="Complexity"
OUT_COUNT=`grep "$CHECK_LINE" ${TNAME}.out | wc -l`
SAVED_COUNT=`grep "$CHECK_LINE" ${TNAME}.saved | wc -l`
if [ "$OUT_COUNT" != "$SAVED_COUNT" ]; then
   echo "Incorrect number of \"$CHECK_LINE\" lines in ${TNAME}.out" >&2
fi

if [ -z $HYPRE_NO_SAVED ]; then
   (../runcheck.sh ${TNAME}.out ${TNAME}.saved $RTOL $ATOL) >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
rm -r ${TNAME}.mgr_testdata*
