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
 ${TNAME}.out.900\
 ${TNAME}.out.901\
 ${TNAME}.out.902\
 ${TNAME}.out.903\
 ${TNAME}.out.904\
 ${TNAME}.out.905\
 ${TNAME}.out.906\
 ${TNAME}.out.910\
 ${TNAME}.out.911\
 ${TNAME}.out.912\
 ${TNAME}.out.913\
 ${TNAME}.out.914\
 ${TNAME}.out.915\
 ${TNAME}.out.916\
 ${TNAME}.out.917\
 ${TNAME}.out.918\
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

FILES="\
 ${TNAME}.out.300\
 ${TNAME}.out.301\
 ${TNAME}.out.302\
 ${TNAME}.out.303\
 ${TNAME}.out.304\
 ${TNAME}.out.305\
 ${TNAME}.out.306\
 ${TNAME}.out.307\
 ${TNAME}.out.308\
 ${TNAME}.out.309\
 ${TNAME}.out.310\
 ${TNAME}.out.311\
 ${TNAME}.out.312\
 ${TNAME}.out.313\
 ${TNAME}.out.314\
 ${TNAME}.out.315\
 ${TNAME}.out.316\
 ${TNAME}.out.317\
 ${TNAME}.out.318\
 ${TNAME}.out.319\
 ${TNAME}.out.320\
 ${TNAME}.out.321\
 ${TNAME}.out.322\
 ${TNAME}.out.323\
 ${TNAME}.out.324\
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
