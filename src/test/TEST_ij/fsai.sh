#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

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
 ${TNAME}.out.100\
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
