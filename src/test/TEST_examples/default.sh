#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
 ${TNAME}.out.5\
 ${TNAME}.out.5f\
 ${TNAME}.out.6\
 ${TNAME}.out.7\
 ${TNAME}.out.8\
 ${TNAME}.out.9\
 ${TNAME}.out.12\
 ${TNAME}.out.12f\
 ${TNAME}.out.13\
 ${TNAME}.out.14\
 ${TNAME}.out.15\
"
# ${TNAME}.out.11\

# Need to avoid output lines about "no global partition"
for i in $FILES
do
  echo "# Output file: $i"
  tail -5 $i
done > ${TNAME}.out

# Make sure that the output file is reasonable
RUNCOUNT=9
OUTCOUNT=`grep "Iterations" ${TNAME}.out | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

