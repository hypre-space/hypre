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
 ${TNAME}.out.0P\
 ${TNAME}.out.0R\
 ${TNAME}.out.1P\
 ${TNAME}.out.1R\
 ${TNAME}.out.2P\
 ${TNAME}.out.2R\
 ${TNAME}.out.3P\
 ${TNAME}.out.3R\
 ${TNAME}.out.4P\
 ${TNAME}.out.4R\
 ${TNAME}.out.5P\
 ${TNAME}.out.5R\
 ${TNAME}.out.6P\
 ${TNAME}.out.6R\
 ${TNAME}.out.7P\
 ${TNAME}.out.7R\
 ${TNAME}.out.100P\
 ${TNAME}.out.100R\
 ${TNAME}.out.101P\
 ${TNAME}.out.101R\
 ${TNAME}.out.102P\
 ${TNAME}.out.102R\
 ${TNAME}.out.103P\
 ${TNAME}.out.103R\
 ${TNAME}.out.104P\
 ${TNAME}.out.104R\
 ${TNAME}.out.105P\
 ${TNAME}.out.105R\
 ${TNAME}.out.106P\
 ${TNAME}.out.106R\
 ${TNAME}.out.107P\
 ${TNAME}.out.107R\
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
