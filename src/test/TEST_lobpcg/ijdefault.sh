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
 ${TNAME}.out.vfromfile\
 ${TNAME}.out.vout.1\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -15 $i | head -5
done > ${TNAME}.out.a


FILES="\
 ${TNAME}.out.1.lobpcg\
 ${TNAME}.out.2.lobpcg\
 ${TNAME}.out.12.lobpcg\
"
#${TNAME}.out.8.lobpcg\
#${TNAME}.out.43.lobpcg\

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
  echo "# Output file: $i.1"
  tail -13 $i.1 | head -3
  echo "# Output file: $i.5"
  tail -21 $i.5 | head -11
done > ${TNAME}.out.b

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out.b | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out.b" >&2
fi

# put all of the output files together
cat ${TNAME}.out.[a-z] > ${TNAME}.out

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
rm -f residuals.txt values.txt vectors.[01].*
