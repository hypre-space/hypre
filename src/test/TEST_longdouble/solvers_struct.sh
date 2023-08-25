#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
CONVTOL=$1

# Set default check tolerance
if [ x$CONVTOL = "x" ];
then
    CONVTOL=0.0
fi
#echo "tol = $CONVTOL"
#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out.a

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out.a | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

FILES="\
 ${TNAME}.out.10.lobpcg\
 ${TNAME}.out.11.lobpcg\
 ${TNAME}.out.17.lobpcg\
 ${TNAME}.out.18.lobpcg\
 ${TNAME}.out.19.lobpcg\
"

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
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

# put all of the output files together
cat ${TNAME}.out.[a-z] > ${TNAME}.out

#=============================================================================
# remove temporary files
#=============================================================================

# rm -f ${TNAME}.testdata*
