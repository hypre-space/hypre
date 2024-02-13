#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
#=============================================================================

FILES3D0="\
 ${TNAME}.out.3Dx.1\
 ${TNAME}.out.3Dx.2\
 ${TNAME}.out.3Dx.3\
 ${TNAME}.out.3Dy.1\
 ${TNAME}.out.3Dy.2\
 ${TNAME}.out.3Dy.3\
 ${TNAME}.out.3Dz.1\
 ${TNAME}.out.3Dz.2\
 ${TNAME}.out.3Dz.3\
"

FILES3D="\
 ${TNAME}.out.3Dx.5\
 ${TNAME}.out.3Dx.6\
 ${TNAME}.out.3Dy.5\
 ${TNAME}.out.3Dy.6\
 ${TNAME}.out.3Dz.5\
 ${TNAME}.out.3Dz.6\
"

FILES2D0="\
 ${TNAME}.out.2Dx.1\
 ${TNAME}.out.2Dx.2\
 ${TNAME}.out.2Dx.3\
 ${TNAME}.out.2Dy.1\
 ${TNAME}.out.2Dy.2\
 ${TNAME}.out.2Dy.3\
"

FILES2D="\
 ${TNAME}.out.2Dx.5\
 ${TNAME}.out.2Dx.6\
 ${TNAME}.out.2Dy.5\
 ${TNAME}.out.2Dy.6\
"

FILES1D0="\
 ${TNAME}.out.1Dx.1\
 ${TNAME}.out.1Dx.2\
"

#=============================================================================
# Check that the zero residual files are all zero
#=============================================================================

tail -2 ${TNAME}.out.3Dx.1 > ${TNAME}.testdata
for i in $FILES3D0 $FILES2D0 $FILES1D0
do
  tail -2 $i > ${TNAME}.testdata.temp
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

#=============================================================================
# Check that the nonzero residual files are all the same
#=============================================================================

tail -2 ${TNAME}.out.3Dx.5 > ${TNAME}.testdata
for i in $FILES3D
do
  tail -2 $i > ${TNAME}.testdata.temp
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

tail -2 ${TNAME}.out.2Dx.5 > ${TNAME}.testdata
for i in $FILES2D
do
  tail -2 $i > ${TNAME}.testdata.temp
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

#=============================================================================
# compare with baseline case
#=============================================================================

for i in $FILES3D0 $FILES2D0 $FILES1D0 $FILES3D $FILES2D
do
  echo "# Output file: $i"
  tail -2 $i
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

rm -f ${TNAME}.testdata*
