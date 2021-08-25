#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
 ${TNAME}.out.5\
 ${TNAME}.out.6\
 ${TNAME}.out.7\
 ${TNAME}.out.8\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out

for i in $FILES
do
  echo "# Output file: $i"
  setup_time=$(grep -A 1 "PCG Setup" $i | tail -n 1)
  echo "PCG Setup"${setup_time}
  solve_time=$(grep -A 1 "PCG Solve" $i | tail -n 1)
  echo "PCG Solve"${solve_time}
done > ${TNAME}.perf.out

# Make sure that the output file is reasonable
RUNCOUNT=`echo $FILES | wc -w`
OUTCOUNT=`grep "Iterations" ${TNAME}.out | wc -l`
if [ "$OUTCOUNT" != "$RUNCOUNT" ]; then
   echo "Incorrect number of runs in ${TNAME}.out" >&2
fi

# Check performance
HOST=`hostname`
case $HOST in
   lassen*)
      SavePerfExt="saved.lassen"
      rtol=0.15
      ;;
   *) SavePerfExt=""
      ;;
esac

if [ -n "$SavePerfExt" ]; then
   ../runcheck.sh $TNAME.perf.out $TNAME.perf.$SavePerfExt $rtol >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*

