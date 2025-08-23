#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`
RTOL=$1
ATOL=$2

#=============================================================================
# Check read-from-file runs against driver-only runs
#=============================================================================

tail -3 ${TNAME}.out.1P > ${TNAME}.testdata
tail -3 ${TNAME}.out.1RA > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.1RAb > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.1RAbx > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.3P > ${TNAME}.testdata
tail -3 ${TNAME}.out.3RA > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.3RAb > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.3RAbx > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.4P > ${TNAME}.testdata
tail -3 ${TNAME}.out.4RA > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.4RAb > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
tail -3 ${TNAME}.out.4RAbx > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.101P > ${TNAME}.testdata
tail -3 ${TNAME}.out.101RAb > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.103P > ${TNAME}.testdata
tail -3 ${TNAME}.out.103RAb > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.104P > ${TNAME}.testdata
tail -3 ${TNAME}.out.104RAb > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.1P\
 ${TNAME}.out.1RA\
 ${TNAME}.out.1RAb\
 ${TNAME}.out.1RAbx\
 ${TNAME}.out.3P\
 ${TNAME}.out.3RA\
 ${TNAME}.out.3RAb\
 ${TNAME}.out.3RAbx\
 ${TNAME}.out.4P\
 ${TNAME}.out.4RA\
 ${TNAME}.out.4RAb\
 ${TNAME}.out.4RAbx\
 ${TNAME}.out.101P\
 ${TNAME}.out.101RAb\
 ${TNAME}.out.103P\
 ${TNAME}.out.103RAb\
 ${TNAME}.out.104P\
 ${TNAME}.out.104RAb\
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

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
