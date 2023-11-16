#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`

tail -3 ${TNAME}.out.1.a | head -2 > ${TNAME}.testdata
tail -3 ${TNAME}.out.1.b | head -2 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.2.a | head -2 > ${TNAME}.testdata
tail -3 ${TNAME}.out.2.b | head -2 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.3.a | head -2 > ${TNAME}.testdata
tail -3 ${TNAME}.out.3.b | head -2 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

tail -3 ${TNAME}.out.4.a | head -2 > ${TNAME}.testdata
tail -3 ${TNAME}.out.4.b | head -2 > ${TNAME}.testdata.temp
diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2

#=============================================================================
# Compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.1\
 ${TNAME}.out.100\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
