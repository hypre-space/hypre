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
 ${TNAME}.out.10\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -21 $i | head -6
done > ${TNAME}.out

# Remove the "hypre error" lines from '.err' file and append them to '.out'
mv ${TNAME}.err ${TNAME}.tmp
egrep    "hypre error" ${TNAME}.tmp | sort >> ${TNAME}.out
egrep -v "hypre error" ${TNAME}.tmp        >  ${TNAME}.err

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
rm -f ${TNAME}.tmp
