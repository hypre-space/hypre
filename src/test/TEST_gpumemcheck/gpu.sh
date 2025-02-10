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
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
"

for i in $FILES
do
  echo "# Output file: $i"
  grep "LEAK SUMMARY" $i
  grep "ERROR SUMMARY" $i
  echo
done > ${TNAME}.out

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
