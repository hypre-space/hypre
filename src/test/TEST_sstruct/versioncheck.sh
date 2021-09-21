#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`

#=============================================================================
# Check the HYPRE_DEVELOP variables
#=============================================================================

grep "Using HYPRE_DEVELOP_STRING" ${TNAME}.out.1 > ${TNAME}.testdata

if [ -d ../../../.git ]; then
  DEVSTRING=`git describe --match 'v*' --long --abbrev=9`
  DEVNUMBER=`echo $DEVSTRING | awk -F- '{print $2}'`
  DEVBRANCH=`git rev-parse --abbrev-ref HEAD`
  if [ -n "$DEVBRANCH" ]; then
    echo "Using HYPRE_DEVELOP_STRING: $DEVSTRING (not main development branch)" \
     > ${TNAME}.testdatacheck
  else
    echo "Using HYPRE_DEVELOP_STRING: $DEVSTRING (main development branch $DEVBRANCH)" \
     > ${TNAME}.testdatacheck
  fi
fi
diff ${TNAME}.testdata ${TNAME}.testdatacheck >&2

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
