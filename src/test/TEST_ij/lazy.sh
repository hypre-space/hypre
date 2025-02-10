#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`

tail -3 ${TNAME}.out.1.a | head -2 > ${TNAME}.testdata
tail -3 ${TNAME}.out.1.b | head -2 > ${TNAME}.testdata.temp

# Abuse HYPRE_NO_SAVED to skip the following diff with OMP
if [ -z $HYPRE_NO_SAVED ]; then
   diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
fi

tail -3 ${TNAME}.out.2.a | head -2 > ${TNAME}.testdata
tail -3 ${TNAME}.out.2.b | head -2 > ${TNAME}.testdata.temp

# Abuse HYPRE_NO_SAVED to skip the following diff with OMP
if [ -z $HYPRE_NO_SAVED ]; then
   diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
