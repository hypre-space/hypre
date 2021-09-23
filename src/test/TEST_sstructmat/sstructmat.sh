#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

TNAME=`basename $0 .sh`

#=============================================================================
#=============================================================================

FILESmatmat="\
 ${TNAME}.out.2quadsRAPx\
 ${TNAME}.out.2quadsRAPx.FF\
 ${TNAME}.out.2quadsRAPx.CC\
 ${TNAME}.out.2quadsRAPy\
 ${TNAME}.out.2quadsRAPxy\
 ${TNAME}.out.2quadsRAPx.2procs\
 ${TNAME}.out.2quadsRAPx.FF.2procs\
 ${TNAME}.out.2quadsRAPx.CC.2procs\
 ${TNAME}.out.2quadsRAPy.2procs\
 ${TNAME}.out.2quadsRAPxy.2procs\
 ${TNAME}.out.3quadsRAPx.idle\
 ${TNAME}.out.3quadsRAPx.idle.3procs\
 ${TNAME}.out.2cubesRAPx\
 ${TNAME}.out.2cubesRAPx.2procs\
 ${TNAME}.out.2cubesRAPy\
 ${TNAME}.out.2cubesRAPy.2procs\
 ${TNAME}.out.2cubesRAPz\
 ${TNAME}.out.2cubesRAPz.2procs\
 ${TNAME}.out.2cubesRAPxy\
 ${TNAME}.out.2cubesRAPxy.2procs\
 ${TNAME}.out.2cubesRAPxz\
 ${TNAME}.out.2cubesRAPxz.2procs\
"
#=============================================================================
# Find failed tests
#=============================================================================

# Matmat tests
for i in $FILESmatmat
do
    FAILED=`grep "[test_SStructMatmult]: failed" $i | wc -l`
    if [ "$FAILED" -gt "0" ]
    then
        echo "[test_SStructMatmult]: failed at $i" >&2
    fi
done
