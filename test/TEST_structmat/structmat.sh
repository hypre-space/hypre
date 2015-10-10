#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

TNAME=`basename $0 .sh`

#=============================================================================
#=============================================================================

FILES2D="\
 ${TNAME}.out.2D0.x\
 ${TNAME}.out.2D1.x\
 ${TNAME}.out.2D2.x\
 ${TNAME}.out.2D3.x\
 ${TNAME}.out.2D0.y\
 ${TNAME}.out.2D1.y\
 ${TNAME}.out.2D2.y\
 ${TNAME}.out.2D3.y\
 ${TNAME}.out.2D0.x4\
 ${TNAME}.out.2D1.x4\
 ${TNAME}.out.2D2.x4\
 ${TNAME}.out.2D3.x4\
 ${TNAME}.out.2D0.y4\
 ${TNAME}.out.2D1.y4\
 ${TNAME}.out.2D2.y4\
 ${TNAME}.out.2D3.y4\
 ${TNAME}.out.2D1.xidle\
 ${TNAME}.out.2D1.xidle4\
 ${TNAME}.out.2D1.yidle\
 ${TNAME}.out.2D1.yidle5\
"
#=============================================================================
# Diff the output with the saved output
#=============================================================================

for i in $FILES2D
do
   for j in ${i}.matmat.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

#=============================================================================
# remove temporary files
#=============================================================================

