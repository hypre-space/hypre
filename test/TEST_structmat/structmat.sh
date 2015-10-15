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

FILESmatmat="\
 ${TNAME}.out.2D0.x\
 ${TNAME}.out.2D1.x\
 ${TNAME}.out.2D2.x\
 ${TNAME}.out.2D3.x\
 ${TNAME}.out.2D0.y\
 ${TNAME}.out.2D1.y\
 ${TNAME}.out.2D2.y\
 ${TNAME}.out.2D3.y\
 ${TNAME}.out.2D0.x.4\
 ${TNAME}.out.2D1.x.4\
 ${TNAME}.out.2D2.x.4\
 ${TNAME}.out.2D3.x.4\
 ${TNAME}.out.2D0.y.4\
 ${TNAME}.out.2D1.y.4\
 ${TNAME}.out.2D2.y.4\
 ${TNAME}.out.2D3.y.4\
 ${TNAME}.out.2D1.xidle\
 ${TNAME}.out.2D1.xidle.4\
 ${TNAME}.out.2D1.yidle\
 ${TNAME}.out.2D1.yidle.5\
"
FILESmatvec="\
 ${TNAME}.out.2D0.mv0\
 ${TNAME}.out.2D0.mv1\
 ${TNAME}.out.2D0.mv2\
 ${TNAME}.out.2D0.mv3\
 ${TNAME}.out.2D0.mv4\
 ${TNAME}.out.2D1.mv0\
 ${TNAME}.out.2D1.mv1\
 ${TNAME}.out.2D1.mv2\
 ${TNAME}.out.2D1.mv3\
 ${TNAME}.out.2D1.mv4\
 ${TNAME}.out.2D3.mv0\
 ${TNAME}.out.2D0.mv0.4\
 ${TNAME}.out.2D0.mv1.4\
 ${TNAME}.out.2D0.mv2.4\
 ${TNAME}.out.2D0.mv3.4\
 ${TNAME}.out.2D0.mv4.4\
 ${TNAME}.out.2D1.mv0.4\
 ${TNAME}.out.2D1.mv1.4\
 ${TNAME}.out.2D1.mv2.4\
 ${TNAME}.out.2D1.mv3.4\
 ${TNAME}.out.2D1.mv4.4\
 ${TNAME}.out.2D3.mv0.4\
"
#=============================================================================
# Diff the output with the saved output
#=============================================================================

for i in $FILESmatmat
do
   for j in ${i}.matmat.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

for i in $FILESmatvec
do
   for j in ${i}.matvec.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

#=============================================================================
# remove temporary files
#=============================================================================

