#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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
 ${TNAME}.out.2D3.x.RA\
 ${TNAME}.out.2D3.x.AP\
"
FILESmatmatsym="\
 ${TNAME}.out.2D0.sym.x\
 ${TNAME}.out.2D1.sym.x\
 ${TNAME}.out.2D2.sym.x\
 ${TNAME}.out.2D3.sym.x\
 ${TNAME}.out.2D0.sym.x.4\
 ${TNAME}.out.2D1.sym.x.4\
 ${TNAME}.out.2D2.sym.x.4\
 ${TNAME}.out.2D3.sym.x.4\
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
FILESmatvecsym="\
 ${TNAME}.out.2D0.mv0.4\
 ${TNAME}.out.2D1.mv0.4\
 ${TNAME}.out.2D3.mv0.4\
"
FILESab="\
 ${TNAME}.out.ab0.mv0\
 ${TNAME}.out.ab0.mv1\
 ${TNAME}.out.ab0.mv2\
 ${TNAME}.out.ab1.mv0\
 ${TNAME}.out.ab1.mv1\
 ${TNAME}.out.ab1.mv2\
 ${TNAME}.out.ab2.mv0\
 ${TNAME}.out.ab2.mv1\
 ${TNAME}.out.ab2.mv2\
"

#=============================================================================
# Diff the output with the saved output
#=============================================================================

# Matmat tests
for i in $FILESmatmat
do
   for j in ${i}.matmat.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

# Matvec tests
for i in $FILESmatvec
do
   for j in ${i}.matvec.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

# MatvecT tests
for i in $FILESmatvec
do
   for j in ${i}.matvec.*
   do
      matvecT=`echo $j | sed 's/mv/mvT/' | sed 's/matvec/matvecT/'`
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $matvecT $saved >&2
   done
done

# Check the x==y and beta=0 tests (compare to appropriate saved files)
for i in ${TNAME}.out.*zzz.matvec*
do
   zfile=`echo $i | sed 's/.zzz//'`
   diff -U3 -bI"time" $i $zfile >&2
done

# Matvec alpha/beta tests
for i in $FILESab
do
   for j in ${i}.matvec.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

# Symmetric Matmat tests
for i in $FILESmatmatsym
do
   for j in ${i}.matmat.*
   do
      saved=`echo $j | sed 's/.out/.saved/'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

# Symmetric Matvec tests
for i in $FILESmatvec
do
   for j in ${i}.matvec.*
   do
      saved=`echo $j | sed 's/.out/.saved/' | sed 's/.sym//'`
      diff -U3 -bI"time" $j $saved >&2
   done
done

# Symmetric MatvecT tests
for i in $FILESmatvec
do
   for j in ${i}.matvec.*
   do
      matvecT=`echo $j | sed 's/mv/mvT/' | sed 's/matvec/matvecT/'`
      saved=`echo $j | sed 's/.out/.saved/' | sed 's/.sym//'`
      diff -U3 -bI"time" $matvecT $saved >&2
   done
done

#=============================================================================
# remove temporary files
#=============================================================================
