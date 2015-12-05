#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision: 1.7 $
#EHEADER**********************************************************************

TNAME=`basename $0 .sh`

#=============================================================================
# sstruct: Test various empty proc problems
#=============================================================================

TNUMS="\
 00 01 02 03 04 05 06    08 09\
 10 11 12    14 15 16 17 18   \
 20 21 22 23 24 25 26 27 28 29\
 30 31 32 33 34 35 36 37 38   \
"

for i in $TNUMS
do
  tail -3 ${TNAME}.out.${i}  > ${TNAME}.testdata
  tail -3 ${TNAME}.out.1${i} > ${TNAME}.testdata.temp
  diff ${TNAME}.testdata ${TNAME}.testdata.temp >&2
done

#=============================================================================
# compare with baseline case
#=============================================================================

for i in $TNUMS
do
  echo "# Output file: ${TNAME}.out.${i}"
  tail -3 ${TNAME}.out.${i}
done > ${TNAME}.out

if [ -z $HYPRE_NO_SAVED ]; then
   diff -U3 -bI"time" ${TNAME}.saved ${TNAME}.out >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

rm -f ${TNAME}.testdata*
