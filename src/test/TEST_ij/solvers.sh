#!/bin/ksh
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
# $Revision: 1.12 $
#EHEADER**********************************************************************

TNAME=`basename $0 .sh`

#=============================================================================
# compare with baseline case
#=============================================================================

FILES="\
 ${TNAME}.out.0\
 ${TNAME}.out.1\
 ${TNAME}.out.2\
 ${TNAME}.out.3\
 ${TNAME}.out.4\
 ${TNAME}.out.5\
 ${TNAME}.out.6\
 ${TNAME}.out.7\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
done > ${TNAME}.out

FILES="\
 ${TNAME}.out.8\
 ${TNAME}.out.9\
 ${TNAME}.out.10\
 ${TNAME}.out.11\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -5 $i
done >> ${TNAME}.out

FILES="\
 ${TNAME}.out.1.lobpcg\
 ${TNAME}.out.2.lobpcg\
 ${TNAME}.out.8.lobpcg\
 ${TNAME}.out.12.lobpcg\
 ${TNAME}.out.43.lobpcg\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -3 $i
  echo "# Output file: $i.1"
  tail -13 $i.1 | head -3
  echo "# Output file: $i.5"
  tail -21 $i.5 | head -11
done >> ${TNAME}.out

FILES="\
 ${TNAME}.out.sysh\
 ${TNAME}.out.sysn\
 ${TNAME}.out.sysu\
"

for i in $FILES
do
  echo "# Output file: $i"
  tail -17 $i | head -6
done >> ${TNAME}.out

if [ -z $HYPRE_NO_SAVED ]; then
   diff -U3 -bI"time" ${TNAME}.saved ${TNAME}.out >&2
fi

#=============================================================================
# remove temporary files
#=============================================================================

# rm -f ${TNAME}.testdata*
