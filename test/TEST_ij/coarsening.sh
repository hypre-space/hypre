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
# $Revision$
#EHEADER**********************************************************************

#=============================================================================
#  for each test save the results for comparison with baseline case
#=============================================================================
tail -18 coarsening.out.0 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 > coarsening.tests
#=============================================================================
tail -18 coarsening.out.1 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests
#=============================================================================
tail -18 coarsening.out.2 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests
#=============================================================================
tail -18 coarsening.out.3 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests
#=============================================================================
tail -18 coarsening.out.4 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests
#=============================================================================
tail -18 coarsening.out.5 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests
#=============================================================================
tail -4 coarsening.out.6 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests
#=============================================================================
tail -4 coarsening.out.7 > coarsening.testdata
head coarsening.testdata > coarsening.testdata.tmp0

cat coarsening.testdata.tmp0 >> coarsening.tests

#=============================================================================
#   compare with baseline case
#=============================================================================
diff -bI"time" coarsening.saved coarsening.tests >&2

#=============================================================================
#   remove temporary files
#=============================================================================
rm -f coarsening.testdata coarsening.testdata.tmp0 coarsening.tests
