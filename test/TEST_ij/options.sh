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
#  for each test save the results for comparison with the baseline case
#=============================================================================
tail -14 options.out.default > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp > options.tests
#=============================================================================
tail -15 options.out.gen.1 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -15 options.out.gen.2 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -16 options.out.itr.100 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -16 options.out.itr.2 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.no_orthchk > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -15 options.out.orthchk > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.pcgitr.0 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.pcgitr.1 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.pcgitr.2 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.pcgtol.01 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.pcgtol.05 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.seed > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.seed.repeat > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.solver.none > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -40 options.out.verb.0 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -14 options.out.verb.1 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -11 options.out.verb.2 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests
#=============================================================================
tail -16 options.out.vrand.2 > options.testdata
head options.testdata > options.testdata.tmp

cat options.testdata.tmp >> options.tests

#=============================================================================
#  compare with the baseline case
#=============================================================================
diff -bI"time" options.saved options.tests >&2

#=============================================================================
#  remove temporary files
#=============================================================================
rm -f options.testdata* options.tests
