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
tail -4 solvers.out.0 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp > solvers.tests
#=============================================================================
tail -4 solvers.out.1 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -6 solvers.out.10 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -6 solvers.out.11 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.12.lobpcg > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -14 solvers.out.12.lobpcg.1 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -22 solvers.out.12.lobpcg.5 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.1.lobpcg > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -14 solvers.out.1.lobpcg.1 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -22 solvers.out.1.lobpcg.5 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.2 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.2.lobpcg > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -14 solvers.out.2.lobpcg.1 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -22 solvers.out.2.lobpcg.5 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.3 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.4 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.43.lobpcg > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -14 solvers.out.43.lobpcg.1 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -22 solvers.out.43.lobpcg.5 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.5 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.6 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.7 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -6 solvers.out.8 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -4 solvers.out.8.lobpcg > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -14 solvers.out.8.lobpcg.1 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -22 solvers.out.8.lobpcg.5 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -6 solvers.out.9 > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -18 solvers.out.sysh > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -18 solvers.out.sysn > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests
#=============================================================================
tail -18 solvers.out.sysu > solvers.testdata
head solvers.testdata > solvers.testdata.tmp

cat solvers.testdata.tmp >> solvers.tests

#=============================================================================
#  compare with the baseline case
#=============================================================================
cp solvers.tests solvers.saved
diff -bI"time" solvers.saved solvers.tests >&2

#=============================================================================
#  remove temporary files
#=============================================================================
rm -f solvers.testdata* solvers.tests
