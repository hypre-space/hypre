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
# struct: Test parallel and blocking by diffing against base 3d case
#
#   for each test, save the results for comparison with the baseline case
#=============================================================================

tail -3 smgbase3d.out.0 > smgbase3d.testdata
tail -3 smgbase3d.out.1 > smgbase3d.testdata.temp
diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2

cat smgbase3d.testdata > smgbase3d.tests
cat smgbase3d.testdata.temp >> smgbase3d.tests
#=============================================================================

tail -3 smgbase3d.out.2 > smgbase3d.testdata.temp
diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2

cat smgbase3d.testdata.temp >> smgbase3d.tests
#=============================================================================

tail -3 smgbase3d.out.3 > smgbase3d.testdata.temp
diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2

cat smgbase3d.testdata.temp >> smgbase3d.tests
#=============================================================================

tail -3 smgbase3d.out.4 > smgbase3d.testdata.temp
diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2

cat smgbase3d.testdata.temp >> smgbase3d.tests
#=============================================================================

# tail -3 smgbase3d.out.5 > smgbase3d.testdata.temp
# diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2
#
# cat smgbase3d.testdata.temp >> smgbase3d.tests
#=============================================================================
# 
# tail -3 smgbase3d.out.6 > smgbase3d.testdata.temp
# diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2
# cat smgbase3d.testdata.temp >> smgbase3d.tests
#=============================================================================

tail -3 smgbase3d.out.7 > smgbase3d.testdata.temp
diff -bI"time" smgbase3d.testdata smgbase3d.testdata.temp >&2

cat smgbase3d.testdata.temp >> smgbase3d.tests

#=============================================================================
#   compare with the baseline case
#=============================================================================
diff -bI"time" smgbase3d.saved smgbase3d.tests >&2

#=============================================================================
#   remove temporary files
#=============================================================================
rm -f smgbase3d.testdata* smgbase3d.tests
