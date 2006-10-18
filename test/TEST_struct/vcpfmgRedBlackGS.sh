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
#   for each test, save the results for comparison with the baseline case
#=============================================================================

#=============================================================================
# struct: Test parallel and blocking by diff -bI"time"ing against base "true" 2d case
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.0 > vcpfmgRedBlackGS.testdata
tail -3 vcpfmgRedBlackGS.out.1 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata > vcpfmgRedBlackGS.tests
cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.2 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.3 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.4 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.5 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

#=============================================================================
# struct: symmetric GS
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.6 > vcpfmgRedBlackGS.testdata
tail -3 vcpfmgRedBlackGS.out.7 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata >> vcpfmgRedBlackGS.tests
cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.8 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.9 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.10 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.11 > vcpfmgRedBlackGS.testdata.temp
diff -bI"time" vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

cat vcpfmgRedBlackGS.testdata.temp >> vcpfmgRedBlackGS.tests

#=============================================================================
#    compare with the baseline case
#=============================================================================
diff -bI"time" vcpfmgRedBlackGS.saved vcpfmgRedBlackGS.tests >&2

#=============================================================================
#    remove temporary files
#=============================================================================
rm -f vcpfmgRedBlackGS.testdata* vcpfmgRedBlackGS.tests
