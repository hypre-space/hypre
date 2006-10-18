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
# no test comparison for now. Just a holder file with fake tests. Hard to
# develop tests because of the coarsening scheme.
#    for each test, save results for comparison with baseline case
#=============================================================================

tail -3 maxwell.out.0 > maxwell.testdata
tail -3 maxwell.out.0 > maxwell.testdata.temp
diff -bI"time" maxwell.testdata maxwell.testdata.temp >&2

cat maxwell.testdata > maxwell.tests
cat maxwell.testdata.temp >> maxwell.tests
#=============================================================================
tail -3 maxwell.out.1 > maxwell.testdata
tail -3 maxwell.out.1 > maxwell.testdata.temp
diff -bI"time" maxwell.testdata maxwell.testdata.temp >&2

cat maxwell.testdata >> maxwell.tests
cat maxwell.testdata.temp >> maxwell.tests
#=============================================================================
tail -3 maxwell.out.2 > maxwell.testdata
tail -3 maxwell.out.2 > maxwell.testdata.temp
diff -bI"time" maxwell.testdata maxwell.testdata.temp >&2

cat maxwell.testdata >> maxwell.tests
cat maxwell.testdata.temp >> maxwell.tests

#=============================================================================
#    compare with baseline case
#=============================================================================
diff -bI"time" maxwell.saved maxwell.tests >&2

#=============================================================================
#    remove temporary files
#=============================================================================
rm -f maxwell.testdata* maxwell.tests
