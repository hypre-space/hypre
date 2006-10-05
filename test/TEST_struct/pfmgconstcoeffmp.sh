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
# struct: Test PFMG constant coefficient runs on 1 and 2 processors
# by diffing against each other.
#
#   for each test, save the results for comparison with baseline case
#=============================================================================

tail -3 pfmgconstcoeffmp.out.0 > pfmgconstcoeffmp.testdata
tail -3 pfmgconstcoeffmp.out.1 > pfmgconstcoeffmp.testdata.temp
diff pfmgconstcoeffmp.testdata pfmgconstcoeffmp.testdata.temp  >&2

cat pfmgconstcoeffmp.testdata > pfmgconstcoeffmp.tests
cat pfmgconstcoeffmp.testdata.temp >> pfmgconstcoeffmp.tests

#=============================================================================
#    compare with baseline case 
#=============================================================================
diff pfmgconstcoeffmp.saved pfmgconstcoeffmp.tests >&2

#=============================================================================
#  remove temporary files
#=============================================================================
rm -f pfmgconstcoeffmp.testdata* pfmgconstcoeffmp.tests
