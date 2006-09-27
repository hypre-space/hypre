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
# struct: Test Periodic SMG base 3d case (periodic in x), test parallel and blocking,
# and run a full periodic case. Note: driver sets up right hand size for
# full periodic case that satifies compatibility condition, it (the rhs)
# is dependent on blocking and parallel partitioning. Thus results will
# differ with number of blocks and processors.
#=============================================================================

tail -3 psmgtest3d.out.0 > psmgtest3d.testdata

tail -3 psmgtest3d.out.1 > psmgtest3d.testdata.temp
diff psmgtest3d.testdata psmgtest3d.testdata.temp >&2

rm -f psmgtest3d.testdata psmgtest3d.testdata.temp

#=============================================================================
# Concatenate *.out.* files then compare with baseline case
#=============================================================================
cat psmgtest3d.out.* > psmgtest3d.out
diff psmgtest3d.out psmgtest3d.saved >&2
