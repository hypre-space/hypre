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
# $Revision: 1.3 $
#EHEADER**********************************************************************

#=============================================================================
# struct: Test parallel and blocking by diffing against base "true" 2d case
#=============================================================================

tail -3 vdpfmgRedBlackGS.out.0 > vdpfmgRedBlackGS.testdata

tail -3 vdpfmgRedBlackGS.out.1 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.2 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.3 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.4 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.5 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

#=============================================================================
# struct: symmetric GS
#=============================================================================

tail -3 vdpfmgRedBlackGS.out.6 > vdpfmgRedBlackGS.testdata

tail -3 vdpfmgRedBlackGS.out.7 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.8 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.9 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.10 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.11 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

rm -f vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp
