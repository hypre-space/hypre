#!/bin/sh
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
# EXAMPLES: Compare ex*.base files with default.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================

diff -bI"time" ex1.base default.out.1 >&2

diff -bI"time" ex2.base default.out.2 >&2

diff -bI"time" ex3.base default.out.3 >&2

diff -bI"time" ex4.base default.out.4 >&2

diff -bI"time" ex5.base default.out.5 >&2

diff -bI"time" ex6.base default.out.6 >&2

diff -bI"time" ex7.base default.out.7 >&2

diff -bI"time" ex8.base default.out.8 >&2

diff -bI"time" ex9.base default.out.9 >&2
