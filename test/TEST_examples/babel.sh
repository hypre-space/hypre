#!/bin/sh 
#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer and the GNU Lesser General Public License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************

#=============================================================================
# EXAMPLES: Compare ex*.base files with babel.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================


tail -21 babel.out.5b > babel.test.tmp
head babel.test.tmp > babel.test

tail -21 ex5b.base > babel.base.tmp
head babel.base.tmp > babel.base

diff babel.base babel.test >&2

diff ex5b77.base babel.out.5b77 >&2

rm -f babel.test.tmp babel.test babel.base.tmp babel.base
