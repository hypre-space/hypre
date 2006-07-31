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


#notes:
# this script should be called by: IJ_linear_solvers.sh per:
# ../distributed_ls/Euclid/test/euclid.sh
#
# HYPRE_ARCH and MPIRUN are initialized by caller.

#===========================================================================
# goto Euclid's test directory
#===========================================================================

CALLING_DIR=`pwd`
EUCLID_TEST_DIR=${CALLING_DIR}/../distributed_ls/Euclid/test
cd $EUCLID_TEST_DIR

#=================================================================
# single cpu tests
#=================================================================

rm -rf *.out* *temp *database

#
#read options from "database" (the default configuration filename)
#
cp input/level.3 ./database
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData eu.temp
diff eu.temp  output/test3.out >&2
rm -f database eu.temp

#
#specify a different configuration filename, and read options therefrom
#
cp input/level.3 .
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -db_filename level.3 -printTestData eu.temp
diff eu.temp output/test3.out >&2
rm -f level.3 eu.temp

#
#ensure command line options override options in the configuration file
#
cp input/level.3  ./database
$MPIRUN -np 1 $DRIVER -solver 43 -laplacian -printTestData eu.temp
diff eu.temp output/test2.out >&2
rm -f database eu.temp


#=================================================================
# mulitple cpu tests
#=================================================================

rm -rf *.out* *temp *database
