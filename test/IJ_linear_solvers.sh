#!/bin/sh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************


#===========================================================================
#
# To do: - test a few runs
#===========================================================================

#===========================================================================
# Define HYPRE_ARCH and MPIRUN
#===========================================================================

. ./autotest_arch.sh

MPIRUN="./mpirun.$HYPRE_ARCH"
SLS="IJ_linear_solvers"

#=============================================================================
# IJ_linear_solvers: Run default case, BoomerAMG
#=============================================================================

$MPIRUN -np 1 $SLS 
tail -3 $SLS.log > $SLS.testdata

#=============================================================================
# IJ_linear_solvers: Run 2 proc parallel case, BoomerAMG
#=============================================================================

$MPIRUN -np 2 $SLS -P 1 1 2  
tail -3 $SLS.log > $SLS.testdata

#=============================================================================
# IJ_linear_solvers: Run default case with BoomerAMG_PCG
#=============================================================================

$MPIRUN -np 2 $SLS -solver 1 -rhsrand
tail -3 $SLS.log > $SLS.testdata

#=============================================================================
# IJ_linear_solvers: Run default case with DS_PCG
#=============================================================================

$MPIRUN -np 2 $SLS -solver 2 -rhsrand
tail -3 $SLS.log > $SLS.testdata

#=============================================================================
# IJ_linear_solvers: Run default case with BoomerAMG_GMRES
#=============================================================================

$MPIRUN -np 2 $SLS -solver 3 -rhsrand
tail -3 $SLS.log > $SLS.testdata

#=============================================================================
# IJ_linear_solvers: Run default case with DS_GMRES
#=============================================================================

$MPIRUN -np 2 $SLS -solver 4 -rhsrand
tail -3 $SLS.log > $SLS.testdata

