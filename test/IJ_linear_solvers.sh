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
# IJ_linear_solvers: Run default case, weigthed Jacobi, BoomerAMG
#=============================================================================

$MPIRUN -np 1 $SLS -rlx 0 -xisone
tail -21 $SLS.log > $SLS.testdata.tmp0
head $SLS.testdata.tmp0 > $SLS.testdata

#=============================================================================
# IJ_linear_solvers: Run 2 and 3 proc parallel case, weighted Jacobi, BoomerAMG 
#		     diffs it against 1 proc case
#=============================================================================

$MPIRUN -np 2 $SLS -P 1 1 2 -rlx 0 -xisone -P 1 1 2 
tail -21 $SLS.log > $SLS.testdata.tmp0
head $SLS.testdata.tmp0 > $SLS.testdata.temp
diff $SLS.testdata $SLS.testdata.temp >&2

$MPIRUN -np 3 $SLS -P 1 1 3 -rlx 0 -xisone -P 1 1 3 
tail -21 $SLS.log > $SLS.testdata.tmp0
head $SLS.testdata.tmp0 > $SLS.testdata.temp
diff $SLS.testdata $SLS.testdata.temp >&2

rm -f $SLS.testdata $SLS.testdata.tmp0 $SLS.testdata.temp

#=============================================================================
# IJ_linear_solvers: tests different ways of generating IJMatrix
#=============================================================================

$MPIRUN -np 2 $SLS -rhsrand
tail -21 $SLS.log > $SLS.testdata.tmp0
head $SLS.testdata.tmp0 > $SLS.testdata

$MPIRUN -np 2 $SLS -rhsrand -exact_size
tail -21 $SLS.log > $SLS.testdata.tmp0
head $SLS.testdata.tmp0 > $SLS.testdata.temp
diff $SLS.testdata $SLS.testdata.temp >&2

$MPIRUN -np 2 $SLS -rhsrand -low_storage
tail -21 $SLS.log > $SLS.testdata.tmp0
head $SLS.testdata.tmp0 > $SLS.testdata.temp
diff $SLS.testdata $SLS.testdata.temp >&2

rm -f $SLS.testdata $SLS.testdata.tmp0 $SLS.testdata.temp

#=============================================================================
# IJ_linear_solvers: Run default case with different coarsenings, hybrid GS,
#    1: Cleary_LJP
#    2: parallel Ruge
#    3: Ruge 3rd pass
#    4: Falgout
#=============================================================================

$MPIRUN -np 4 $SLS -rhsrand -n 15 15 10 -P 2 2 1 -27pt

$MPIRUN -np 4 $SLS -rhsrand -n 15 15 10 -P 2 2 1 -ruge -27pt

$MPIRUN -np 4 $SLS -rhsrand -n 15 15 10 -P 2 2 1 -ruge3c -gm -27pt

$MPIRUN -np 4 $SLS -rhsrand -n 15 15 10 -P 2 2 1 -falgout -27pt

#=============================================================================
#=============================================================================
#=============================================================================
# IJ_linear_solvers: Run default case with different solvers
#    1: BoomerAMG_PCG
#    2: DS_PCG
#    3: BoomerAMG_GMRES
#    4: DS_GMRES
#=============================================================================

$MPIRUN -np 2 $SLS -solver 1 -rhsrand

$MPIRUN -np 2 $SLS -solver 2 -rhsrand

$MPIRUN -np 2 $SLS -solver 3 -rhsrand

$MPIRUN -np 2 $SLS -solver 4 -rhsrand

