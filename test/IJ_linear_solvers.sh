#!/bin/sh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#temporarily turn off test suite
#fmtmsg -u print "IJ_linear_solvers test suite is currently hanging"
#exit

#===========================================================================
#
# To do: - test a few runs
#===========================================================================

#===========================================================================
# Define HYPRE_ARCH and MPIRUN
#===========================================================================

. ./hypre_arch.sh

MPIRUN="./mpirun.$HYPRE_ARCH"
DRIVER="./IJ_linear_solvers"

#=============================================================================
# IJ_linear_solvers: Run default case, weigthed Jacobi, BoomerAMG
#=============================================================================

$MPIRUN -np 1 $DRIVER -rlx 0 -xisone
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

#=============================================================================
# IJ_linear_solvers: Run 2 and 3 proc parallel case, weighted Jacobi, BoomerAMG 
#		     diffs it against 1 proc case
#=============================================================================

$MPIRUN -np 2 $DRIVER -P 1 1 2 -rlx 0 -xisone 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 3 $DRIVER -P 1 1 3 -rlx 0 -xisone 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0 $DRIVER.testdata.temp

#=============================================================================
# IJ_linear_solvers: tests different ways of generating IJMatrix
#=============================================================================

$MPIRUN -np 2 $DRIVER -rhsrand
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 2 $DRIVER -rhsrand -exact_size
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 2 $DRIVER -rhsrand -low_storage
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0 $DRIVER.testdata.temp

#=============================================================================
# IJ_linear_solvers: Run default case with different coarsenings, hybrid GS,
#    1: Cleary_LJP
#    2: parallel Ruge
#    3: Ruge 3rd pass
#    4: Falgout
#=============================================================================

$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -27pt

$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -ruge -27pt

$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -ruge3c -gm -27pt

$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -falgout -27pt

#=============================================================================
#=============================================================================
#=============================================================================
# IJ_linear_solvers: Run default case with different solvers
#    1: BoomerAMG_PCG
#    2: DS_PCG
#    3: BoomerAMG_GMRES
#    4: DS_GMRES
#    5: BoomerAMG_CGNR
#    6: DS_CGNR
#    8: ParaSails_PCG
#=============================================================================

$MPIRUN -np 2 $DRIVER -solver 1 -rhsrand

$MPIRUN -np 2 $DRIVER -solver 2 -rhsrand

$MPIRUN -np 2 $DRIVER -solver 3 -rhsrand

$MPIRUN -np 2 $DRIVER -solver 4 -rhsrand

$MPIRUN -np 2 $DRIVER -solver 5 -rhsrand

$MPIRUN -np 2 $DRIVER -solver 6 -rhsrand

$MPIRUN -np 2 $DRIVER -solver 8 -rhsrand
