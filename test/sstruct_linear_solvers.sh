#!/bin/sh
#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************


#===========================================================================
#
#===========================================================================

#===========================================================================
# Define HYPRE_ARCH and MPIRUN
#===========================================================================

. ./hypre_arch.sh

MPIRUN="./mpirun.$HYPRE_ARCH"
DRIVER="./sstruct_linear_solvers"

#=============================================================================
# 3D: Test various blockings and distributions of default problem
#=============================================================================

# base case
$MPIRUN -np 1 $DRIVER -r 2 2 2 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -b 2 2 2 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 2 $DRIVER -P 2 1 1 -b 1 2 1 -r 1 1 2 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -P 2 1 2 -r 1 2 1 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# 2D: Test various blockings and distributions of default problem
#=============================================================================

# base case
$MPIRUN -np 1 $DRIVER -in sstruct_default_2d.in -r 2 2 1 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -in sstruct_default_2d.in -b 2 2 1 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 2 $DRIVER -in sstruct_default_2d.in -P 1 2 1 -r 2 1 1 -solver 19
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# Run all of the solvers
#=============================================================================

$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 10
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 11
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 18
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 19
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 20
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 21
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 22
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 30
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 31
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 38
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 39
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 40
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 41
$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 42
