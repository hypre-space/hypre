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
# To do: - test symmetric/non-symmetric in driver.
#        - add "true" 1d capability. Driver has this - breaks solver.
#        - answer why 2d results differ (see: NOTE below).
#===========================================================================

#===========================================================================
# Define HYPRE_ARCH and MPIRUN
#===========================================================================

. ./hypre_arch.sh

MPIRUN="./mpirun.$HYPRE_ARCH"
DRIVER="./struct_linear_solvers"

#=============================================================================
# SMG: Run base 3d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 12 12 -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# SMG: Test parallel and blocking by diffing against base 3d case
#=============================================================================

$MPIRUN -np 8 $DRIVER -n 6 6 6 -P 2 2 2  -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 3 12 12 -P 4 1 1 -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 12 3 12 -P 1 4 1 -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 12 12 3 -P 1 1 4 -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 1 $DRIVER -n 3 4 3 -b 4 3 4  -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 8 $DRIVER -n 3 3 3 -b 2 2 2 -P 2 2 2  -c 2.0 3.0 40 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# SMG: Run base "true" 2d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 12 1  -d 2 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# SMG: Test parallel and blocking by diffing against base "true" 2d case.
#=============================================================================

$MPIRUN -np 3 $DRIVER -n 4 12 1 -P 3 1 1 -d 2 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 3 $DRIVER -n 4 4 1 -P 1 3 1 -b 3 1 1 -d 2 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

#=============================================================================
# SMG: Test 2d run as 3d by diffing against base "true" 2d case
# Note: last test currently doesn't work.  Why?
#=============================================================================

$MPIRUN -np 4 $DRIVER -n 3 1 12 -P 4 1 1 -c 1 0 1 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 2 $DRIVER -n 1 12 6 -P 1 1 2 -c 0 1 1 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

#$MPIRUN -np 3 $DRIVER -n 12 4 1 -P 1 3 1 -c 1 1 0 -solver 0
#tail -3 $DRIVER.log > $DRIVER.testdata.temp
#diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# SMG: Test 1d run as 2d and 3d by diffing against each other.
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 1 1  -c 1 0 0 -d 2 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata

$MPIRUN -np 2 $DRIVER -n 2 1 1 -P 2 1 1 -b 3 1 1 -c 1 0 0 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# SMG: Test 1d runs as 2d and 3d in different configurations.
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 1 12 1  -c 0 1 0 -d 2 -solver 0

$MPIRUN -np 3 $DRIVER -n 1 2 1 -P 1 3 1 -b 1 2 1 -c 0 1 0 -solver 0

$MPIRUN -np 4 $DRIVER -n 1 1 3 -P 1 1 4  -c 0 0 1 -solver 0

#=============================================================================
# SMG: Test V(1,0) cycle.
#=============================================================================

$MPIRUN -np 3 $DRIVER -P 1 1 3 -v 1 0 -solver 0

if [ "0" = "1" ]
then
  echo "Error: something's wrong" >&2
fi

#=============================================================================
# Periodic SMG: Run base 3d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 8 8 8 -p 0 8 8 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# Periodic SMG: Test parallel and blocking by diffing against base 3d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 2 2 2 -P 1 1 1  -p 0 8 8 -b 4 4 4 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 2 8 8 -P 4 1 1  -p 0 8 8 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 8 2 8 -P 1 4 1  -p 0 8 8 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 8 8 2 -P 1 1 4  -p 0 8 8 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 8 $DRIVER -n 2 2 2 -P 2 2 2  -p 0 8 8 -b 2 2 2 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# Periodic SMG: Run base 3d case (periodic in x), test parallel and blocking,
# and run a full periodic case. Note: driver sets up right hand size for
# full periodic case that satifies compatibility condition, it (the rhs)
# is dependent on blocking and parallel partitioning. Thus results will
# differ with number of blocks and processors. 
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 8 8 8 -p 8 0 0 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata

$MPIRUN -np 8 $DRIVER -n 2 2 2 -P 2 2 2  -p 8 0 0 -b 2 2 2 -solver 0
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 4 8 4 -P 2 1 2  -p 8 8 8 -solver 0

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Run base 3d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 12 12 -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# PFMG: Test parallel and blocking by diffing against base 3d case
#=============================================================================

$MPIRUN -np 8 $DRIVER -n 6 6 6 -P 2 2 2  -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 3 12 12 -P 4 1 1 -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 12 3 12 -P 1 4 1 -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 4 $DRIVER -n 12 12 3 -P 1 1 4 -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 1 $DRIVER -n 3 4 3 -b 4 3 4  -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 8 $DRIVER -n 3 3 3 -b 2 2 2 -P 2 2 2  -c 2.0 3.0 40 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Run base "true" 2d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 12 1  -d 2 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# PFMG: Test parallel and blocking by diffing against base "true" 2d case.
#=============================================================================

$MPIRUN -np 3 $DRIVER -n 4 12 1 -P 3 1 1 -d 2 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 3 $DRIVER -n 4 4 1 -P 1 3 1 -b 3 1 1 -d 2 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

#=============================================================================
# PFMG: Test 2d run as 3d by diffing against base "true" 2d case
#=============================================================================

$MPIRUN -np 4 $DRIVER -n 3 1 12 -P 4 1 1 -c 1 0 1 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 2 $DRIVER -n 1 12 6 -P 1 1 2 -c 0 1 1 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 3 $DRIVER -n 12 4 1 -P 1 3 1 -c 1 1 0 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Test 1d run as 2d and 3d by diffing against each other.
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 1 1  -c 1 0 0 -d 2 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata

$MPIRUN -np 2 $DRIVER -n 2 1 1 -P 2 1 1 -b 3 1 1 -c 1 0 0 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Test 1d runs as 2d and 3d in different configurations.
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 1 12 1  -c 0 1 0 -d 2 -solver 1

$MPIRUN -np 3 $DRIVER -n 1 2 1 -P 1 3 1 -b 1 2 1 -c 0 1 0 -solver 1

$MPIRUN -np 4 $DRIVER -n 1 1 3 -P 1 1 4  -c 0 0 1 -solver 1

#=============================================================================
# PFMG: Test solve of the same problem in different orientations
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 10 10 10 -c 1 1 256 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -n 10 10 10 -c 1 256 1 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

$MPIRUN -np 1 $DRIVER -n 10 10 10 -c 256 1 1 -solver 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Test V(1,0) and V(0,1) cycles.
#=============================================================================

$MPIRUN -np 3 $DRIVER -P 1 1 3 -v 1 0 -solver 1
$MPIRUN -np 3 $DRIVER -P 1 1 3 -v 0 1 -solver 1


#=============================================================================
# CG+PFMG with skip: Run base 3d case
#=============================================================================

$MPIRUN -np 1 $DRIVER -n 12 12 12 -solver 11 -skip 1
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# CG+PFMG with skip: Test parallel and blocking by diffing against base 3d case
#=============================================================================

$MPIRUN -np 8 $DRIVER -n 3 3 3 -b 2 2 2 -P 2 2 2  -solver 11 -skip 1
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

if [ "0" = "1" ]
then
  echo "Error: something's wrong" >&2
fi

#=============================================================================
# Test SMG-CG, PFMG-CG, DSCG, CG, and Hybrid.
#=============================================================================

$MPIRUN -np 3 $DRIVER -P 1 1 3 -solver 10
$MPIRUN -np 3 $DRIVER -P 1 3 1 -solver 11
$MPIRUN -np 3 $DRIVER -P 3 1 1 -solver 17
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 18
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 19

# Test Hybrid without the switch
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 20
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 21
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 25

# Test Hybrid with the switch
$MPIRUN -np 1 $DRIVER -n 30 30 30 -P 1 1 1 -solver 20
$MPIRUN -np 1 $DRIVER -n 30 30 30 -P 1 1 1 -solver 21
$MPIRUN -np 1 $DRIVER -n 30 30 30 -P 1 1 1 -solver 25

$MPIRUN -np 3 $DRIVER -P 1 1 3 -solver 30
$MPIRUN -np 3 $DRIVER -P 1 3 1 -solver 31
$MPIRUN -np 3 $DRIVER -P 3 1 1 -solver 37
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 38
$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 39
