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
DRIVER="./fei_linear_solvers"

#=============================================================================
# fei_linear_solvers 
#=============================================================================

$MPIRUN -np 1 $DRIVER -solver 0 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -solver 1 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -solver 2 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -solver 3 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -solver 4 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 1 $DRIVER -solver 5 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0

#=============================================================================
# fei_linear_solvers: Run 4 proc parallel case
#=============================================================================

$MPIRUN -np 4 $DRIVER -solver 0 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 4 $DRIVER -solver 1 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 4 $DRIVER -solver 2 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 4 $DRIVER -solver 3 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

$MPIRUN -np 4 $DRIVER -solver 5 
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0

