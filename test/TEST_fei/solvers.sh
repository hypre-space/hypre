#!/bin/ksh 
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#=============================================================================
# No testing done
# fei: diffs it against 1 proc case
#=============================================================================

tail -21 solvers.out.0 > solvers.testdata.tmp0
head solvers.testdata.tmp0 > solvers.testdata

tail -21 solvers.out.1 > solvers.testdata.tmp0
head solvers.testdata.tmp0 > solvers.testdata.temp

tail -21 solvers.out.2 > solvers.testdata.tmp0
head solvers.testdata.tmp0 > solvers.testdata.temp

tail -21 solvers.out.3 > solvers.testdata.tmp0
head solvers.testdata.tmp0 > solvers.testdata.temp

tail -21 solvers.out.4 > solvers.testdata.tmp0
head solvers.testdata.tmp0 > solvers.testdata.temp

tail -21 solvers.out.5 > solvers.testdata.tmp0
head solvers.testdata.tmp0 > solvers.testdata.temp

#rm -f solvers.out.0 solvers.out.1 solvers.out.2
#rm -f solvers.out.3 solvers.out.4 solvers.out.5
#rm -f solvers.testdata solvers.testdata.tmp0 solvers.testdata.temp
