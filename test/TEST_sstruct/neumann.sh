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
# sstruct: Test various blockings and distributions of default problem
#=============================================================================

tail -3 neumann.out.0 > neumann.testa
grep 'Final Relative Residual Norm' neumann.testa | sed 's/[0-9][0-9]e/e/' > neumann.testdata

tail -3 neumann.out.2 > neumann.testb
grep 'Final Relative Residual Norm' neumann.testb | sed 's/[0-9][0-9]e/e/' > neumann.testdata.temp

diff neumann.testdata neumann.testdata.temp >&2


tail -3 neumann.out.1 > neumann.testa
grep 'Final Relative Residual Norm' neumann.testa | sed 's/[0-9][0-9]e/e/' > neumann.testdata

tail -3 neumann.out.3 > neumann.testb
grep 'Final Relative Residual Norm' neumann.testb | sed 's/[0-9][0-9]e/e/' > neumann.testdata.temp

diff neumann.testdata neumann.testdata.temp >&2

rm -f neumann.testdata neumann.testdata.temp neumann.testa neumann.testb
