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

tail -3 neumann.out.0 > neumann.testdata

tail -3 neumann.out.2 > neumann.testdata.temp
diff neumann.testdata neumann.testdata.temp >&2

tail -3 neumann.out.1 > neumann.testdata

tail -3 neumann.out.3 > neumann.testdata.temp
diff neumann.testdata neumann.testdata.temp >&2

rm -f neumann.testdata neumann.testdata.temp
