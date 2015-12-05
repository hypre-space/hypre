#!/bin/ksh 
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1 $
#EHEADER***********************************************************************

#=============================================================================
# struct: Test parallel and blocking by diffing against base 3d case
#=============================================================================

tail -3 parflow.out.0 > parflow.testdata

tail -3 parflow.out.1 > parflow.testdata.temp
diff parflow.testdata parflow.testdata.temp >&2

tail -3 parflow.out.2 > parflow.testdata.temp
diff parflow.testdata parflow.testdata.temp >&2

#=============================================================================
# struct: Test parallel and blocking by diffing against base 2d case
#=============================================================================

tail -3 parflow.out.3 > parflow.testdata

tail -3 parflow.out.4 > parflow.testdata.temp
diff parflow.testdata parflow.testdata.temp >&2

tail -3 parflow.out.5 > parflow.testdata.temp
diff parflow.testdata parflow.testdata.temp >&2

rm -f parflow.testdata parflow.testdata.temp
