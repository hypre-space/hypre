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
# Check SetNeighborBox for periodic problems (2D)
#=============================================================================

tail -3 periodic.out.20 > periodic.testdata

tail -3 periodic.out.21 > periodic.testdata.temp
diff periodic.testdata periodic.testdata.temp >&2

#=============================================================================
# Check SetNeighborBox for periodic problems (2D)
#=============================================================================

tail -3 periodic.out.30 > periodic.testdata

tail -3 periodic.out.31 > periodic.testdata.temp
diff periodic.testdata periodic.testdata.temp >&2

#=============================================================================
#=============================================================================

rm -f periodic.testdata periodic.testdata.temp
