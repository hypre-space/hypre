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
# struct: Test Periodic SMG base 3d case (periodic in x), test parallel and blocking,
# and run a full periodic case. Note: driver sets up right hand size for
# full periodic case that satifies compatibility condition, it (the rhs)
# is dependent on blocking and parallel partitioning. Thus results will
# differ with number of blocks and processors.
#=============================================================================

tail -3 psmgtest3d.out.0 > psmgtest3d.testdata

tail -3 psmgtest3d.out.1 > psmgtest3d.testdata.temp
diff psmgtest3d.testdata psmgtest3d.testdata.temp >&2

#rm -f psmgtest3d.out.0 psmgtest3d.out.1 psmgtest3d.out.2
#rm -f psmgtest3d.testdata psmgtest3d.testdata.temp
