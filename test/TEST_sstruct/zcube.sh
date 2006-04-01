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
# sstruct: Test SetNeighborBox by comparing one-part problem
#          against equivalent multi-part problems
#=============================================================================

tail -3 zcube.out.0 > zcube.testdata

tail -3 zcube.out.1 > zcube.testdata.temp
diff zcube.testdata zcube.testdata.temp >&2

rm -f zcube.testdata zcube.testdata.temp
