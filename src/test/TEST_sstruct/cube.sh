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
# sstruct: Test SetNeighborBox by comparing one-part problem
#          against equivalent multi-part problems
#=============================================================================

tail -3 cube.out.0 > cube.testdata

tail -3 cube.out.1 > cube.testdata.temp
diff cube.testdata cube.testdata.temp >&2

tail -3 cube.out.2 > cube.testdata.temp
diff cube.testdata cube.testdata.temp >&2

rm -f cube.testdata cube.testdata.temp
