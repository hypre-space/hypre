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
# struct: Test 1d run as 2d and 3d by diffing against each other.
#=============================================================================

tail -3 smgbase1d.out.0 > smgbase1d.testdata

tail -3 smgbase1d.out.1 > smgbase1d.testdata.temp
diff smgbase1d.testdata smgbase1d.testdata.temp  >&2

#rm -f smgbase1d.out.0 smgbase1d.out.1
#rm -f smgbase1d.testdata smgbase1d.testdata.temp
