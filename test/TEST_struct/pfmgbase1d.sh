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
# struct: Test PFMG 1d run as 2d and 3d by diffing against each other.
#=============================================================================

tail -3 pfmgbase1d.out.0 > pfmgbase1d.testdata

tail -3 pfmgbase1d.out.1 > pfmgbase1d.testdata.temp
diff pfmgbase1d.testdata pfmgbase1d.testdata.temp  >&2

#rm -f pfmgbase1d.out.0 pfmgbase1d.out.1
#rm -f pfmgbase1d.testdata pfmgbase1d.testdata.temp
