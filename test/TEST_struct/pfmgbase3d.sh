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
# struct: Test PFMG parallel and blocking by diffing against base 3d case
#=============================================================================

tail -3 pfmgbase3d.out.0 > pfmgbase3d.testdata

tail -3 pfmgbase3d.out.1 > pfmgbase3d.testdata.temp
diff pfmgbase3d.testdata pfmgbase3d.testdata.temp >&2

tail -3 pfmgbase3d.out.2 > pfmgbase3d.testdata.temp
diff pfmgbase3d.testdata pfmgbase3d.testdata.temp >&2

tail -3 pfmgbase3d.out.3 > pfmgbase3d.testdata.temp
diff pfmgbase3d.testdata pfmgbase3d.testdata.temp >&2

tail -3 pfmgbase3d.out.4 > pfmgbase3d.testdata.temp
diff pfmgbase3d.testdata pfmgbase3d.testdata.temp >&2

tail -3 pfmgbase3d.out.5 > pfmgbase3d.testdata.temp
diff pfmgbase3d.testdata pfmgbase3d.testdata.temp >&2

tail -3 pfmgbase3d.out.6 > pfmgbase3d.testdata.temp
diff pfmgbase3d.testdata pfmgbase3d.testdata.temp >&2

#rm -f pfmgbase3d.out.0 pfmgbase3d.out.1 pfmgbase3d.out.2 pfmgbase3d.out.3
#rm -f pfmgbase3d.out.4 pfmgbase3d.out.5 pfmgbase3d.out.6
#rm -f pfmgbase3d.testdata pfmgbase3d.testdata.temp
