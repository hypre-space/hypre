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
# struct: Test parallel and blocking by diffing against base "true" 2d case
#=============================================================================

tail -3 pfmgbase2d.out.0 > pfmgbase2d.testdata

tail -3 pfmgbase2d.out.1 > pfmgbase2d.testdata.temp
diff pfmgbase2d.testdata pfmgbase2d.testdata.temp >&2

tail -3 pfmgbase2d.out.2 > pfmgbase2d.testdata.temp
diff pfmgbase2d.testdata pfmgbase2d.testdata.temp >&2

tail -3 pfmgbase2d.out.3 > pfmgbase2d.testdata.temp
diff pfmgbase2d.testdata pfmgbase2d.testdata.temp >&2

tail -3 pfmgbase2d.out.4 > pfmgbase2d.testdata.temp
diff pfmgbase2d.testdata pfmgbase2d.testdata.temp >&2

tail -3 pfmgbase2d.out.5 > pfmgbase2d.testdata.temp
diff pfmgbase2d.testdata pfmgbase2d.testdata.temp >&2

#rm -f pfmgbase2d.out.0 pfmgbase2d.out.1 pfmgbase2d.out.2
#rm -f pfmgbase2d.out.3 pfmgbase2d.out.4 pfmgbase2d.out.5
#rm -f pfmgbase2d.testdata pfmgbase2d.testdata.temp
