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
# struct: Test parallel and blocking by diffing against base 3d case
#=============================================================================

tail -3 psmgbase3d.out.0 > psmgbase3d.testdata

tail -3 psmgbase3d.out.1 > psmgbase3d.testdata.temp
diff psmgbase3d.testdata psmgbase3d.testdata.temp >&2

tail -3 psmgbase3d.out.2 > psmgbase3d.testdata.temp
diff psmgbase3d.testdata psmgbase3d.testdata.temp >&2

tail -3 psmgbase3d.out.3 > psmgbase3d.testdata.temp
diff psmgbase3d.testdata psmgbase3d.testdata.temp >&2

tail -3 psmgbase3d.out.4 > psmgbase3d.testdata.temp
diff psmgbase3d.testdata psmgbase3d.testdata.temp >&2

tail -3 psmgbase3d.out.5 > psmgbase3d.testdata.temp
diff psmgbase3d.testdata psmgbase3d.testdata.temp >&2

#rm -f psmgbase3d.out.0 psmgbase3d.out.1 psmgbase3d.out.2
#rm -f psmgbase3d.out.3 psmgbase3d.out.4 psmgbase3d.out.5
#rm -f psmgbase3d.testdata psmgbase3d.testdata.temp
