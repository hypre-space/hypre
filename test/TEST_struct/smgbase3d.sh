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

tail -3 smgbase3d.out.0 > smgbase3d.testdata

tail -3 smgbase3d.out.1 > smgbase3d.testdata.temp
diff smgbase3d.testdata smgbase3d.testdata.temp >&2

tail -3 smgbase3d.out.2 > smgbase3d.testdata.temp
diff smgbase3d.testdata smgbase3d.testdata.temp >&2

tail -3 smgbase3d.out.3 > smgbase3d.testdata.temp
diff smgbase3d.testdata smgbase3d.testdata.temp >&2

tail -3 smgbase3d.out.4 > smgbase3d.testdata.temp
diff smgbase3d.testdata smgbase3d.testdata.temp >&2

tail -3 smgbase3d.out.5 > smgbase3d.testdata.temp
diff smgbase3d.testdata smgbase3d.testdata.temp >&2

tail -3 smgbase3d.out.6 > smgbase3d.testdata.temp
diff smgbase3d.testdata smgbase3d.testdata.temp >&2

#rm -f smgbase3d.out.0 smgbase3d.out.1 smgbase3d.out.2 smgbase3d.out.3
#rm -f smgbase3d.out.4 smgbase3d.out.5 smgbase3d.out.6
#rm -f smgbase3d.testdata smgbase3d.testdata.temp
