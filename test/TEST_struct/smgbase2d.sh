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

tail -3 smgbase2d.out.0 > smgbase2d.testdata

tail -3 smgbase2d.out.1 > smgbase2d.testdata.temp
diff smgbase2d.testdata smgbase2d.testdata.temp >&2

tail -3 smgbase2d.out.2 > smgbase2d.testdata.temp
diff smgbase2d.testdata smgbase2d.testdata.temp >&2

tail -3 smgbase2d.out.3 > smgbase2d.testdata.temp
diff smgbase2d.testdata smgbase2d.testdata.temp >&2

tail -3 smgbase2d.out.4 > smgbase2d.testdata.temp
diff smgbase2d.testdata smgbase2d.testdata.temp >&2

#rm -f smgbase2d.out.0 smgbase2d.out.1 smgbase2d.out.2
#rm -f smgbase2d.out.3 smgbase2d.out.4 
#rm -f smgbase2d.testdata smgbase2d.testdata.temp
