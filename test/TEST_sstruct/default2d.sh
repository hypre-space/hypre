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
# sstruct: Test various blockings and distributions of default problem
#=============================================================================

tail -3 default2d.out.0 > default2d.testdata

tail -3 default2d.out.1 > default2d.testdata.temp
diff default2d.testdata default2d.testdata.temp >&2

tail -3 default2d.out.2 > default2d.testdata.temp
diff default2d.testdata default2d.testdata.temp >&2

#rm -f default2d.out.0 default2d.out.1 default2d.out.2 
#rm -f default2d.testdata default2d.testdata.temp
