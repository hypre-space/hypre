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

tail -3 default3d.out.0 > default3d.testdata

tail -3 default3d.out.1 > default3d.testdata.temp
diff default3d.testdata default3d.testdata.temp >&2

tail -3 default3d.out.2 > default3d.testdata.temp
diff default3d.testdata default3d.testdata.temp >&2

tail -3 default3d.out.3 > default3d.testdata.temp
diff default3d.testdata default3d.testdata.temp >&2

#rm -f default3d.out.0 default3d.out.1 default3d.out.2 default3d.out.3
#rm -f default3d.testdata default3d.testdata.temp
