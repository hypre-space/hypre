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
# IJ: Run 2 and 3 proc parallel case, weighted Jacobi, BoomerAMG 
#                    diffs it against 1 proc case
#=============================================================================

tail -21 default.out.0 > default.testdata.tmp0
head default.testdata.tmp0 > default.testdata

tail -21 default.out.1 > default.testdata.tmp0
head default.testdata.tmp0 > default.testdata.temp
diff default.testdata default.testdata.temp >&2

tail -21 default.out.2 > default.testdata.tmp0
head default.testdata.tmp0 > default.testdata.temp
diff default.testdata default.testdata.temp >&2

#rm -f default.out.0 default.out.1 default.out.2
#rm -f default.testdata default.testdata.tmp0 default.testdata.temp
