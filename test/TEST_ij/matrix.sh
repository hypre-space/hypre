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
# IJ: tests different ways of generating IJMatrix diffs it against 1 proc case
#=============================================================================

tail -21 matrix.out.0 > matrix.testdata.tmp0
head matrix.testdata.tmp0 > matrix.testdata

tail -21 matrix.out.1 > matrix.testdata.tmp0
head matrix.testdata.tmp0 > matrix.testdata.temp
diff matrix.testdata matrix.testdata.temp >&2

tail -21 matrix.out.2 > matrix.testdata.tmp0
head matrix.testdata.tmp0 > matrix.testdata.temp
diff matrix.testdata matrix.testdata.temp >&2

#rm -f matrix.out.0 matrix.out.1 matrix.out.2
#rm -f matrix.testdata matrix.testdata.tmp0 matrix.testdata.temp
