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
# No testing done
# fei: diffs it against 1 proc case
#=============================================================================

tail -21 parallel.out.0 > parallel.testdata.tmp0
head parallel.testdata.tmp0 > parallel.testdata

tail -21 parallel.out.1 > parallel.testdata.tmp0
head parallel.testdata.tmp0 > parallel.testdata.temp

tail -21 parallel.out.2 > parallel.testdata.tmp0
head parallel.testdata.tmp0 > parallel.testdata.temp

tail -21 parallel.out.3 > parallel.testdata.tmp0
head parallel.testdata.tmp0 > parallel.testdata.temp

tail -21 parallel.out.4 > parallel.testdata.tmp0
head parallel.testdata.tmp0 > parallel.testdata.temp

#rm -f parallel.out.0 parallel.out.1 parallel.out.2
#rm -f parallel.out.3 parallel.out.4 
#rm -f parallel.testdata parallel.testdata.tmp0 parallel.testdata.temp
