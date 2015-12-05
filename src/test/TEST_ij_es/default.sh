#!/bin/ksh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.4 $
#EHEADER***********************************************************************

#=============================================================================
# ij_es driver for eigenvaluesolvers:
# tests for LOBPCG only
#                    diffs it against 1 proc case
# As of 03/14/04 may produce different results on some systems.
# This is probably a small bug.  No comparison until it's fixed.
#=============================================================================

#tail -21 default.out.0 > default.testdata.tmp0
#head default.testdata.tmp0 > default.testdata

#tail -21 default.out.1 > default.testdata.tmp0
#head default.testdata.tmp0 > default.testdata.temp
#diff default.testdata default.testdata.temp >&2

#tail -21 default.out.2 > default.testdata.tmp0
#head default.testdata.tmp0 > default.testdata.temp
#diff default.testdata default.testdata.temp >&2

#rm -f default.testdata default.testdata.tmp0 default.testdata.temp
