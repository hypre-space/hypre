#!/bin/sh 
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#=============================================================================
# EXAMPLES: Compare ex*.base files with babel.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================


tail -21 babel.out.5b > babel.test.tmp
head babel.test.tmp > babel.test

tail -21 ex5b.base > babel.base.tmp
head babel.base.tmp > babel.base

diff babel.base babel.test >&2

diff ex5b77.base babel.out.5b77 >&2

rm -f babel.test.tmp babel.test babel.base.tmp babel.base
