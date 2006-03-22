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
# EXAMPLES: Compare ex*.base files with ex*.out files from current runs
#           differences (except for timings) indicate errors
#=============================================================================

diff ex1.base default.out.1 >&2

diff ex2.base default.out.2 >&2

diff ex3.base default.out.3 >&2

diff ex4.base default.out.4 >&2

diff ex5.base default.out.5 >&2

diff ex5b.base default.out.5b >&2

diff ex5b77.base default.out.5b77 >&2

diff ex6.base default.out.6 >&2

diff ex7.base default.out.7 >&2

diff ex8.base default.out.8 >&2

diff ex9.base default.out.9 >&2
