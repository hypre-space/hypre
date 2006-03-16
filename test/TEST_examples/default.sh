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

diff ex1.base ex1.out >&2

diff ex2.base ex2.out >&2

diff ex3.base ex3.out >&2

diff ex4.base ex4.out >&2

diff ex5.base ex5.out >&2

diff ex5b.base ex5b.out >&2

diff ex5b77.base ex5b77.out >&2

diff ex6.base ex6.out >&2

diff ex7.base ex7.out >&2

diff ex8.base ex8.out >&2

diff ex9.base ex9.out >&2
