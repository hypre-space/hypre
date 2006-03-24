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

tail -21 ex1.base > tmp.base.1
head tmp.base.1 > base.1

tail -21 default.out.1 > tmp.out.1
head tmp.out.1 > out.1
diff base.1 out.1 >&2


tail -21 ex2.base > tmp.base.2
head tmp.base.2 > base.2

tail -21 default.out.2 > tmp.out.2
head tmp.out.2 > out.2
diff base.2 out.2 >&2


tail -21 ex3.base > tmp.base.3
head tmp.base.3 > base.3

tail -21 default.out.3 > tmp.out.3
head tmp.out.3 > out.3
diff base.3 out.3 >&2


tail -21 ex4.base > tmp.base.4
head tmp.base.4 > base.4

tail -21 default.out.4 > tmp.out.4
head tmp.out.4 > out.4
diff base.4 out.4 >&2


tail -21 ex5.base > tmp.base.5
head tmp.base.5 > base.5

tail -21 default.out.5 > tmp.out.5
head tmp.out.5 > out.5
diff base.5 out.5 >&2


tail -21 ex5b.base > tmp.base.5b
head tmp.base.5b > base.5b

tail -21 default.out.5b > tmp.out.5b
head tmp.out.5b > out.5b
diff base.5b out.5b >&2


tail -21 ex5b77.base > tmp.base.5b77
head tmp.base.5b77 > base.5b77

tail -21 default.out.5b77 > tmp.out.5b77
head tmp.out.5b77 > out.5b77
diff base.5b77 out.5b77 >&2


tail -21 ex6.base > tmp.base.6
head tmp.base.6 > base.6

tail -21 default.out.6 > tmp.out.6
head tmp.out.6 > out.6
diff base.6 out.6 >&2


tail -21 ex7.base > tmp.base.7
head tmp.base.7 > base.7

tail -21 default.out.7 > tmp.out.7
head tmp.out.7 > out.7
diff base.7 out.7 >&2


tail -21 ex8.base > tmp.base.8
head tmp.base.8 > base.8

tail -21 default.out.8 > tmp.out.8
head tmp.out.8 > out.8
diff base.8 out.8 >&2


tail -21 ex9.base > tmp.base.9
head tmp.base.9 > base.9

tail -21 default.out.9 > tmp.out.9
head tmp.out.9 > out.9
diff base.9 out.9 >&2


rm -f tmp.base.* base.* tmp.out.* out.*
