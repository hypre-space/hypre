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
# The outputs below should differ only in timings.
#=============================================================================

diff -I"time" solvers.out.0 solvers.out.1 >&2
diff -I"time" solvers.out.2 solvers.out.3 >&2
diff -I"time" solvers.out.4 solvers.out.5 >&2
diff -I"time" solvers.out.6 solvers.out.7 >&2
