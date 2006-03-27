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


diff ex5b.base babel.out.5b >&2

diff ex5b77.base babel.out.5b77 >&2
