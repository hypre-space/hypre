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
# struct: Test CG+PFMG with skip parallel and blocking by diffing against
# base 3d case
#=============================================================================

tail -3 cgpfmg3d.out.0 > cgpfmg3d.testdata

tail -3 cgpfmg3d.out.1 > cgpfmg3d.testdata.temp
diff cgpfmg3d.testdata cgpfmg3d.testdata.temp >&2

#rm -f cgpfmg3d.out.0 cgpfmg3d.out.1
#rm -f cgpfmg3d.testdata cgpfmg3d.testdata.temp
