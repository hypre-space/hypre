#!/bin/ksh 
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1 $
#EHEADER***********************************************************************

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_SSTRUCT data type (2D)
#=============================================================================

tail -3 symm.out.20 > symm.testdata

tail -3 symm.out.21 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

tail -3 symm.out.22 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

tail -3 symm.out.23 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_PARCSR data type (2D)
#=============================================================================

tail -3 symm.out.24 > symm.testdata

tail -3 symm.out.25 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_SSTRUCT data type (3D)
#=============================================================================

tail -3 symm.out.30 > symm.testdata

tail -3 symm.out.31 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

tail -3 symm.out.32 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

tail -3 symm.out.33 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

#=============================================================================
# sstruct: Check SetSymmetric for HYPRE_PARCSR data type (3D)
#=============================================================================

tail -3 symm.out.34 > symm.testdata

tail -3 symm.out.35 > symm.testdata.temp
diff symm.testdata symm.testdata.temp >&2

#=============================================================================
#=============================================================================

rm -f symm.testdata symm.testdata.temp
