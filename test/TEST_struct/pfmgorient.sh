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
# struct: Test PFMG 1d solve of the same problem in different orientations
#=============================================================================

tail -3 pfmgorient.out.0 > pfmgorient.testdata

tail -3 pfmgorient.out.1 > pfmgorient.testdata.temp
diff pfmgorient.testdata pfmgorient.testdata.temp  >&2

tail -3 pfmgorient.out.2 > pfmgorient.testdata.temp
diff pfmgorient.testdata pfmgorient.testdata.temp  >&2

#rm -f pfmgorient.out.0 pfmgorient.out.1 pfmgorient.out.2
#rm -f pfmgorient.testdata pfmgorient.testdata.temp
