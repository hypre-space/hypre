#!/bin/ksh 
#BHEADER***********************************************************************
# (c) 2004   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#=============================================================================
# struct: Test PFMG constant coefficient runs on 1 and 2 processors
# by diffing against each other.
#=============================================================================

tail -3 pfmgconstcoeffmp.out.0 > pfmgconstcoeffmp.testdata

tail -3 pfmgconstcoeffmp.out.1 > pfmgconstcoeffmp.testdata.temp
diff pfmgconstcoeffmp.testdata pfmgconstcoeffmp.testdata.temp  >&2

rm -f pfmgconstcoeffmp.testdata pfmgconstcoeffmp.testdata.temp
