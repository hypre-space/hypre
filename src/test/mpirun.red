#!/bin/sh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.0 $
#EHEADER***********************************************************************

shift
NP=$1
shift

echo "(yod -sz $NP $*)"
yod -sz $NP $*

