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
# This script sets the HYPRE_ARCH variables for the autotest scripts.
#=============================================================================

HYPRE_OS=""
HYPRE_ARCH=""

#=============================================================================
# Determine the OS
#=============================================================================

if [ -f "/bin/uname" ] 
then
    HYPRE_OS="`/bin/uname -s`"
fi

#=============================================================================
# Based on what we found from system queries set HYPRE_ARCH
#=============================================================================

if [ -z "$HYPRE_ARCH" ]
then
    case "$HYPRE_OS" in
	OSF1)
	    HYPRE_ARCH="dec";;
	AIX)
	    HYPRE_ARCH="frost";;
	"TFLOPS O/S")
	    HYPRE_ARCH="red";;
	Linux)
	    HYPRE_ARCH="linux";;
	IRIX64)
	    HYPRE_ARCH="sgi";;
    esac
fi

