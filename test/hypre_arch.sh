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
    HYPRE_OS_Release="`/bin/uname -r`"
fi

#=============================================================================
# Based on what we found from system queries set HYPRE_ARCH
#=============================================================================

if [ -z "$HYPRE_ARCH" ]
then
    case "$HYPRE_OS" in
	SunOS)
	    HYPRE_ARCH="casc";;
	OSF1)
	    HYPRE_ARCH="dec";;
	AIX)
	    HYPRE_ARCH="blue";;
	"TFLOPS O/S")
	    HYPRE_ARCH="red";;
	Linux)
	    HYPRE_ARCH="linux";;
    esac
fi

