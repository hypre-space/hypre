#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

#=============================================================================
# This script prints the hypre version number, date, and time.
# It currently inspects the 'configure' file for this info.
#=============================================================================

case $1 in
    -h|-help) 
        echo 
        echo "$0 [options]"
        echo "  -h|-help       - prints usage information"
        echo "  -version       - prints the release version"
        echo "  -number        - prints the release number"
        echo "  -date          - prints the release day"
        echo "  -time          - prints the release day and time"
        echo 
        exit;;
esac

# NOTE: In order to call this script from other directories,
# we need to get the path info from the command line
VPATH=`dirname $0`
VFILE="${VPATH}/../configure"
VERSION=`grep "HYPRE_VERSION=" $VFILE | cut -d= -f 2 | sed 's/"//g'`
NUMBER=`grep "HYPRE_NUMBER=" $VFILE | cut -d= -f 2`
DATE=`grep "HYPRE_DATE=" $VFILE | cut -d= -f 2 | sed 's/"//g'`
TIME=`grep "HYPRE_TIME=" $VFILE | cut -d= -f 2 | sed 's/"//g'`

# this is the no-option print line
VPRINT=`echo hypre Version $VERSION Date: $DATE`

# this defines the print lines for the various options
case $1 in
    -version)
	VPRINT=$VERSION;;
    -number)
	VPRINT=$NUMBER;;
    -date)
	VPRINT=$DATE;;
    -time)
	VPRINT=$TIME;;
esac

# print the version information
echo $VPRINT
