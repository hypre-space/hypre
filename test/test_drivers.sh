#!/bin/sh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#===========================================================================
# Define test driver names
#===========================================================================

TEST_DRIVERS="\
 struct_linear_solvers\
 IJ_linear_solvers\
"

#===========================================================================
# Define HYPRE_ARCH
#===========================================================================

. ./autotest_arch.sh

#===========================================================================
# Run test drivers and log results and errors to file
#===========================================================================

for i in $TEST_DRIVERS
do
    echo "running ${i} test suite..."
    ./${i}.sh 1> ${i}.log 2> ${i}.err
done

#===========================================================================
# Check for errors and send appropriate email
# NOTE: HYPRE_MAIL must support `-s' subject option
#===========================================================================

if [ "$1" = "-mail" ]
then
    echo "checking for errors..."

    HYPRE_MAIL=/usr/ucb/Mail
    case $HYPRE_ARCH in
	dec)
	    HYPRE_MAIL=/usr/bin/Mail;;
	blue)
	    HYPRE_MAIL=/usr/bin/Mail;;
	red)
	    HYPRE_MAIL=/usr/ucb/Mail;;
    esac

    for i in $TEST_DRIVERS
    do
    if [ -s "${i}.err" ]
    then
	RECIPIENTS=`cat ${i}.email`
	$HYPRE_MAIL -s "Error(s) in ${i} test suite" $RECIPIENTS < ${i}.err
    fi
    done
fi


