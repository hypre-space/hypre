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
# Set hypre test driver prefixes
#
# NOTE: Assumes test driver names are of the form <prefix>_linear_solvers
#=============================================================================

HYPRE_TESTS="struct sstruct IJ fei"
 
#=============================================================================
# Parse arguments and define test driver names
#=============================================================================

# set help line
HYPRE_HELP=""
for i in $HYPRE_TESTS
do HYPRE_HELP="$HYPRE_HELP[${i}] "
done

HYPRE_TEST_ARGS=""
while [ "$*" != "" ]
do case $1 in
  -h|-help) echo 
    echo "$0 [-b|-batch] [-h|-help] [-mail] $HYPRE_HELP"
    echo "  -batch         use batch queues, DPCS Scheduling"
    echo "  -help          prints usage information"
    echo "  -mail          sends email if test suites fail"
    echo 
    exit
    ;;
  -b|-batch) HYPRE_BATCH_QUEUE="yes"
    shift
    ;;
  -mail) HYPRE_SEND_MAIL="yes"
    shift
    ;;
  *) HYPRE_TEST_ARGS="$HYPRE_TEST_ARGS $1"
    shift
    ;;
  esac
done

#=============================================================================
# Define test driver names
#=============================================================================

# if no driver arguments, run all drivers
if [ "$HYPRE_TEST_ARGS" = "" ]
then HYPRE_TEST_ARGS="$HYPRE_TESTS"
fi

HYPRE_TEST_DRIVERS=""
for i in $HYPRE_TEST_ARGS
do HYPRE_TEST_DRIVERS="$HYPRE_TEST_DRIVERS ${i}_linear_solvers"
done

#===========================================================================
# Define HYPRE_ARCH
#===========================================================================

. ./hypre_arch.sh

#===========================================================================
# Run test drivers and log results and errors to file
#===========================================================================

SUBMSG="test suite"
for i in $HYPRE_TEST_DRIVERS
do if test -x ${i}
  then echo "running ${i} test suite..."
    if [ "$HYPRE_BATCH_QUEUE" = "yes" ]
    then if [ "$HYPRE_SEND_MAIL" = "yes" ]
      then ./${i}.sh -batch -mail
      else ./${i}.sh -batch
      fi
    else ./${i}.sh 1> ${i}.log 2> ${i}.err
    fi
    if test -f purify.log
    then mv purify.log ${i}.purify.log
      mv ${i}.err ${i}.err.log
      grep -i hypre_ ${i}.purify.log > ${i}.err
      SUBMSG="from purify"
    fi
  else echo "ERROR ${i} test suite did not compile..."
    echo "${i} does not exist" > ${i}.log
    echo "test suite not run, check for compile errors." >> ${i}.log
    echo "${i} does not exist" > ${i}.err
    echo "test suite not run, check for compile errors." >> ${i}.err
  fi
done

#===========================================================================
# Check for errors and send appropriate email
# NOTE: HYPRE_MAIL must support `-s' subject option
#===========================================================================

if [ "$HYPRE_SEND_MAIL" = "yes" ]
then if [ "$HYPRE_BATCH_QUEUE" != "yes" ]
  then echo "checking for errors..."

    [ -x /usr/bin/Mail ] && HYPRE_MAIL=/usr/bin/Mail
    [ -x /usr/bin/mailx ] && HYPRE_MAIL=/usr/bin/mailx
    [ -x /usr/sbin/mailx ] && HYPRE_MAIL=/usr/sbin/mailx

    for i in $HYPRE_TEST_DRIVERS
    do if test -s "${i}.err"
      then if test -r "${i}.email"
        then RECIPIENTS=`cat ${i}.email`
            SUBJECT="Error(s) in ${i} ${SUBMSG} ($HYPRE_ARCH)"
	    $HYPRE_MAIL -s "$SUBJECT" $RECIPIENTS < ${i}.err
        fi
      fi
    done
  fi
fi
