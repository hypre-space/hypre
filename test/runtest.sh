#!/bin/ksh
#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

typeset -fx StartCrunch ParseJobFile 
typeset -fx ExecuteJobs ExecuteTest
typeset -fx MpirunString CheckBatch
typeset -fx CalcNodes CalcProcs PsubCmdStub
integer BatchMode=0
integer SendMail=0
integer DebugMode=0
integer JobCheckInterval=60             # sleep time between jobs finished check
integer GiveUpOnJob=10                  # number of hours to wait for job finish
InputString=""
RunString=""
. ./AUTOTEST/hypre_arch.sh
. ./funcs.sh
while [ "$*" != "" ]
do case $1 in
    -h|-help)
      echo
      echo "$0 [-b|-batch] [-h|-help] [-mail] $HYPRE_HELP"
      echo "  -batch         use batch queues, DPCS Scheduling"
      echo "  -help          prints usage information"
      echo "  -mail          sends email if test suite fail"
      echo "  -debug         turn on debug mode"
      echo
      exit
    ;;
    -b|-batch)
      BatchMode=1
      shift
    ;;
    -d|-debug)
      DebugMode=1
      shift
    ;;
    -m|-mail)
      SendMail=1
      shift
    ;;
    *) InputString=$1
      if [[ $InputString != "" ]]
      then
        if [[ -f $InputString ]] && [[ -r $InputString ]]
        then FilePart=$(basename $InputString .sh)
          DirPart=$(dirname $InputString)
          if (( BatchMode == 0 ))       # machine DCSP capable
          then
            CheckBatch
            BatchMode=$?
          fi
          if (( DebugMode > 0 ))
          then print "FilePart:$FilePart DirPart:$DirPart" ; fi
          if [[ -f $DirPart/$FilePart.jobs ]] && [[ -r $DirPart/$FilePart.jobs ]]
          then 
            if (( BatchMode == 0 ))
            then                        # strict serial execution
              StartCrunch $DirPart $FilePart
            else                        # parallel; shotgun mode
              StartCrunch $DirPart $FilePart &
            fi
          else print "$0: test command file $DirPart/$FilePart.jobs does not exist"
            exit 1
          fi
        else print "$0: test command file $InputString does not exist"
          print "can not find .sh file"
          exit 1
        fi
      else print "$0: Strange input parameter=$InputString"
        exit 1
      fi
      shift
    ;;
  esac
done
