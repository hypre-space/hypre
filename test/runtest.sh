#!/bin/sh
#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

usage () {
  printf "$0 [-d|-debug] [-n|-norun] [-h|-help] TEST_dir/testfile.sh\n"
  printf "\n"
  printf "Hypre test driver utility. It is assumed that:\n"
  printf "1) the testing scripts live in test/TEST_{solver}\n" 
  printf "directories; 2) in these directories are individual\n"
  printf "test scripts, named {test_name}.job which provides\n"
  printf "the mpirun execution syntax; 3) execution produces\n"
  printf "output files (names of the form {test_name}.out.{number}\n"
  printf "is used by convension) for the test; 4) A file named\n"
  printf "{test_name}.sh is used to check the test results\n"
  printf "(usually by "diffing" the output files).  Idealy the\n"
  printf ".jobs, and .sh files can be executed stand alone (i.e.,\n"
  printf "standalone shell script files). Test success is assumed\n"
  printf "when no output is generated (from the .sh script file\n"
  printf "the .job script file requires output files), and;\n"
  printf "6) Autotest will by default runs all test in the TEST_fei,\n"
  printf "TEST_ij, TEST_sstruct, and, TEST_struct directories.\n"
  printf "Note: runtest.sh knows about most the ASCI machines,\n"
  printf "and will automatically use DPCS batch queue if required).\n"
  printf "Example usage: ./runtest.sh TEST_sstruct/*.sh\n"
  printf "\n"
  printf "  -help          prints usage information\n"
  printf "  -debug         turn on debug messages\n"
  printf "  -norun         turn off execute, echo mode\n"
  printf "  -trace         turn on debug, and echo each command\n"
  printf "\n"
}
#typeset -fx StartCrunch ParseJobFile 
#typeset -fx ExecuteJobs ExecuteTest
#typeset -fx MpirunString CheckBatch
#typeset -fx CalcNodes CalcProcs PsubCmdStub
BatchMode=0
SendMail=0
DebugMode=0
NoRun=0
JobCheckInterval=10      # sleep time between jobs finished check
GiveUpOnJob=10           # number of hours to wait for job finish
InputString=""
RunString=""
. ./AUTOTEST/hypre_arch.sh
. ./funcs.sh
while [ "$*" ]
do case $1 in
    -h|-help)
      usage
      exit
    ;;
    -d|-debug)
      DebugMode=1
      shift
    ;;
    -n|-norun)
      NoRun=1
      shift
    ;;
    -t|-trace)
      DebugMode=1
      set -xv
      shift
    ;;
    *) InputString=$1
      if [ "$InputString" ]
      then
        if test -f $InputString && test -r $InputString
        then FilePart=`basename $InputString .sh`
          DirPart=`dirname $InputString`
          CurDir=`pwd`
          if [ "$BatchMode" -eq 0 ]       # machine DCSP capable
          then
            CheckBatch
            BatchMode=$?
          fi
          if [ "$DebugMode" -gt 0 ]
          then printf "FilePart:%s DirPart:%s\n" $FilePart $DirPart ; fi
          if test -f $DirPart/$FilePart.jobs && test -r $DirPart/$FilePart.jobs
          then StartCrunch $CurDir $DirPart $FilePart # strict serial execution
          else printf "%s: test command file %s/%s.jobs does not exist\n" \
              $0 $DirPart $FilePart
            exit 1
          fi
        else printf "%s: test command file %s does not exist\n" \
            $0 $InputString
          printf "can not find .sh file\n"
          exit 1
        fi
      else printf "%s: Strange input parameter=%s\n" $0 $InputString
        exit 1
      fi
      shift
    ;;
  esac
done
