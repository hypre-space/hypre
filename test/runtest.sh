#!/bin/sh
#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

# globals
BatchMode=0
SendMail=0
DebugMode=0
NoRun=0
JobCheckInterval=10      # sleep time between jobs finished check
GiveUpOnJob=10           # number of hours to wait for job finish
InputString=""
RunString=""

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
function MpirunString
{ # generate default mpirun or psub command
  HOST=`hostname|cut -c1-4`
  POE_NUM_PROCS=1
  POE_NUM_NODES=1
  CPUS_PER_NODE=1
# HOST=${HOST%%.*}
  case $HOST in
    fros*) CPUS_PER_NODE=16
      POE_NUM_PROCS=$2
      POE_NUM_NODES=`expr $POE_NUM_PROCS + $CPUS_PER_NODE - 1`
      POE_NUM_NODES=`expr $POE_NUM_NODES / $CPUS_PER_NODE`
      shift
      shift
      RunString="poe $* -procs $POE_NUM_PROCS -nodes $POE_NUM_NODES"
      ;;
    blue*) CPUS_PER_NODE=4
      POE_NUM_PROCS=$2
      POE_NUM_NODES=`expr $POE_NUM_PROCS + $CPUS_PER_NODE - 1`
      POE_NUM_NODES=`expr $POE_NUM_NODES / $CPUS_PER_NODE`
      shift
      shift
      RunString="poe $* -procs $POE_NUM_PROCS -nodes $POE_NUM_NODES"
      ;;
    tckk*) shift
      RunString="prun -n$*"
      ;;
    peng*) shift
      RunString="prun -n$*"
      ;;
    perr*|achi*|poin*|esak*|ares*|hypn*|weyl*|juve*) MACHINES_FILE="hostname"
      if [ ! -f $MACHINES_FILE ]
      then
        hostname > $MACHINES_FILE
      fi
      MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN -machinefile $MACHINES_FILE $*"
      ;;
    gps*) MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN $*"
      ;;
    lx*) MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN $*"
      ;;
    tc*) MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN $*"
      ;;
    tux*|perr*|achi*) MACHINES_FILE="hostname"
      if [ ! -f $MACHINES_FILE ]
      then
        hostname > $MACHINES_FILE
      fi
      MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN -machinefile $MACHINES_FILE $*"
      ;;
    vivi*) MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN $*"
      ;;
    ript*) MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN $*"
      ;;
    janu*) shift
      RunString="yod -sz $*"
      ;;
    *) MPIRUN=`type mpirun|sed -e 's/^.* //'`
      RunString="$MPIRUN $*"
      ;;
  esac
}
function CheckBatch
{ # determine if host can process psub (batch queues)
  BATCH_MODE=0
  HOST=`hostname|cut -c1-4`
  case $HOST in
    fros*) BATCH_MODE=1
      ;;
    blue*) BATCH_MODE=1
      ;;
    tckk*) BATCH_MODE=1
      ;;
    peng*) BATCH_MODE=1
      ;;
    gps*) BATCH_MODE=1
      ;;
    lx*) BATCH_MODE=0
      ;;
    tc*) BATCH_MODE=1
      ;;
    *) BATCH_MODE=0
      ;;
  esac
  return $BATCH_MODE
}
function CheckSimpleRun
{ # determine if host can process script file directly
  BATCH_MODE=0
  HOST=`hostname|cut -c1-4`
# HOST=${HOST%%.*}
  case $HOST in
    vivi*) BATCH_MODE=1
      ;;
    ript*) BATCH_MODE=1
      ;;
    lx*) BATCH_MODE=1
      ;;
    *) BATCH_MODE=0
      ;;
  esac
  return $BATCH_MODE
}
function CalcNodes
{ # determine the "number of nodes" desired by dividing the
  # "number of processes" by the "number of CPU's per node"
  # which can't be determined dynamically (real ugly hack)
  NUM_PROCS=1
  NUM_NODES=1
  CPUS_PER_NODE=1
  HOST=`hostname|cut -c1-4`
# HOST=${HOST%%.*}              # remove possible ".llnl.gov"
  case $HOST in
    fros*) CPUS_PER_NODE=16
      ;;
    blue*) CPUS_PER_NODE=4
      ;;
    tckk*) CPUS_PER_NODE=4
      ;;
    peng*|ilx*) CPUS_PER_NODE=2
      ;;
    gps*) CPUS_PER_NODE=4
      ;;
    lx*) CPUS_PER_NODE=2
      ;;
    tc*) CPUS_PER_NODE=4
      ;;
    *) CPUS_PER_NODE=1
      ;;
  esac
  while test "$1"
  do case $1 in
    -np) NUM_PROCS=$2
      NUM_NODES=`expr $NUM_PROCS+$CPUS_PER_NODE-1`
      NUM_NODES=`expr $NUM_NODES/$CPUS_PER_NODE`
      return $NUM_NODES
      ;;
    *) shift
      ;;
  esac
  done
  return 1
}
function CalcProcs
{ # extract the "number of processes|task"
  while test "$1"
  do case $1 in
    -np) return $2
      ;;
    *) shift
      ;;
  esac
  done
  return 1
}
function CheckPath
{ # check the path to the executable
  while test "$1"
  do case $1 in
    -np) EXECFILE=$3
      if test -x $StartDir/$EXECFILE
      then
        cp $StartDir/$EXECFILE $EXECFILE
        return 0
      else
        echo "Can not find executable!!!"
        return 1
      fi
      return 0
      ;;
    *) shift
      ;;
  esac
  done
  return 1
}
function ExecuteScripts
{ #   
  if [ "$DebugMode" -gt 0 ] ; then echo "In function ExecuteScripts" ; fi
  StartDir=$1
  WorkingDir=$2
  InputFile=$3
  SavePWD=`pwd`
  cd $WorkingDir
  if [ "$DebugMode" -gt 0 ] ; then echo "./$InputFile.jobs $InputFile.err" ; fi
  ./$InputFile.jobs > $InputFile.err.0 2>&1 
  if [ "$DebugMode" -gt 0 ] ; then echo "./$InputFile.sh $InputFile.err" ; fi
  ./$InputFile.sh > $InputFile.err 2>&1
  cd $SavePWD
}
function PsubCmdStub
{ # initialize the common part of the " PsubCmd" string, ulgy global vars!
  # global "RunName" is assumed to be predefined
  CalcNodes "$@"
  NumNodes=$?
  CalcProcs "$@"
  NumProcs=$?
  HOST=`hostname|cut -c1-4`
# HOST=${HOST%%.*}              # remove possible ".llnl.gov"
  case $HOST in
    fros*) PsubCmd="psub -c frost,pbatch -b a_casc -nettype css0 -r $RunName"
      PsubCmd="$PsubCmd -ln $NumNodes -g $NumProcs"
      ;;
    blue*) PsubCmd="psub -c blue,pbatch -b a_casc -r $RunName"
      PsubCmd="$PsubCmd -ln $NumNodes -g $NumProcs"
      ;;
    tckk*) PsubCmd="psub -c tc2k,pbatch -b casc -r $RunName -ln $NumProcs"
      ;;
    peng*) PsubCmd="psub -c pengra,pbatch -b casc -r $RunName -ln $NumProcs"
      PsubCmd="$PsubCmd -standby"
      ;;
    gps*) PsubCmd="psub -c gps320 -b casc -r $RunName -cpn $NumProcs"
      ;;
    lx*) PsubCmd="psub -c lx -b casc -r $RunName -cpn $NumProcs"
      ;;
    tc*) PsubCmd="psub -c tera -b casc -r $RunName -cpn $NumProcs"
      ;;
    *) PsubCmd="psub -b casc -r $RunName -ln $NumProcs"
      ;;
  esac
}
function ExecuteJobs
{ # read job file line by line saving arguments
  StartDir=$1
  WorkingDir=$2
  InputFile=$3
  if [ "$DebugMode" -gt 0 ]
  then set -xv ; echo "In function ParseJobFile: WorkingDir=$WorkingDir InputFile=$InputFile" ; fi
  ReturnFlag=0              # error return flag
  BatchFlag=0               # #Batch option detected flag 
  BatchCount=0              # different numbering for #Batch option
  PrevPid=0
  SavePWD=`pwd`
  cd $WorkingDir
# exec < $InputFile.jobs              # open *.jobs file for reading
  while read InputLine
  do
    if [ "$DebugMode" -gt 0 ] ; then echo $InputLine ; fi
    case $InputLine in
      \#Bat*|\#bat*|\#BAT*) BatchFlag=1
        BatchFile=""
        if [ "$DebugMode" -gt 0 ] ; then echo "Batch line" ; fi
        ;;
      \#End*|\#end*|\#END*) BatchFlag=0
        if [ "$DebugMode" -gt 0 ] ; then echo "Submit job" ; fi
        chmod +x $BatchFile
        PsubCmd="$PsubCmd -o $OutFile -e $ErrFile `pwd`/$BatchFile"
        if [ "$NoRun" -eq 0 ] ; then CmdReply=`$PsubCmd` ; fi
        PrevPid=`echo $CmdReply | cut -d \  -f 2`
        if [ "$DebugMode" -gt 0 ] ; then echo "PsubCmd=$PsubCmd $PrevPid" ; fi
        while [ "`pstat | grep $PrevPid`" ]
        do sleep $JobCheckInterval      # global, see runtest.sh
        done
        if [ "$DebugMode" -gt 0 ] ; then echo "Job finished `date`" ; fi
        BatchFile=""
        ;;
      *mpirun*)
        if [ "$DebugMode" -gt 0 ] ; then echo "mpirun line" ; fi
        if [ "$DebugMode" -gt 0 ] ; then echo "BatchMode:$BatchMode BatchFlag:$BatchFlag" ; fi
	RunCmd=`echo $InputLine| sed -e 's/^[ \t]*mpirun[ \t]*//'` 
	RunCmd=`echo $RunCmd | sed -e 's/[ \t]*>.*$//'`
        OutFile=`echo $InputLine | sed -e 's/^.*>//'`
        OutFile=`echo $OutFile | sed -e 's/ //g'`
	ErrFile=`echo $OutFile | sed -e 's/\.out\./.err./'`
        RunName=`echo $OutFile | sed -e 's/\.out.*$//'`
        CheckPath $RunCmd               # check path to executable
        if [ "$?" -gt 0 ]
        then
        cat >> $RunName.err <<- EOF
		Executable doesn't exist command: 
		$InputLine 
		EOF
          ReturnFlag=1
          break
        fi
        MpirunString $RunCmd            # construct "RunString"
        if [ "$BatchMode" -eq 0 ]
        then
          if [ "$DebugMode" -gt 0 ] ; then echo "${RunString} > $OutFile 2> $ErrFile" ; fi
#         if [ "$PrevPid" -gt 0 ] ; then wait $PrevPid ; fi
          sh ${RunString} > $OutFile 2> $ErrFile 
#         PrevPid=$!
        else
          if [ "$BatchFlag" -eq 0 ]
          then BatchFile=`echo $OutFile | sed -e 's/\.out\./.batch./'`
            if [ "$DebugMode" -gt 0 ] ; then echo "RunName=$RunName BatchFile=$BatchFile" ; fi
            cat > $BatchFile <<- EOF 
		#!/bin/sh
		cd `pwd`
		${RunString}
		EOF
            chmod +x $BatchFile
            PsubCmdStub ${RunCmd}
            PsubCmd="$PsubCmd -o $OutFile -e $ErrFile `pwd`/$BatchFile"
            if [ "$NoRun" -eq 0 ] ; then CmdReply=`$PsubCmd` ; fi
            PrevPid=`echo $CmdReply | cut -d \  -f 2`
            if [ "$DebugMode" -gt 0 ] ; then echo "$PsubCmd $PrevPid" ; fi
            while [ "`pstat | grep $PrevPid`" ]
            do sleep $JobCheckInterval  # global, see runtest.sh
            done
            if [ "$DebugMode" -gt 0 ] ; then echo "Job finished `date`" ; fi
          else                          # BatchFlag set
            if [ "$DebugMode" -gt 0 ] ; then echo "RunName=$RunName BatchFile=$BatchFile" ; fi
            if [ "$BatchFile" -eq "" ]
            then BatchFile=$InputFile.batch.$BatchCount
              BatchCount=BatchCount+1
              cat > $BatchFile <<- EOF
		#!/bin/sh
		cd `pwd`
		${RunString}
		EOF
            else
              cat >> $BatchFile <<- EOF
		${RunString}
		EOF
            fi
            PsubCmdStub ${RunCmd}       # construct a PsubCmd string
          fi                            # BatchFlag set
        fi                              # BatchMode set
        ;;
     ""|"\n") :
       if [ "$DebugMode" -gt 0 ] ; then echo "Blank line" ; fi
       ;;
     *#*) :
       if [ "$DebugMode" -gt 0 ] ; then echo "# line" ; fi
       ;; 
     *)
       echo "Found something unexpected in $WorkingDir/$InputFile.jobs"
       echo "In line: $InputLine"
       exit 1
       ;;
    esac
  done < $InputFile.jobs              # open *.jobs file for reading
# exec < $InputFile.jobs              # open *.jobs file for reading
# exec 3<&-
  cd $SavePWD
  return $ReturnFlag
}
function ExecuteTest
{ #   
  if [ "$DebugMode" -gt 0 ] ; then echo "In function ExecuteTest" ; fi
  StartDir=$1
  WorkingDir=$2
  InputFile=$3
  SavePWD=`pwd`
  cd $WorkingDir
  if [ "$DebugMode" -gt 0 ] ; then echo "./$InputFile.sh  $InputFile.err " ; fi
  (./$InputFile.sh > $InputFile.err 2>&1 &) 
  cd $SavePWD
}
function PostProcess
{ #   
  if [ "$DebugMode" -gt 0 ] ; then echo "In function PostProcess" ; fi
  StartDir=$1
  WorkingDir=$2
  InputFile=$3
  SavePWD=`pwd`
  cd $WorkingDir
  if [ "$BatchMode" -eq 0 ]
  then
    if test -f purify.log
    then
      mv purify.log $InputFile.purify.log
      grep -i hypre_ $InputFile.purify.log >> $InputFile.err
    elif test -f insure.log
    then
      mv insure.log $InputFile.insure.log
      grep -i hypre_ $InputFile.insure.log >> $InputFile.err
    fi
  fi
  cd $SavePWD
}

function StartCrunch
{ # process files
  CheckSimpleRun                # check if host can just execute scripts
  RtnCode=$?
  if [ "$RtnCode" -ne 0 ]
  then
    ExecuteScripts "$@"
    RtnCode=$?
  else
    ExecuteJobs "$@"
    RtnCode=$?
    if [ "$RtnCode" -eq 0 ]
    then ExecuteTest "$@"
      RtnCode=$?
    fi
  fi
  if [ "$RtnCode" -eq 0 ]
  then PostProcess "$@"
  fi
}

# main
. ./AUTOTEST/hypre_arch.sh
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
