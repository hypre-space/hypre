#!/bin/ksh
#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

. ./hypre_arch.sh
MPIRUN="./mpirun.$HYPRE_ARCH"
integer BatchMode=0
integer SendMail=0
while [ "$*" != "" ]
do case $1 in
    -h|-help)
      echo
      echo "$0 [-b|-batch] [-h|-help] [-mail] $HYPRE_HELP"
      echo "  -batch         use batch queues, DPCS Scheduling"
      echo "  -help          prints usage information"
      echo "  -mail          sends email if test suite fail"
      echo
      exit
    ;;
    -b|-batch)
      BatchMode=1
      MPIRUN="./psub.$HYPRE_ARCH"
      shift
    ;;
    -m|-mail)
      SendMail=1
      shift
    ;;
  esac
done
DRIVER="./sstruct_linear_solvers"
#===========================================================================
# Create an Array containing the test cases to run
#===========================================================================
TestCase[0]="$MPIRUN -np 1 $DRIVER -r 2 2 2 -solver 19"
TestCase[1]="$MPIRUN -np 1 $DRIVER -b 2 2 2 -solver 19"
TestCase[2]="$MPIRUN -np 2 $DRIVER -P 2 1 1 -b 1 2 1 -r 1 1 2 -solver 19"
TestCase[3]="$MPIRUN -np 4 $DRIVER -P 2 1 2 -r 1 2 1 -solver 19"
TestCase[4]="$MPIRUN -np 1 $DRIVER -in sstruct_default_2d.in -r 2 2 1 -solver 19"
TestCase[5]="$MPIRUN -np 1 $DRIVER -in sstruct_default_2d.in -b 2 2 1 -solver 19"
TestCase[6]="$MPIRUN -np 2 $DRIVER -in sstruct_default_2d.in -P 1 2 1 -r 2 1 1 -solver 19"
TestCase[7]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 10"
TestCase[8]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 11"
TestCase[9]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 18"
TestCase[10]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 19"
TestCase[11]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 20"
TestCase[12]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 21"
TestCase[13]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 22"
TestCase[14]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 30"
TestCase[15]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 31"
TestCase[16]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 38"
TestCase[17]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 39"
TestCase[18]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 40"
TestCase[19]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 41"
TestCase[20]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -solver 42"

#===========================================================================
# Main
#===========================================================================

#=============================================================================
# Batch mode, DPCS Scheduling
#=============================================================================
if ((BatchMode > 0))
then typeset -L15 RunName="$(basename $DRIVER)"
  integer loop=0
  integer PrevPid=0
  integer elements=${#TestCase[*]}
  typeset -fx CalcNodes CalcProcs PsubCmdStub
  . ./test_profile.sh

#===========================================================================
# Main batch processing loop
# Enter each test into the batch queue sequentially
#===========================================================================
while ((loop < elements))
do if ((loop>0))
  then PrevPid=${CmdPid[((loop-1))]}
  fi
  PsubCmdStub ${TestCase[$loop]}
  PsubCmd="$PsubCmd -d $PrevPid"
  CmdReply=$($PsubCmd ${TestCase[$loop]}) 
  CmdPid[$loop]=$(print $CmdReply | cut -d \  -f 2)
  print "($PsubCmd ${TestCase[$loop]}) ${CmdPid[$loop]}"
  loop=loop+1
done
PidList="${CmdPid[@]}"

#=============================================================================
# Create error log file
# Log file testing make this part a little messy.
#=============================================================================
typeset TmpEFile=./$RunName.E$$
cat > $TmpEFile <<- EOF
#!/bin/ksh
cd \$PSUB_WORKDIR
> $DRIVER.err
if test -f ./$RunName.e${CmdPid[0]}
then cat ./$RunName.e${CmdPid[0]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[0]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[1]}
then cat ./$RunName.e${CmdPid[1]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[1]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[2]}
then cat ./$RunName.e${CmdPid[2]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[2]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[3]}
then cat ./$RunName.e${CmdPid[3]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[3]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[4]}
then cat ./$RunName.e${CmdPid[4]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[4]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[5]}
then cat ./$RunName.e${CmdPid[5]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[5]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[6]}
then cat ./$RunName.e${CmdPid[6]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[6]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
integer j=0
for i in $PidList
do if ((j>6))
  then if test -f ./$RunName.e\$i
    then cat ./$RunName.e\$i >> $DRIVER.err
    fi
  fi
  j=j+1
done
EOF
chmod +x $TmpEFile 
PrevPid=${CmdPid[((loop-1))]}
PsubCmdStub "-np" "1"
PsubCmd="$PsubCmd -d $PrevPid $TmpEFile" 
CmdReply=$($PsubCmd) 
CmdPid[$loop]=$(print $CmdReply | cut -d \  -f 2)
print "($PsubCmd) ${CmdPid[$loop]}"
loop=loop+1

#=============================================================================
# Check for errors and send appropriate email
#=============================================================================
if ((SendMail > 0))
then typeset TmpMFile=./$RunName.M$$
  typeset MailProg Recipients Subject
  [[ -x /usr/bin/Mail ]] && MailProg=/usr/bin/Mail
  [[ -x /usr/bin/mailx ]] && MailProg=/usr/bin/mailx
  [[ -x /usr/sbin/mailx ]] && MailProg=/usr/sbin/mailx
  if test -r "${DRIVER}.email"
  then Recipients="$(cat ${DRIVER}.email|tr '\n' '\ ')"
    Subject="Errors in ${DRIVER} or DPCS ${SUBMSG} on $(hostname -s)"
    cat > $TmpMFile <<- EOF
#!/bin/ksh
cd \$PSUB_WORKDIR
if test -s "${DRIVER}.err"
then $MailProg -s "$Subject" "$Recipients" < ${DRIVER}.err
fi
EOF
    chmod +x $TmpMFile
    PrevPid=${CmdPid[((loop-1))]}
    PsubCmdStub "-np" "1"
    PsubCmd="$PsubCmd -d $PrevPid $TmpMFile"
    CmdReply=$($PsubCmd)
    CmdPid[$loop]=$(print $CmdReply | cut -d \  -f 2)
    print "($PsubCmd) ${CmdPid[$loop]}"
    loop=loop+1
  fi
fi

#=============================================================================
# Create log file and clean up
#=============================================================================
typeset TmpLFile=./$RunName.L$$
cat > $TmpLFile <<- EOF
#!/bin/ksh
cd \$PSUB_WORKDIR
> $DRIVER.log
for i in $PidList
do [[ -r ./$RunName.o\$i ]] && cat ./$RunName.o\$i >> $DRIVER.log
done
rm -f ./$RunName.\*
EOF
chmod +x $TmpLFile 
PrevPid=${CmdPid[((loop-1))]}
PsubCmdStub "-np" "1"
PsubCmd="$PsubCmd -d $PrevPid $TmpLFile" 
CmdReply=$($PsubCmd) 
CmdPid[$loop]=$(print $CmdReply | cut -d \  -f 2)
print "($PsubCmd) ${CmdPid[$loop]}"
loop=loop+1
#
#=============================================================================
# Interactive mode
#=============================================================================
else 

#=============================================================================
# 3D: Test various blockings and distributions of default problem
#=============================================================================

# base case
CmdReply=$(${TestCase[0]})
tail -3 $DRIVER.log > $DRIVER.testdata

CmdReply=$(${TestCase[1]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[2]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[3]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# 2D: Test various blockings and distributions of default problem
#=============================================================================

# base case
CmdReply=$(${TestCase[4]})
tail -3 $DRIVER.log > $DRIVER.testdata

CmdReply=$(${TestCase[5]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[6]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# Run all of the solvers
#=============================================================================

CmdReply=$(${TestCase[7]})
CmdReply=$(${TestCase[8]})
CmdReply=$(${TestCase[9]})
CmdReply=$(${TestCase[10]})
CmdReply=$(${TestCase[11]})
CmdReply=$(${TestCase[12]})
CmdReply=$(${TestCase[13]})
CmdReply=$(${TestCase[14]})
CmdReply=$(${TestCase[15]})
CmdReply=$(${TestCase[16]})
CmdReply=$(${TestCase[17]})
CmdReply=$(${TestCase[18]})
CmdReply=$(${TestCase[19]})
CmdReply=$(${TestCase[20]})
fi # endif !BatchMode
