#!/bin/ksh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
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
DRIVER="./fei_linear_solvers"

#=============================================================================
# fei_linear_solvers 
# Create an Array containing the test cases to run
#=============================================================================
TestCase[0]="$MPIRUN -np 1 $DRIVER -solver 0"
TestCase[1]="$MPIRUN -np 1 $DRIVER -solver 1"
TestCase[2]="$MPIRUN -np 1 $DRIVER -solver 2"
TestCase[3]="$MPIRUN -np 1 $DRIVER -solver 4"
TestCase[4]="$MPIRUN -np 1 $DRIVER -solver 5"
TestCase[5]="$MPIRUN -np 1 $DRIVER -solver 6"
TestCase[6]="$MPIRUN -np 4 $DRIVER -solver 0"
TestCase[7]="$MPIRUN -np 4 $DRIVER -solver 1"
TestCase[8]="$MPIRUN -np 4 $DRIVER -solver 2"
TestCase[9]="$MPIRUN -np 4 $DRIVER -solver 3"
TestCase[10]="$MPIRUN -np 4 $DRIVER -solver 5"

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
# Currently, No real testing done on results
#=============================================================================
typeset TmpEFile=./$RunName.E$$
cat > $TmpEFile <<- EOF
#!/bin/ksh
cd \$PSUB_WORKDIR
> $DRIVER.err
integer j=0
for i in $PidList
do if test -f ./$RunName.e\$i
  then cat ./$RunName.e\$i >> $DRIVER.err
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
rm -f ./$RunName.*
chmod -R a+rX,ug+w .
chgrp -fR hypre .
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

CmdReply=$(${TestCase[0]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[1]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[2]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[3]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[4]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[5]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0

#=============================================================================
# fei_linear_solvers: Run 4 proc parallel case
#=============================================================================

CmdReply=$(${TestCase[6]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[7]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[8]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[9]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

CmdReply=$(${TestCase[10]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0

fi # endif !BatchMode
