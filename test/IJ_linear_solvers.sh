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
DRIVER="./IJ_linear_solvers"
#===========================================================================
# Create an Array containing the test cases to run
#===========================================================================
TestCase[0]="$MPIRUN -np 1 $DRIVER -cljp -rlx 0 -xisone"
TestCase[1]="$MPIRUN -np 2 $DRIVER -P 1 1 2 -cljp -rlx 0 -xisone"
TestCase[2]="$MPIRUN -np 3 $DRIVER -P 1 1 3 -cljp -rlx 0 -xisone"
TestCase[3]="$MPIRUN -np 2 $DRIVER -rhsrand"
TestCase[4]="$MPIRUN -np 2 $DRIVER -rhsrand -exact_size"
TestCase[5]="$MPIRUN -np 2 $DRIVER -rhsrand -low_storage"
TestCase[6]="$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -cljp -27pt"
TestCase[7]="$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -ruge -27pt"
TestCase[8]="$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -ruge3c -gm -27pt"
TestCase[9]="$MPIRUN -np 4 $DRIVER -rhsrand -n 15 15 10 -P 2 2 1 -falgout -27pt"
TestCase[10]="$MPIRUN -np 2 $DRIVER -solver 1 -rhsrand"
TestCase[11]="$MPIRUN -np 2 $DRIVER -solver 2 -rhsrand"
TestCase[12]="$MPIRUN -np 2 $DRIVER -solver 3 -rhsrand"
TestCase[13]="$MPIRUN -np 2 $DRIVER -solver 4 -rhsrand"
TestCase[14]="$MPIRUN -np 2 $DRIVER -solver 5 -rhsrand"
TestCase[15]="$MPIRUN -np 2 $DRIVER -solver 6 -rhsrand"
TestCase[16]="$MPIRUN -np 2 $DRIVER -solver 7 -rhsrand"
TestCase[17]="$MPIRUN -np 2 $DRIVER -solver 8 -rhsrand"
TestCase[18]="$MPIRUN -np 2 $DRIVER -solver 20 -rhsrand"
TestCase[19]="$MPIRUN -np 2 $DRIVER -solver 20 -cf 0.5 -rhsrand"

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
if test -f ./$RunName.e${CmdPid[0]}
then cat ./$RunName.e${CmdPid[0]} >> $DRIVER.err
fi
tail -21 ./$RunName.o${CmdPid[0]} > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[1]}
then cat ./$RunName.e${CmdPid[1]} >> $DRIVER.err
fi
tail -21 ./$RunName.o${CmdPid[1]} > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[2]}
then cat ./$RunName.e${CmdPid[2]} >> $DRIVER.err
fi
tail -21 ./$RunName.o${CmdPid[2]} > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.tmp0 $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[3]}
then cat ./$RunName.e${CmdPid[3]} >> $DRIVER.err
fi
tail -21 ./$RunName.o${CmdPid[3]} > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[4]}
then cat ./$RunName.e${CmdPid[4]} >> $DRIVER.err
fi
tail -21 ./$RunName.o${CmdPid[4]} > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[5]}
then cat ./$RunName.e${CmdPid[5]} >> $DRIVER.err
fi
tail -21 ./$RunName.o${CmdPid[5]} > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.tmp0 $DRIVER.testdata.temp
integer j=0
for i in $PidList
do if ((j>5))
  then if test -r ./$RunName.e\$i
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
# IJ_linear_solvers: Run default case, weigthed Jacobi, BoomerAMG
#=============================================================================

CmdReply=$(${TestCase[0]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata

#=============================================================================
# IJ_linear_solvers: Run 2 and 3 proc parallel case, weighted Jacobi, BoomerAMG 
#		     diffs it against 1 proc case
#=============================================================================

CmdReply=$(${TestCase[1]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[2]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.tmp0 $DRIVER.testdata.temp

###=============================================================================
### IJ_linear_solvers: tests different ways of generating IJMatrix
###=============================================================================
##
CmdReply=$(${TestCase[3]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata
##
CmdReply=$(${TestCase[4]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2
##
CmdReply=$(${TestCase[5]})
tail -21 $DRIVER.log > $DRIVER.testdata.tmp0
head $DRIVER.testdata.tmp0 > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2
##
rm -f $DRIVER.testdata $DRIVER.testdata.tmp0 $DRIVER.testdata.temp
##
###=============================================================================
### IJ_linear_solvers: Run default case with different coarsenings, hybrid GS,
###    1: Cleary_LJP
###    2: parallel Ruge
###    3: Ruge 3rd pass
###    4: Falgout
###=============================================================================
##
CmdReply=$(${TestCase[6]})
CmdReply=$(${TestCase[7]})
CmdReply=$(${TestCase[8]})
CmdReply=$(${TestCase[9]})
##
###=============================================================================
###=============================================================================
###=============================================================================
### IJ_linear_solvers: Run default case with different solvers
###    1: BoomerAMG_PCG
###    2: DS_PCG
###    3: BoomerAMG_GMRES
###    4: DS_GMRES
###    5: BoomerAMG_CGNR
###    6: DS_CGNR
###    7: PILUT_GMRES
###    8: ParaSails_PCG
###   20: Hybrid_PCG
###=============================================================================
##
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
fi # endif !BatchMode
