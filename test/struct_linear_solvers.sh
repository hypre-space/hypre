#!/bin/ksh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

#===========================================================================
#
# To do: - test symmetric/non-symmetric in driver.
#        - add "true" 1d capability. Driver has this - breaks solver.
#        - answer why 2d results differ (see: NOTE below).
#===========================================================================

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
DRIVER="./struct_linear_solvers"
#===========================================================================
# Create an Array containing the test cases to run
#===========================================================================
TestCase[0]="$MPIRUN -np 1 $DRIVER -n 12 12 12 -c 2.0 3.0 40 -solver 0"
TestCase[1]="$MPIRUN -np 8 $DRIVER -n 6 6 6 -P 2 2 2  -c 2.0 3.0 40 -solver 0"
TestCase[2]="$MPIRUN -np 4 $DRIVER -n 3 12 12 -P 4 1 1 -c 2.0 3.0 40 -solver 0"
TestCase[3]="$MPIRUN -np 4 $DRIVER -n 12 3 12 -P 1 4 1 -c 2.0 3.0 40 -solver 0"
TestCase[4]="$MPIRUN -np 4 $DRIVER -n 12 12 3 -P 1 1 4 -c 2.0 3.0 40 -solver 0"
TestCase[5]="$MPIRUN -np 1 $DRIVER -n 3 4 3 -b 4 3 4  -c 2.0 3.0 40 -solver 0"
TestCase[6]="$MPIRUN -np 8 $DRIVER -n 3 3 3 -b 2 2 2 -P 2 2 2  -c 2.0 3.0 40 -solver 0"
TestCase[7]="$MPIRUN -np 1 $DRIVER -n 12 12 1  -d 2 -solver 0"
TestCase[8]="$MPIRUN -np 3 $DRIVER -n 4 12 1 -P 3 1 1 -d 2 -solver 0"
TestCase[9]="$MPIRUN -np 3 $DRIVER -n 4 4 1 -P 1 3 1 -b 3 1 1 -d 2 -solver 0"
TestCase[10]="$MPIRUN -np 4 $DRIVER -n 3 1 12 -P 4 1 1 -c 1 0 1 -solver 0"
TestCase[11]="$MPIRUN -np 2 $DRIVER -n 1 12 6 -P 1 1 2 -c 0 1 1 -solver 0"
TestCase[12]="$MPIRUN -np 1 $DRIVER -n 12 1 1  -c 1 0 0 -d 2 -solver 0"
TestCase[13]="$MPIRUN -np 2 $DRIVER -n 2 1 1 -P 2 1 1 -b 3 1 1 -c 1 0 0 -solver 0"
TestCase[14]="$MPIRUN -np 1 $DRIVER -n 1 12 1  -c 0 1 0 -d 2 -solver 0"
TestCase[15]="$MPIRUN -np 3 $DRIVER -n 1 2 1 -P 1 3 1 -b 1 2 1 -c 0 1 0 -solver 0"
TestCase[16]="$MPIRUN -np 4 $DRIVER -n 1 1 3 -P 1 1 4  -c 0 0 1 -solver 0"
TestCase[17]="$MPIRUN -np 3 $DRIVER -P 1 1 3 -v 1 0 -solver 0"
TestCase[18]="$MPIRUN -np 1 $DRIVER -n 8 8 8 -p 0 8 8 -solver 0"
TestCase[19]="$MPIRUN -np 1 $DRIVER -n 2 2 2 -P 1 1 1  -p 0 8 8 -b 4 4 4 -solver 0"
TestCase[20]="$MPIRUN -np 4 $DRIVER -n 2 8 8 -P 4 1 1  -p 0 8 8 -solver 0"
TestCase[21]="$MPIRUN -np 4 $DRIVER -n 8 2 8 -P 1 4 1  -p 0 8 8 -solver 0"
TestCase[22]="$MPIRUN -np 4 $DRIVER -n 8 8 2 -P 1 1 4  -p 0 8 8 -solver 0"
TestCase[23]="$MPIRUN -np 8 $DRIVER -n 2 2 2 -P 2 2 2  -p 0 8 8 -b 2 2 2 -solver 0"
TestCase[24]="$MPIRUN -np 1 $DRIVER -n 8 8 8 -p 8 0 0 -solver 0"
TestCase[25]="$MPIRUN -np 8 $DRIVER -n 2 2 2 -P 2 2 2  -p 8 0 0 -b 2 2 2 -solver 0"
TestCase[26]="$MPIRUN -np 4 $DRIVER -n 4 8 4 -P 2 1 2  -p 8 8 8 -solver 0"
TestCase[27]="$MPIRUN -np 1 $DRIVER -n 12 12 12 -c 2.0 3.0 40 -solver 1"
TestCase[28]="$MPIRUN -np 8 $DRIVER -n 6 6 6 -P 2 2 2  -c 2.0 3.0 40 -solver 1"
TestCase[29]="$MPIRUN -np 4 $DRIVER -n 3 12 12 -P 4 1 1 -c 2.0 3.0 40 -solver 1"
TestCase[30]="$MPIRUN -np 4 $DRIVER -n 12 3 12 -P 1 4 1 -c 2.0 3.0 40 -solver 1"
TestCase[31]="$MPIRUN -np 4 $DRIVER -n 12 12 3 -P 1 1 4 -c 2.0 3.0 40 -solver 1"
TestCase[32]="$MPIRUN -np 1 $DRIVER -n 3 4 3 -b 4 3 4  -c 2.0 3.0 40 -solver 1"
TestCase[33]="$MPIRUN -np 8 $DRIVER -n 3 3 3 -b 2 2 2 -P 2 2 2  -c 2.0 3.0 40 -solver 1"
TestCase[34]="$MPIRUN -np 1 $DRIVER -n 12 12 1  -d 2 -solver 1"
TestCase[35]="$MPIRUN -np 3 $DRIVER -n 4 12 1 -P 3 1 1 -d 2 -solver 1"
TestCase[36]="$MPIRUN -np 3 $DRIVER -n 4 4 1 -P 1 3 1 -b 3 1 1 -d 2 -solver 1"
TestCase[37]="$MPIRUN -np 4 $DRIVER -n 3 1 12 -P 4 1 1 -c 1 0 1 -solver 1"
TestCase[38]="$MPIRUN -np 2 $DRIVER -n 1 12 6 -P 1 1 2 -c 0 1 1 -solver 1"
TestCase[39]="$MPIRUN -np 3 $DRIVER -n 12 4 1 -P 1 3 1 -c 1 1 0 -solver 1"
TestCase[40]="$MPIRUN -np 1 $DRIVER -n 12 1 1  -c 1 0 0 -d 2 -solver 1"
TestCase[41]="$MPIRUN -np 2 $DRIVER -n 2 1 1 -P 2 1 1 -b 3 1 1 -c 1 0 0 -solver 1"
TestCase[42]="$MPIRUN -np 1 $DRIVER -n 1 12 1  -c 0 1 0 -d 2 -solver 1"
TestCase[43]="$MPIRUN -np 3 $DRIVER -n 1 2 1 -P 1 3 1 -b 1 2 1 -c 0 1 0 -solver 1"
TestCase[44]="$MPIRUN -np 4 $DRIVER -n 1 1 3 -P 1 1 4  -c 0 0 1 -solver 1"
TestCase[45]="$MPIRUN -np 1 $DRIVER -n 10 10 10 -c 1 1 256 -solver 1"
TestCase[46]="$MPIRUN -np 1 $DRIVER -n 10 10 10 -c 1 256 1 -solver 1"
TestCase[47]="$MPIRUN -np 1 $DRIVER -n 10 10 10 -c 256 1 1 -solver 1"
TestCase[48]="$MPIRUN -np 3 $DRIVER -P 1 1 3 -v 1 0 -solver 1"
TestCase[49]="$MPIRUN -np 3 $DRIVER -P 1 1 3 -v 0 1 -solver 1"
TestCase[50]="$MPIRUN -np 1 $DRIVER -n 12 12 12 -solver 11 -skip 1"
TestCase[51]="$MPIRUN -np 8 $DRIVER -n 3 3 3 -b 2 2 2 -P 2 2 2  -solver 11 -skip 1"
TestCase[52]="$MPIRUN -np 3 $DRIVER -P 1 1 3 -solver 10"
TestCase[53]="$MPIRUN -np 3 $DRIVER -P 1 3 1 -solver 11"
TestCase[54]="$MPIRUN -np 3 $DRIVER -P 3 1 1 -solver 17"
TestCase[55]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 18"
TestCase[56]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 19"
TestCase[57]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 20"
TestCase[58]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 21"
TestCase[59]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 25"
TestCase[60]="$MPIRUN -np 1 $DRIVER -n 30 30 30 -P 1 1 1 -solver 20"
TestCase[61]="$MPIRUN -np 1 $DRIVER -n 30 30 30 -P 1 1 1 -solver 21"
TestCase[62]="$MPIRUN -np 1 $DRIVER -n 30 30 30 -P 1 1 1 -solver 25"
TestCase[63]="$MPIRUN -np 3 $DRIVER -P 1 1 3 -solver 30"
TestCase[64]="$MPIRUN -np 3 $DRIVER -P 1 3 1 -solver 31"
TestCase[65]="$MPIRUN -np 3 $DRIVER -P 3 1 1 -solver 37"
TestCase[66]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 38"
TestCase[67]="$MPIRUN -np 1 $DRIVER -P 1 1 1 -solver 39"

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
if test -f ./$RunName.e${CmdPid[4]}
then cat ./$RunName.e${CmdPid[4]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[4]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
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
if test -f ./$RunName.e${CmdPid[7]}
then cat ./$RunName.e${CmdPid[7]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[7]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[8]}
then cat ./$RunName.e${CmdPid[8]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[8]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[9]}
then cat ./$RunName.e${CmdPid[9]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[9]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[10]}
then cat ./$RunName.e${CmdPid[10]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[10]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[11]}
then cat ./$RunName.e${CmdPid[11]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[11]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[12]}
then cat ./$RunName.e${CmdPid[12]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[12]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[13]}
then cat ./$RunName.e${CmdPid[13]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[13]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[14]}
then cat ./$RunName.e${CmdPid[14]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[14]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
for i in 15 16 17 18
do if test -f ./$RunName.e${CmdPid[$i]}
  then cat ./$RunName.e${CmdPid[$i]} >> $DRIVER.err
  fi
done
if test -f ./$RunName.e${CmdPid[19]}
then cat ./$RunName.e${CmdPid[19]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[19]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[20]}
then cat ./$RunName.e${CmdPid[20]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[20]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[21]}
then cat ./$RunName.e${CmdPid[21]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[21]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[22]}
then cat ./$RunName.e${CmdPid[22]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[22]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[23]}
then cat ./$RunName.e${CmdPid[23]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[23]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[24]}
then cat ./$RunName.e${CmdPid[24]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[24]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[25]}
then cat ./$RunName.e${CmdPid[25]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[25]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[26]}
then cat ./$RunName.e${CmdPid[26]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[26]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[27]}
then cat ./$RunName.e${CmdPid[27]} >> $DRIVER.err
fi
if test -f ./$RunName.e${CmdPid[28]}
then cat ./$RunName.e${CmdPid[28]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[28]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[29]}
then cat ./$RunName.e${CmdPid[29]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[29]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[30]}
then cat ./$RunName.e${CmdPid[30]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[30]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[31]}
then cat ./$RunName.e${CmdPid[31]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[31]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[32]}
then cat ./$RunName.e${CmdPid[32]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[32]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[33]}
then cat ./$RunName.e${CmdPid[33]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[33]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[34]}
then cat ./$RunName.e${CmdPid[34]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[34]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[35]}
then cat ./$RunName.e${CmdPid[35]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[35]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[36]}
then cat ./$RunName.e${CmdPid[36]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[36]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[37]}
then cat ./$RunName.e${CmdPid[37]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[37]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[38]}
then cat ./$RunName.e${CmdPid[38]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[38]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[39]}
then cat ./$RunName.e${CmdPid[39]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[39]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[40]}
then cat ./$RunName.e${CmdPid[40]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[40]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
if test -f ./$RunName.e${CmdPid[41]}
then cat ./$RunName.e${CmdPid[41]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[41]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[42]}
then cat ./$RunName.e${CmdPid[42]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[42]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
for i in 43 44 45
do if test -f ./$RunName.e${CmdPid[$i]}
  then cat ./$RunName.e${CmdPid[$i]} >> $DRIVER.err
  fi
done
if test -f ./$RunName.e${CmdPid[46]}
then cat ./$RunName.e${CmdPid[46]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[46]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[47]}
then cat ./$RunName.e${CmdPid[47]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[47]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
if test -f ./$RunName.e${CmdPid[48]}
then cat ./$RunName.e${CmdPid[48]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[48]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
for i in 49 50
do if test -f ./$RunName.e${CmdPid[$i]}
  then cat ./$RunName.e${CmdPid[$i]} >> $DRIVER.err
  fi
done
if test -f ./$RunName.e${CmdPid[51]}
then cat ./$RunName.e${CmdPid[51]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[51]} > $DRIVER.testdata
if test -f ./$RunName.e${CmdPid[52]}
then cat ./$RunName.e${CmdPid[52]} >> $DRIVER.err
fi
tail -3 ./$RunName.o${CmdPid[52]} > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >> $DRIVER.err
rm -f $DRIVER.testdata $DRIVER.testdata.temp
integer j=0
for i in $PidList
do if ((j>52))
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
# SMG: Run base 3d case
#=============================================================================

CmdReply=$(${TestCase[0]})
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# SMG: Test parallel and blocking by diffing against base 3d case
#=============================================================================

CmdReply=$(${TestCase[1]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[2]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[3]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[4]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[5]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[6]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# SMG: Run base "true" 2d case
#=============================================================================

CmdReply=$(${TestCase[7]})
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# SMG: Test parallel and blocking by diffing against base "true" 2d case.
#=============================================================================

CmdReply=$(${TestCase[8]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[9]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

#=============================================================================
# SMG: Test 2d run as 3d by diffing against base "true" 2d case
# Note: last test currently doesn't work.  Why?
#=============================================================================

CmdReply=$(${TestCase[10]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[11]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

#$MPIRUN -np 3 $DRIVER -n 12 4 1 -P 1 3 1 -c 1 1 0 -solver 0
#tail -3 $DRIVER.log > $DRIVER.testdata.temp
#diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# SMG: Test 1d run as 2d and 3d by diffing against each other.
#=============================================================================

CmdReply=$(${TestCase[12]})
tail -3 $DRIVER.log > $DRIVER.testdata

CmdReply=$(${TestCase[13]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# SMG: Test 1d runs as 2d and 3d in different configurations.
#=============================================================================

CmdReply=$(${TestCase[14]})

CmdReply=$(${TestCase[15]})

CmdReply=$(${TestCase[16]})

#=============================================================================
# SMG: Test V(1,0) cycle.
#=============================================================================

CmdReply=$(${TestCase[17]})

if [ "0" = "1" ]
then
  echo "Error: something's wrong" >&2
fi

#=============================================================================
# Periodic SMG: Run base 3d case
#=============================================================================

CmdReply=$(${TestCase[18]})
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# Periodic SMG: Test parallel and blocking by diffing against base 3d case
#=============================================================================

CmdReply=$(${TestCase[19]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[20]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[21]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[22]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[23]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# Periodic SMG: Run base 3d case (periodic in x), test parallel and blocking,
# and run a full periodic case. Note: driver sets up right hand size for
# full periodic case that satifies compatibility condition, it (the rhs)
# is dependent on blocking and parallel partitioning. Thus results will
# differ with number of blocks and processors. 
#=============================================================================

CmdReply=$(${TestCase[24]})
tail -3 $DRIVER.log > $DRIVER.testdata

CmdReply=$(${TestCase[25]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[26]})

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Run base 3d case
#=============================================================================

CmdReply=$(${TestCase[27]})
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# PFMG: Test parallel and blocking by diffing against base 3d case
#=============================================================================

CmdReply=$(${TestCase[28]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[29]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[30]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[31]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[32]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[33]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Run base "true" 2d case
#=============================================================================

CmdReply=$(${TestCase[34]})
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# PFMG: Test parallel and blocking by diffing against base "true" 2d case.
#=============================================================================

CmdReply=$(${TestCase[35]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[36]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

#=============================================================================
# PFMG: Test 2d run as 3d by diffing against base "true" 2d case
#=============================================================================

CmdReply=$(${TestCase[37]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[38]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[39]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Test 1d run as 2d and 3d by diffing against each other.
#=============================================================================

CmdReply=$(${TestCase[40]})
tail -3 $DRIVER.log > $DRIVER.testdata

CmdReply=$(${TestCase[41]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Test 1d runs as 2d and 3d in different configurations.
#=============================================================================

CmdReply=$(${TestCase[42]})

CmdReply=$(${TestCase[43]})

CmdReply=$(${TestCase[44]})

#=============================================================================
# PFMG: Test solve of the same problem in different orientations
#=============================================================================

CmdReply=$(${TestCase[45]})
tail -3 $DRIVER.log > $DRIVER.testdata

CmdReply=$(${TestCase[46]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

CmdReply=$(${TestCase[47]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

#=============================================================================
# PFMG: Test V(1,0) and V(0,1) cycles.
#=============================================================================

CmdReply=$(${TestCase[48]})
CmdReply=$(${TestCase[49]})

#=============================================================================
# CG+PFMG with skip: Run base 3d case
#=============================================================================

CmdReply=$(${TestCase[50]})
tail -3 $DRIVER.log > $DRIVER.testdata

#=============================================================================
# CG+PFMG with skip: Test parallel and blocking by diffing against base 3d case
#=============================================================================

CmdReply=$(${TestCase[51]})
tail -3 $DRIVER.log > $DRIVER.testdata.temp
diff $DRIVER.testdata $DRIVER.testdata.temp >&2

rm -f $DRIVER.testdata $DRIVER.testdata.temp

if [ "0" = "1" ]
then
  echo "Error: something's wrong" >&2
fi

#=============================================================================
# Test SMG-CG, PFMG-CG, DSCG, CG, and Hybrid.
#=============================================================================

CmdReply=$(${TestCase[52]})
CmdReply=$(${TestCase[53]})
CmdReply=$(${TestCase[54]})
CmdReply=$(${TestCase[55]})
CmdReply=$(${TestCase[56]})

# Test Hybrid without the switch
CmdReply=$(${TestCase[57]})
CmdReply=$(${TestCase[58]})
CmdReply=$(${TestCase[59]})

# Test Hybrid with the switch
CmdReply=$(${TestCase[60]})
CmdReply=$(${TestCase[61]})
CmdReply=$(${TestCase[62]})
CmdReply=$(${TestCase[63]})
CmdReply=$(${TestCase[64]})
CmdReply=$(${TestCase[65]})
CmdReply=$(${TestCase[66]})
CmdReply=$(${TestCase[67]})
fi # endif !BatchMode
