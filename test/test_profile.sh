#!/bin/ksh
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

function CalcNodes
{ # determine the "number of nodes" desired by dividing the
  # "number of processes" by the "number of CPU's per node"
  # which can't be determined dynamically (real ugly hack)
  typeset -i NUM_PROCS=1
  typeset -i NUM_NODES=1
  typeset -i CPUS_PER_NODE
  typeset -L4 HOST=$(hostname -s)
  case $HOST in
    fros*) CPUS_PER_NODE=16
      ;;
    blue*) CPUS_PER_NODE=4
      ;;
    tckk*) CPUS_PER_NODE=4
      ;;
    peng*) CPUS_PER_NODE=2
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
      NUM_NODES=NUM_PROCS+CPUS_PER_NODE-1
      NUM_NODES=NUM_NODES/CPUS_PER_NODE
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
function PsubCmdStub
{ # initialize the common part of the " PsubCmd" string, ulgy global vars!
  CalcNodes "$@"
  NumNodes=$?
  CalcProcs "$@"
  NumProcs=$?
  typeset -L4 HOST=$(hostname -s)
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
