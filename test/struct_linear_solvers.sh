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
# Run ...
#=============================================================================

mpirun -np 1 struct_linear_solvers

if [ "0" = "1" ]
then
  echo "Error: something's wrong" >&2
fi

