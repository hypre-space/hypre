#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************

while [ "$*" ]
   do
   case $1 in
      -h|-help)
cat <<EOF

   NOTES:
   - organize the directory structures
   - change the hypre group permissions appropriately
   - checkout the repository before calling 'testsrc.sh'
   - create summary report (option -summary)
   - will have arguments such as '-tux', '-up', etc.

   $0 [options] {machine} {testdir}/{testname}.sh [{testname}.sh args]

   where: {machine} is the name of the machine to run on
          {testdir} is the remote directory where the test scripts are located
          {testname} is the user-defined name for the test script

   with options:
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   This script is the same as 'test.sh', except that it runs the remote test
   {testdir}/{testname}.sh on {machine}.  The output is still collected locally
   in exactly the same way as 'test.sh'.

   Example usage: $0 tux149 linear_solvers/AUTOTEST/default.sh ..

EOF
         exit
         ;;
      -t|-trace)
         set -xv
         shift
         ;;
      *)
         break
         ;;
   esac
done

# machine=$1
# shift
# testdir=`dirname $1`
# testname=`basename $1 .sh`
# shift
# echo "Running remote test [$testname] on $machine"
# ssh $machine "cd ${testdir}; test.sh ${testname}.sh $@"
# echo "Copying output files from $machine"
# scp -r $machine:$testdir/$testname.[eod]?? .
