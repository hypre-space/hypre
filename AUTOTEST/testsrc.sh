#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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

   $0 [options] {src_dir} {machine:rem_path} {testname}.sh

   where: {src_dir}  is the hypre source directory
          {machine}  is the name of the machine to run on
          {rem_path} is the remote path where the {src_dir} directory
                     will be copied
          {testname} is the user-defined name for the test script

   with options:
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   This script is a specialized version of 'test.sh' that runs script
   {testname}.sh remotely on {machine}.  It is assumed that {testname}.sh takes
   only one argument, which will be set to '..' on the remote machine.

   The script first copies the {src_dir} directory into {machine:rem_path}, then
   copies the current AUTOTEST script directory there (potentially overwriting
   an already existing AUTOTEST directory).

   The output is still collected locally in exactly the same way as 'test.sh'.

   Example usage: $0 .. tux149:. machine-tux.sh

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

# Setup
src_dir=`cd $1; pwd`
machine=`echo $2 | awk -F: '{print $1}'`
rem_path=`echo $2 | awk -F: '{print $2}'`
testname=`basename $3 .sh`
rem_dir=`basename $src_dir`

# Copy the source and AUTOTEST directories using rsync/tar+ssh
# ssh $machine "rm -fr $rem_path/$rem_dir"
# scp -r $src_dir $machine:$rem_path/$rem_dir
# scp -r . $machine:$rem_path/$rem_dir/AUTOTEST
rem_dir_exists=`ssh -q $machine "(/bin/sh -c \"[ -d $rem_path/$rem_dir ] && echo \"yes\" || (mkdir -p $rem_path/$rem_dir; echo \"no\")\")"`
if [ "$rem_dir_exists" == "yes" ]
then
   rsync -zvae ssh --delete $src_dir/ $machine:$rem_path/$rem_dir
else
   tar -C `dirname $src_dir` -zvcf - $rem_dir | ssh $machine tar -C $rem_path -zxf -
fi
rsync -zvae ssh --delete . $machine:$rem_path/$rem_dir/AUTOTEST

# Run the test and copy the results
ssh -q $machine "cd $rem_path/$rem_dir/AUTOTEST; ./test.sh ${testname}.sh .."
rm -fr $testname.???
echo "Copying output files from $machine"
scp -q -r $machine:$rem_path/$rem_dir/AUTOTEST/$testname.\?\?\? .
