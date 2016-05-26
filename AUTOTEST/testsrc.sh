#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
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

   Example usage: $0 ../src tux149:. machine-tux.sh

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
echo "Copying sources to $machine"
rem_dir_exists=`ssh -q $machine "(/bin/sh -c \"[ -d $rem_path/$rem_dir ] && echo \"yes\" || (mkdir -p $rem_path/$rem_dir; echo \"no\")\")"`
if [ "$rem_dir_exists" == "no" ]
then
   tar -C `dirname $src_dir` -zcf - $rem_dir | ssh -q $machine tar -C $rem_path -zxf -
else
   rsync -zae "ssh -q" --delete $src_dir/ $USER@$machine:$rem_path/$rem_dir
fi
rsync -zae "ssh -q" --delete . $USER@$machine:$rem_path/$rem_dir/AUTOTEST

# Run the test and copy the results
# Use the '.hyprerc' file when needed to customize the environment
hyprerc_exists=`ssh -q $machine "( /bin/sh -c '[ -f .hyprerc ] && echo yes' )"`
if [ "$hyprerc_exists" == "yes" ]
then
   ssh -q $machine "source .hyprerc; cd $rem_path/$rem_dir/AUTOTEST; ./test.sh ${testname}.sh .."
else
   ssh -q $machine "cd $rem_path/$rem_dir/AUTOTEST; ./test.sh ${testname}.sh .."
fi
rm -fr $testname.???
echo "Copying output files from $machine"
scp -q -r $machine:$rem_path/$rem_dir/AUTOTEST/$testname.\?\?\? .
