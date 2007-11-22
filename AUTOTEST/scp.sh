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

testname=`basename $0 .sh`

# Echo usage information
case $1 in
   -h|-help)
cat <<EOF

   $0 [-h] {src_dir} {machine}:{rem_dir}

   where: {src_dir}  is the source directory
          {machine}  is the name of the remote machine
          {rem_dir}  is the destination directory
          -h|-help   prints this usage information and exits

   This script copies (efficiently) the directory tree {src_dir} into
   {machine}:{dest_dir} based on ssh, tar and rsync.

   Example usage: $0 .. thunder:.

EOF
   exit
   ;;
esac

# Setup
src_dir=`cd $1; pwd`
machine=`echo $2 | awk -F: '{print $1}'`
rem_path=`echo $2 | awk -F: '{print $2}'`
rem_dir=`basename $src_dir`

# Check if the remove directory exists
rem_dir_exists=`ssh $machine "(/bin/sh -c \"[ -d $rem_path/$rem_dir ] && echo \"yes\" || (mkdir -p $rem_path/$rem_dir; echo \"no\")\")"`

# Choose between rsync+ssh and tar+ssh
if [ "$rem_dir_exists" == "yes" ]
then
    rsync -zvae ssh --delete $src_dir/ $machine:$rem_path/$rem_dir
else
    if [ "$machine" = "up" ]
    then
	TAR=/usr/local/bin/tar
    else
	TAR=/bin/tar
    fi
    cd `dirname $src_dir`
    tar -zvcf - $rem_dir | ssh $machine $TAR -C $rem_path -zxf -
fi

