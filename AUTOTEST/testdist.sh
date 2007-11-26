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

   $0 [options] {release} {machine:rem_path} {testname}.sh

   where: {release}  is a hypre release tar file (gzipped)
          {machine}  is the name of the machine to run on
          {rem_path} is the remote path where the {release} source directory
                     will be copied
          {testname} is the user-defined name for the test script

   with options:
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   This script is similar to the 'testsrc.sh' script.  The main difference is
   that this script unpacks {release} (in the /tmp directory) to create the
   {src_dir} argument of 'testsrc.sh'.

   Example usage: $0 hypre-2.0.0.tar.gz tux149:. machine-tux.sh

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
release_file=$1
release_name=`basename $release_file`
release_dir=`echo $release_name | awk -F.t '{print $1}'`
current_dir=`pwd`
shift

# Extract release in the /tmp directory
cd /tmp
rm -fr /tmp/$release_dir
tar -zxvf $release_file -C /tmp $release_dir/src
mv -f /tmp/$release_dir/src /tmp/$release_dir/$release_dir-src
cd $current_dir

# Run the test
./testsrc.sh /tmp/$release_dir/$release_dir-src $@
