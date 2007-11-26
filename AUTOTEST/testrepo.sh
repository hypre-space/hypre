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

   $0 [options] "{cvs_opts}" {machine:rem_path} {testname}.sh

   where: {cvs_opts} are options to be passed to cvs checkout
          {machine}  is the name of the machine to run on
          {rem_path} is the remote path where the {release} source directory
                     will be copied
          {testname} is the user-defined name for the test script

   with options:
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   This script is similar to the 'testsrc.sh' script.  The main difference is
   that this script checks out a version from the CVS repository (in the /tmp
   directory) to create the {src_dir} argument of 'testsrc.sh'.

   Example usage: $0 "" tux149:. machine-tux.sh

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
cvs_opts=$1
current_dir=`pwd`
shift

# Checkout the repository in the /tmp directory
cd /tmp
rm -fr linear_solvers
cvs -d /home/casc/repository checkout $cvs_opts linear_solvers
cd $current_dir

# Run the test
./testsrc.sh /tmp/linear_solvers $@
