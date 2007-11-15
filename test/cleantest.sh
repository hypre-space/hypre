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

# Echo usage information
case $1 in
   -h|-help)
cat <<EOF

   $0 [-h|-help] [{test_dir}]

   where: {test_dir} is the name of some number of runtest directories
          -h|-help   prints this usage information and exits

   This script removes the '*err*', '*out*', and '*log*' files from the
   specified runtest directories.  If no directory is specified, it is assumed
   that the script is being run from within the hypre 'test' directory, and all
   of the 'TEST_*' directories are cleaned.

   Example usage: $0 TEST_struct TEST_ij

EOF
   exit
   ;;
esac

if [ "x$1" = "x" ]
then
   for testdir in TEST*
   do
      rm -f $testdir/*err*
      rm -f $testdir/*out*
      rm -f $testdir/*log*
   done
else
   while [ "$*" ]
   do
      rm -f $1/*err*
      rm -f $1/*out*
      rm -f $1/*log*
      shift
   done
fi
