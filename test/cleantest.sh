#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer and the GNU Lesser General Public License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************


# globals
DefaultTestDirs="TEST_ams TEST_fac TEST_fei TEST_ij TEST_sstruct TEST_struct"
InputString=""

function usage
{
   printf "\n"
   printf "$0 [-h | -help] [-t | -trace] {test_path}\n"
   printf "\n"
   printf " where: {test_path} the directory from which to remove *.err* and \n"
   printf "                    *.out* files.\n"
   printf "                    If TEST, TEST_ or no argument is given for \n"
   printf "                    {test path}, all TEST_* directories are cleaned.\n"
   printf "\n"
   printf "        -h | -help:      prints this usage information and exits.\n"
   printf "        -t | -trace:     echo each command as it is executed.\n"
   printf "\n"
   printf " This script is used to removed output files, *.err* and *.out* \n". 
   printf " from the specified TEST directory.  If the argument is TEST, \n"
   printf " TEST_ or omitted, all TEST directories will have output files removed.\n" 
   printf "\n"
   printf " Example usage: ./cleantest.sh \n"
   printf " Example usage: ./cleantest.sh TEST\n"
   printf " Example usage: ./cleantest.sh -t TEST_sstruct\n"
   printf "\n"
}


# main

CurrentDir=`pwd`
if test "x$1" = "x"
then
   for testdir in $DefaultTestDirs
   do
      cd $testdir
      rm -f *.err* *.out*
      cd $CurrentDir
   done
else
   while [ "$*" ]
   do
      case $1 in
         -h|-help)
             usage
             exit
             ;;
         -t|-trace)
             set -xv
             shift
             ;;
         TEST_fac )
             cd $1
             rm -f *.err* *.out*
             cd $CurrentDir
             shift
             ;;
         TEST_fei )
             cd $1
             rm -f *.err* *.out*
             cd $CurrentDir
             shift
             ;;
         TEST_ij )
             cd $1
             rm -f *.err* *.out*
             cd $CurrentDir
             shift
             ;;
         TEST_sstruct )
             cd $1
             rm -f *.err* *.out*
             cd $CurrentDir
             shift
             ;;
         TEST_struct )
             cd $1
             rm -f *.err* *.out*
             cd $CurrentDir
             shift
             ;;
         * )
             for testdir in $DefaultTestDirs
             do
                cd $testdir
                rm -f *.err* *.out*
                cd $CurrentDir
             done
             shift
             ;;
      esac
   done
fi
