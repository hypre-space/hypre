#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2015,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

# Echo usage information
case $1 in
   -h|-help)
cat <<EOF

   $0 [-h|-help] [{testname}]

   where: {testname} is the name of an autotest test (or multiple tests)
          -h|-help   prints this usage information and exits

   This script removes the '.???' files and directories (e.g., .err and .dir)
   for the specified tests.  If no test is specified, the '.err' files in the
   current directory determine the test names to use.

   Example usage: $0 machine-tux

EOF
   exit
   ;;
esac

if [ "x$1" = "x" ]
then
   for i in *.err
   do
      if [ -f $i ] # This check is important in the case that there are no .err files
      then
         testname=`basename $i .err`
         rm -fr $testname.???
      fi
   done
else
   while [ "$*" ]
   do
      testname=$1
      rm -fr $testname.???
      shift
   done
fi

