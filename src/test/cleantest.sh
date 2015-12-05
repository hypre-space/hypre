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
# $Revision: 1.11 $
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
      rm -f $testdir/*.fil
   done
else
   while [ "$*" ]
   do
      rm -f $1/*err*
      rm -f $1/*out*
      rm -f $1/*log*
      rm -f $1/*.fil
      shift
   done
fi
