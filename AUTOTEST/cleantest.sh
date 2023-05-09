#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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
         # Use explicit extensions to avoid removing '.bat' files
         rm -fr $testname.err $testname.dir $testname.out $testname.fil
      fi
   done
else
   while [ "$*" ]
   do
      testname=$1
      # Use explicit extensions to avoid removing '.bat' files
      rm -fr $testname.err $testname.dir $testname.out $testname.fil
      shift
   done
fi

