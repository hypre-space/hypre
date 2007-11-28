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

   $0 [options] {testname}.sh [{testname}.sh args]

    where: {testname} is the user-defined name for the test script

    with options:
       -h|-help       prints this usage information and exits
       -t|-trace      echo each command

   This script runs the Bourne shell test '{testname}.sh' and creates output files
   named '{testname}.err' and '{testname}.out' which capture the stderr and stdout
   output from the test.  The test script is run from the current directory, which
   should contain this script, '{test_name}.sh', and any other supporting files.

   A test is deemed to have passed when nothing is written to stderr.  A test may
   call other tests.  A test may take arguments, such as files or directories.
   A test may also create output.  It is recommended that all output be collected
   by the test in a directory named '{testname}.dir'.  Usage documentation should
   appear at the top of each test.

   Example usage: $0 default.sh ..

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

# Run the test and capture stdout, stderr
testname=`basename $1 .sh`
shift
echo "Running test [$testname]"
./$testname.sh $@ 1>"$testname.out" 2>"$testname.err"

# Filter misleading error messages
if [ -e $testname.filters ]; then
    if (egrep -f $testname.filters $testname.err > /dev/null) ; then
	echo "This file contains the original copy of $testname.err before filtering" > $testname.fil
	cat $testname.err >> $testname.fil
	mv $testname.err $testname.tmp
	egrep -v -f $testname.filters $testname.tmp > $testname.err
	rm -f $testname.tmp
    fi
fi
