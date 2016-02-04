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

# Which tests to run?
TEST_PATCH="-tux339"
TEST_MINOR="$TEST_PATCH -rzzeus -rzmerl -vulcan"
TEST_MAJOR="$TEST_MINOR"
TERMCMD=""

while [ "$*" ]
do
   case $1 in
      -h|-help)
         cat <<EOF

   $0 [options] {release}

   where: {release}  is a hypre release tar file (gzipped, absolute path)

   with options:
      -xterm         run the tests in parallel using multiple xterm windows
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   This script unpacks {release} in the parent directory and lists the tests
   needed to verify it, based on the type of release (MAJOR, MINOR, or PATCH).
   If all required tests pass, a verification file is generated containing the
   logs from the runs.  Otherwise, tests that have failed or have not been run
   yet can be started, and the script will have to be re-run after their
   completion to generate the verification file.

   Example usage: $0 /usr/casc/hypre/hypre-2.10.1.tar.gz

   NOTE: The absolute path for the release is required.

   NOTE: Because of ssh restrictions at LLNL, run this script on an LC machine.
   You may need to change the default tux platform at the top of this file to
   your own tux machine.  Finally, for each release tar file, it is recommended
   that you run this script inside a separate copy of the AUTOTEST directory
   (this will avoid result conflicts in common tests).

EOF
         exit
         ;;
      -t|-trace)
         set -xv
         shift
         ;;
      -xterm)
         # Get the terminal command and make sure it runs bash
         TERMCMD="$TERM -e"; SHELL=/bin/sh
         shift
         ;;
      *)
         break
         ;;
   esac
done

# Setup
testing_dir=`cd ..; pwd`
autotest_dir="$testing_dir/AUTOTEST"
release_file=$1
release_dir=`basename $release_file | awk -F.tar '{print $1}'`
release=`echo $release_dir | sed 's/hypre-//' | sed 's/.tar.gz//'`
output_dir="$testing_dir/AUTOTEST-hypre-$release"
case $release in
   [1-9][0-9]*.0.0)                     NAME="MAJOR"; TESTS=$TEST_MAJOR ;;
   [1-9][0-9]*.[1-9][0-9]*.0)           NAME="MINOR"; TESTS=$TEST_MINOR ;;
   [1-9][0-9]*.[1-9][0-9]*.[1-9][0-9]*) NAME="PATCH"; TESTS=$TEST_PATCH ;;
   *)                                   NAME="PATCH"; TESTS=$TEST_PATCH ;;
esac

# Extract the release
cd $testing_dir
echo "Checking the distribution file..."
tmpdir=$release_dir.TMP
mkdir -p $tmpdir
rm -rf $tmpdir/$release_dir
tar -C $tmpdir -zxf $release_file
if !(diff -r $release_dir $tmpdir/$release_dir 2>/dev/null 1>&2) then
   rm -rf $release_dir $output_dir $autotest_dir/autotest-*
   tar -zxf $release_file
fi
rm -rf $tmpdir
echo ""
echo "The following tests are needed to verify this $NAME release: $TESTS"
echo ""

# List the status of the required tests
cd $autotest_dir
NOTRUN=""
FAILED=""
PENDING=""
for test in $TESTS
do
   name=`echo $test | sed 's/[0-9]//g'`
   # Determine failed, pending, passed and tests that have not been run
   if [ -f $output_dir/machine$name.err ]; then
      if [ -s $output_dir/machine$name.err ]; then
         status="[FAILED] "; FAILED="$FAILED $test"
      else
         status="[PASSED] ";
      fi
   elif [ ! -e autotest$name-start ]; then
      status="[NOT RUN]"; NOTRUN="$NOTRUN $test"
   elif [ ! -e autotest$name-done ]; then
      status="[PENDING]"; PENDING="$PENDING $test"
   else
      status="[UNKNOWN]";
   fi
   if [ "$TERMCMD" == "" ]; then
      echo "$status ./autotest.sh -dist $release $test"
   else
      echo "$status $TERMCMD ./autotest.sh -dist $release $test &"
   fi
done

# If all tests have been run, create a tarball of the log files
if [ "$NOTRUN$PENDING" == "" ]; then
   echo ""; echo "Generating the verification file AUTOTEST-hypre-$release.tgz"
   cd $testing_dir
   mv -f $autotest_dir/autotest-* $output_dir
   tar -zcf $autotest_dir/AUTOTEST-hypre-$release.tgz `basename $output_dir`
fi

# If all tests have passed, print a message and exit
if [ "$NOTRUN$FAILED$PENDING" == "" ]; then
   echo "The release is verified!"
   exit
fi

cat <<EOF

The release can not be automatically verified at this time because not all tests
are listed as [PASSED].  You may choose to continue with the release anyway, but
it is your responsibility to ensure that the test errors are acceptable.

This script can start the remaining tests now.  Alternatively, you can run the
above commands manually (or in a cron job). If you do this, make sure to examine
the standart error of the autotest.sh script.

EOF

echo -n "Do you want to start the remaining tests? (yes,no) : "
read -e RUN
if [ "$RUN" == "yes" ]; then
   for test in $FAILED $NOTRUN
   do
      name=`echo $test | sed 's/[0-9]//g'`
      rm -rf $output_dir/machine$name.??? autotest$name*
      if [ "$TERMCMD" == "" ]; then
         echo "Running test [./autotest.sh -dist $release $test]"
         ./autotest.sh -dist $release $test 2>> autotest$name.err
      else
         echo "Running test [$TERMCMD ./autotest.sh -dist $release $test &]"
         $TERMCMD "./autotest.sh -dist $release $test 2>> autotest$name.err" 2>> autotest$name.err &
      fi
      echo ""
   done
fi
echo ""
echo "Re-run the script after tests have completed to verify the release."
echo ""
