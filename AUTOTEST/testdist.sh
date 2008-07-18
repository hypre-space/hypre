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




TEST_ALPHA="-`hostname -a`"
TEST_BETA="$TEST_ALPHA -alc"
TEST_GENERAL="$TEST_BETA -thunder -up -zeus"

while [ "$*" ]
do
   case $1 in
      -h|-help)
         cat <<EOF

   $0 [options] {release}

   where: {release}  is a hypre release tar file (gzipped)

   with options:
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   This script unpacks {release} in the parent directory and lists the tests
   needed to verify it. The list depends on the release type (alpha, beta, or
   general).

   Example usage: $0 hypre-2.0.0.tar.gz

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
testing_dir=`cd ..; pwd`
autotest_dir="$testing_dir/AUTOTEST"
release_file=$1
release_dir=`basename $release_file | awk -F.t '{print $1}'`
case `basename $release_file | awk -F. '{print $3}'` in
   *a)
      TESTS=$TEST_ALPHA
      ;;
   *b)
      TESTS=$TEST_BETA
      ;;
   *)
      TESTS=$TEST_GENERAL
      ;;
esac

# Extract the release
cd $testing_dir
if [ ! -d $release_dir ]; then
   echo "Unpacking the release"
   tar -zxf $release_file
   rm -rf $autotest_dir/machine-*.???
fi

# List the status of the required tests
cd $autotest_dir
src_dir="../$release_dir/src"
echo ""
echo "The followinfg tests are needed to verify this release:"
echo ""
for test in $TESTS
do
   case $test in
      -tux[0-9]*-compilers)
         host=`echo $test | awk -F- '{print $2}'`
         name="tux-compilers"
         ;;

      -tux[0-9]*)
         host=`echo $test | awk -F- '{print $2}'`
         name="tux"
         ;;

      -alc|-thunder|-up|-zeus)
         host=`echo $test | awk -F- '{print $2}'`
         name=$host
         ;;

      -mac)
         host="kolev-mac"
         name="mac"
         ;;
   esac
   if [ ! -e machine-$name.err ]; then
      status="[NOT RUN]"
   else
      if [ -s machine-$name.err ]; then
         status="[FAILED] "
      else
         status="[PASSED] "
      fi
   fi
   echo "$status $TERM -e ./testsrc.sh $src_dir $host:hypre/testing/$host machine-$name.sh &"
done
cat <<EOF

Each of the above tests will run in a new terminal.  The release is verified
when all tests are listed as [PASSED].  If a test fails, you can examine
its error files in the current directory, delete them, and re-run it again.

EOF
