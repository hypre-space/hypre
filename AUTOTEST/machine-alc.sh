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

testname=`basename $0 .sh`

# Echo usage information
case $1 in
   -h|-help)
cat <<EOF

   **** Only run this script on the alc cluster ****

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs a number of tests suitable for the alc cluster.

   Example usage: $0 ..

EOF
   exit
   ;;
esac

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=$1
shift

# Set some environment variables
PATH=/usr/local/KAI/KCC_BASE/bin:$PATH
PATH=/usr/local/mpi/bin:/opt/intel/compiler90/bin:$PATH
export PATH
LD_LIBRARY_PATH=/opt/intel/compiler90/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
LM_LICENSE_FILE="/usr/local/etc/license.client"
export LM_LICENSE_FILE

# Test various builds (last one is the default build)
configure_opts="--without-MPI --with-strict-checking"
for copt in $configure_opts ""
do
   ./test.sh configure.sh $src_dir $copt --enable-debug
   output_subdir=$output_dir/build$copt
   mkdir -p $output_subdir
   mv -f configure.??? $output_subdir
   ./test.sh make.sh $src_dir test
   mv -f make.??? $output_subdir
done

# Test link for C++
cd $src_dir/test
make clean
make all++ 1> $output_dir/link-c++.out 2> $output_dir/link-c++.err
cd $test_dir

# Test link for Fortran
cd $src_dir/test
make clean
make all77 1> $output_dir/link-f77.out 2> $output_dir/link-f77.err
cd $test_dir

# Test examples

# Test runtest tests
./test.sh default.sh $src_dir
mv -f default.??? $output_dir

# Filter misleading error messages
for errfile in $( find $output_dir -name "*.err*" )
do
  for filter in \
      '[A-Za-z0-9_]*\.[cCf]$'\
      '[A-Za-z0-9_]*\.cxx$'\
      '^[ \t]*$'\
      '^[0-9]*\ Lines\ Compiled$'\
      'autoconf\ has\ been\ disabled'\
      'automake\ has\ been\ disabled'\
      'autoheader\ has\ been\ disabled'\
      'ltdl.c:'\
      'sidl'\
      'cpu\ clock'\
      'wall\ clock'\
      'queued'\
      'allocated'\
      '\ remark:\ '
  do
    if (egrep "$filter" $errfile > /dev/null) ; then
	mv $errfile $errfile.tmp
	egrep -v "$filter" $errfile.tmp > $errfile
	echo "-- applied filter:$filter" >> $errfile.orig
	cat $errfile.tmp >> $errfile.orig
    fi
  done
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir -not -empty -and -name "*.err*" )
do
   echo $errfile >&2
done
