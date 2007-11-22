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

   **** Only run this script on one of the tux machines. ****

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs a number of tests suitable for the tux machines.

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
MPICH=/usr/apps/mpich/default/bin:/usr/apps/mpich/1.2.7p1/bin
PARASOFT=/usr/apps/insure++/default
INSURE=/usr/apps/ParaSoft/insure++7.1.0/bin:/usr/apps/insure++/default/bin
PATH=$MPICH:$INSURE:$PATH
export PATH

# Test various builds (last one is the default build)
configure_opts="--with-babel --without-MPI --with-strict-checking"
for copt in $configure_opts ""
do
   ./test.sh configure.sh $src_dir $copt
   output_subdir=$output_dir/build$copt
   mkdir -p $output_subdir
   mv -f configure.??? $output_subdir
   ./test.sh make.sh $src_dir test
   mv -f make.??? $output_subdir
done

# Test link for C++
cd $src_dir/test
make all++ 1> $output_dir/link-c++.out 2> $output_dir/link-c++.err
cd $test_dir

# Test link for Fortran
cd $src_dir/test
make all77 1> $output_dir/link-f77.out 2> $output_dir/link-f77.err
cd $test_dir

# Test examples

# Test runtest tests with debugging and insure turned on
./test.sh debug.sh $src_dir --with-insure
mv -f debug.??? $output_dir

# Filter misleading error messages
for errfile in $( find $output_dir -name "*.err*" )
do
  for filter in \
      'autoconf\ has\ been\ disabled'\
      'automake\ has\ been\ disabled'\
      'autoheader\ has\ been\ disabled'\
      'configure:\ WARNING:\ Configuration\ for'\
      'configure:\ WARNING:\ Skipping\ Java'\
      'cpu\ clock'\
      'wall\ clock'\
      'Error:\ PDF'\
      'babel_config.h.in:\ No\ such\ file\ or\ directory'\
      '../../../babel/runtime'\
      'ltdl.c:'\
      'sidl'\
      'Insure\ messages\ will\ be\ written\ to'\
      'Timeout\ in\ waiting\ for\ processes\ to'\
      'rsh\ program'\
      'problem'\
      'This\ is\ not\ a\ problem'\
      'environment'\
      'process\ termination'\
      'TCA\ log\ data\ will\ be\ merged\ with\ tca'\
      'PGC/x86\ Linux/x86\ 3.3-1:\ compilation\ completed\ with\ warnings'\
      'sed:\ Couldn.t\ close\ '\
      'Warning:\ No\ xauth\ data'
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
