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

   This script runs a number of compiler tests suitable for the tux machines.

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

# Test other builds (last one is the default build)
compiler_opts="-pgi"
#compiler_opts="-pgi -kai -absoft -lahey"
for opt in $compiler_opts
do
   case $opt in
      -pgi)
         export  CC="mpicc  -cc=pgcc"
         export CXX="mpiCC  -CC=pgCC"
         export F77="mpif77 -fc=pgf77"
         ;;
      -kai)
         export  CC="mpicc  -cc=KCC"
         export CXX="mpiCC  -CC=KCC"
         export F77="mpif77 -fc=g77"
         ;;
      -absoft)
         export  CC="mpicc  -cc=gcc"
         export CXX="mpiCC  -CC=g++"
         export F77="mpif77 -fc=f90"
         ;;
      -lahey)
         export  CC="mpicc  -cc=gcc"
         export CXX="mpiCC  -CC=g++"
         export F77="mpif77 -fc=f95"
         ;;
   esac

   output_subdir=$output_dir/build$opt
   mkdir -p $output_subdir
   ./test.sh configure.sh $src_dir
   mv -f configure.??? $output_subdir
   ./test.sh make.sh $src_dir test
   mv -f make.??? $output_subdir

   # Test linking for different languages
   link_opts="all++ all77"
   for lopt in $link_opts
   do
      output_subsubdir=$output_subdir/link$lopt
      mkdir -p $output_subsubdir
      ./test.sh link.sh $src_dir $lopt
      mv -f link.??? $output_subsubdir
   done
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
