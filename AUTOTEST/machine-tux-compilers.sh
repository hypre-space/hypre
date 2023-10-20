#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

testname=`basename $0 .sh`

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   **** Only run this script on one of the tux machines. ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory

   This script runs a number of compiler tests suitable for the tux machines.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=`cd $1; pwd`
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
         # Must first create a link from 'absoftf90' to the Absoft f90 compiler
         # and make sure it is in your path
         export  CC="mpicc  -cc=gcc"
         export CXX="mpiCC  -CC=g++"
         export F77="mpif77 -fc=absoftf90"
         ;;
      -lahey)
         # Must first create a link from 'laheyf95' to the Lahey f95 compiler
         # and make sure it is in your path
         export  CC="mpicc  -cc=gcc"
         export CXX="mpiCC  -CC=g++"
         export F77="mpif77 -fc=laheyf95"
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
