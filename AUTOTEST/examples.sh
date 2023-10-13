#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

testname=`basename $0 .sh`
tests=""

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   $0 [-h] {src_dir} [options] [-rt <options for runtest.sh script>]

   where: {src_dir}     is the hypre source directory
          -<test>       run <test>  (test = default, bigint, maxdim, complex)
          -spack <dir>  compile and link drivers to spack build
          -h|-help      prints this usage information and exits

   This script builds the hypre example codes in {src_dir}/examples and runs the
   example regression tests in test/TEST_examples.

   Example usage: $0 ../src -maxdim

EOF
      exit
      ;;
esac

# Set src_dir
src_dir=`cd $1; pwd`
shift

# Parse the rest of the command line
while [ "$*" ]
do
   case $1 in
      -rt)
         shift
         break
         ;;
      -spack)
         shift; spackdir="$1"; shift
         ;;
      -*)
         tname=`echo $1 | sed 's/-//'`
         tests="$tests $tname"
         shift
         ;;
   esac
done

# If no tests were specified, run default
if [ "$tests" = "" ]; then
   tests="default"
fi

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir

# Run make in the examples directory
cd $src_dir/examples
make clean
mopt=""
if [ -n "$spackdir" ]; then
   mopt="HYPRE_DIR=$spackdir"
fi
for tname in $tests
do
   if [ "$tname" = "gpu" ]; then
      make -j "use_cuda=1" $mopt $tname
   else
      make $mopt $tname
   fi
done

# Run the examples regression test
cd $src_dir/test
for tname in $tests
do
   ./runtest.sh $@ TEST_examples/$tname.sh
done

# Collect all error files from the tests
for errfile in $( find TEST_examples -name "*.err" -o -name "*.fil" -o -name "*.out*" )
do
   mkdir -p $output_dir/`dirname $errfile`
   mv -f $errfile $output_dir/$errfile
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
