#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

testname=`basename $0 .sh`

drivers="ij sstruct struct ams_driver struct_migrate ij_assembly"

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   $0 [-h] {root_dir} [options]

   where: {root_dir}    is the hypre root directory
          -co: <opts>   configuration options
          -mo: <opts>   make options
          -ro: <opts>   call the run script with these options
          -eo: <opts>   call the examples script with these options
          -h|-help      prints this usage information and exits

   This script uses cmake to configure and compile the source in {root_dir}/src, then
   optionally runs driver and example tests. Phase logs and CMake diagnostics are
   saved in cmake.dir.

   Example usage: $0 .. -co -DCMAKE_BUILD_TYPE=Debug -ro: -ij

EOF
      exit
      ;;
esac

# Set root_dir
root_dir=`cd $1; pwd`
shift

# Parse the rest of the command line
copts="-DHYPRE_BUILD_TESTS=ON"
mopts=""
ropts=""
eopts=""
while [ "$*" ]
do
   case $1 in
      -co:)
         opvar="copts"; shift
         ;;
      -mo:)
         opvar="mopts"; shift
         ;;
      -ro:)
         opvar="ropts"; rset="yes"; shift
         ;;
      -eo:)
         opvar="eopts"; eset="yes"; shift
         ;;
      *)
         eval $opvar=\"\$$opvar $1\"
         shift
         ;;
   esac
done

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
set > $output_dir/sh.env
cd $root_dir
root_dir=`pwd`

filter_error_file()
{
   errfile=$1

   if [ -e $test_dir/$testname.filters ] && [ -s $errfile ]; then
      if (egrep -f $test_dir/$testname.filters $errfile > /dev/null) ; then
         original=`dirname $errfile`/`basename $errfile .err`.fil
         echo "This file contains the original $errfile before filtering" \
            > $original
         cat $errfile >> $original
         mv $errfile $errfile.tmp
         egrep -v -f $test_dir/$testname.filters $errfile.tmp > $errfile
         rm -f $errfile.tmp
      fi
   fi
}

save_build_diagnostics()
{
   cd $root_dir/build

   for file in CMakeCache.txt install_manifest.txt \
      CMakeFiles/CMakeOutput.log CMakeFiles/CMakeError.log \
      test/*.err test/*.fil Testing/Temporary/* test/Testing/Temporary/*
   do
      if [ -f $file ]; then
         mkdir -p $output_dir/build/`dirname $file`
         cp -f $file $output_dir/build/$file
      fi
   done
}

run_phase()
{
   phase=$1
   shift

   echo "$*" > $output_dir/$phase.cmd
   eval "$*" > $output_dir/$phase.out 2> $output_dir/$phase.err
   status=$?
   if [ $status != 0 ]; then
      echo "$phase failed with exit code $status" >> $output_dir/$phase.err
   fi
   return $status
}

# Clean up the build directories (do it from root_dir as a precaution)
cd $root_dir
rm -fr build/*

# Clean up the previous install
cd $root_dir
rm -fr src/hypre

# Configure
cd $root_dir/build
run_phase configure cmake $copts ../src
if [ $? = 0 ]; then
   run_phase build cmake --build . -- $mopts
   build_status=$?
else
   build_status=1
   echo "Skipping build because configure failed" > $output_dir/build.out
   touch $output_dir/build.err
fi

save_build_diagnostics

if [ $build_status = 0 ]; then
   run_phase install cmake --install .
else
   echo "Skipping install because build failed" > $output_dir/install.out
   touch $output_dir/install.err
fi

save_build_diagnostics

cd $test_dir

# Run
if [ -n "$rset" ]; then
   ./test.sh run.sh $root_dir/src $ropts
   mv -f run.??? $output_dir
fi

# Examples
if [ -n "$eset" ]; then
   ./test.sh examples.sh $root_dir/src $eopts
   mv -f examples.??? $output_dir
fi

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir -name "*.err" )
do
   filter_error_file $errfile
   if [ -s $errfile ]; then
      echo $errfile >&2
   fi
done

# Clean up
cd $root_dir
rm -fr build/*
rm -fr src/hypre
( cd $root_dir/src/test; rm -f $drivers; ./cleantest.sh )
