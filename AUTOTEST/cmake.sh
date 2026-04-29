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
   optionally runs driver and example tests.

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
cd $root_dir
root_dir=`pwd`

# Clean up the build directories (do it from root_dir as a precaution)
cd $root_dir
rm -fr build/*

# Clean up the previous install
cd $root_dir
rm -fr src/hypre

# Configure
cd $root_dir/build
eval cmake $copts ../src

# Accept historical make-style build options such as "-j test" and map them
# to cmake --build syntax. Always build the default target first so the later
# install step has all requested binaries available, then run any explicit
# targets such as check/test.
build_opts=()
build_targets=()

if [ -n "$mopts" ]; then
   eval "set -- $mopts"
   while [ "$#" -gt 0 ]; do
      case $1 in
         -j|--parallel)
            if [ "$#" -gt 1 ] && [[ $2 =~ ^[0-9]+$ ]]; then
               build_opts+=(--parallel "$2")
               shift 2
            else
               build_opts+=(--parallel)
               shift
            fi
            ;;
         --target|-t)
            shift
            while [ "$#" -gt 0 ] && [[ $1 != -* ]]; do
               if [ "$1" != "all" ]; then
                  build_targets+=("$1")
               fi
               shift
            done
            ;;
         all)
            shift
            ;;
         -*)
            build_opts+=("$1")
            shift
            ;;
         *)
            build_targets+=("$1")
            shift
            ;;
      esac
   done
fi

build_cmd=(cmake --build . "${build_opts[@]}")
"${build_cmd[@]}"

if [ ${#build_targets[@]} -gt 0 ]; then
   target_build_cmd=(cmake --build . "${build_opts[@]}" --target "${build_targets[@]}")
   "${target_build_cmd[@]}"
fi

eval cmake --install .

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
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

# Clean up
cd $root_dir
rm -fr build/*
rm -fr src/hypre
( cd $root_dir/src/test; rm -f $drivers; ./cleantest.sh )
