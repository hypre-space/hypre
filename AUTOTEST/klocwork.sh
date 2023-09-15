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

   $0 [-h] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs the static analysis tool klockwork in {src_dir}.

   Example usage: $0 ../src

EOF
      exit
      ;;
esac

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=`cd $1; pwd`
shift

# configure the code
./test.sh configure.sh $src_dir $@
mv -f configure.??? $output_dir

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

cd $src_dir

# do the static analysis
kwinject -T hypre.trace make
kwinject -t hypre.trace -o hypre.out
mkdir tables
kwbuildproject --license-host swordfish --host rzcereal3 --port 8066 -j 4 --project hypre -o tables hypre.out

# save and check tables/build.log and tables/parse_errors.log files
cp tables/build.log tables/parse_errors.log $output_dir
cat tables/parse_errors.log >&2

# upload the results to the host
kwadmin --host rzcereal3 --port 8066 load hypre tables

# get the list of build names (this assumes that the command 'list-builds'
# returns a reverse chronological listing)
build_list=`kwadmin list-builds hypre`
# # The following explicitly sorts the list, assuming the prefix is 'build_'
# kwadmin list-builds hypre | awk 'BEGIN {FS="_"}; {print $2}' | sort -n -r | awk '{print "build_" $1}'

# generate the list of new issues
build_name=`echo $build_list | awk '{print $1}'`
kwinspectreport --license-host swordfish --host rzcereal3 --port 8066 --text hyprenew.txt --state new --project hypre --build $build_name
cat hyprenew.txt >&2

# Delete all but the most recent 5 builds
count=0
for build in $build_list
do
   count=`expr $count + 1`
   if [ $count -gt 5 ]; then
      kwadmin delete-build hypre $build
   fi
done
