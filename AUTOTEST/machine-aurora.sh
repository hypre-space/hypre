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

   **** Only run this script on Aurora nodes                      ****
   **** Successful regressions as of 1/9/2026 with oneAPI 2025.3. ****
   **** Test with:                                                ****
   **** module use /soft/compilers/oneapi/2025.3.0/modulefiles    ****
   **** module load oneapi/public/2025.3.0                        ****
   **** export SYCL_CACHE_PERSISTENT=1                            ****
   **** export SYCL_CACHE_THRESHOLD=0                             ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory

   This script runs a number of tests suitable for the syrah cluster.

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

# Basic build and run tests
mo="-j test"
eo=""

rtol="0.0"
atol="3e-15"

#save=`echo $(hostname) | sed 's/[0-9]\+$//'`
save="aurora"

##########
## SYCL ##
##########

# SYCL with UM in debug mode [ij, struct, sstruct]
# WM: I suppress all warnings for sycl files for now
co="--enable-debug --disable-fpe-trap --with-sycl --enable-unified-memory --with-extra-CFLAGS=\\'-Wno-unused-but-set-variable -Wno-unused-variable -Wno-builtin-macro-redefined -Rno-debug-disables-optimization\\' --with-extra-CUFLAGS=\\'-w\\'"
ro="-ij-gpu -struct -sstruct -rt -save ${save} -script gpu_tile_compact.sh -rtol ${rtol} -atol ${atol}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-sycl-um

##########################################################
# Echo to stderr all nonempty error files in $output_dir #
##########################################################
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

