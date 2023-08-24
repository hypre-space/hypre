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

   **** Only run this script on the tioga cluster ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory

   This script runs a number of tests suitable for the tioga cluster.

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
save="tioga"

##########
## HIP  ##
##########

module -q load rocm/5.2.0

# HIP without UM [benchmark, struct, ams]
co="--with-hip --with-MPI-include=${MPICH_DIR}/include --with-MPI-lib-dirs=${MPICH_DIR}/lib --with-MPI-libs=mpi --with-gpu-arch='gfx90a' CC=cc CXX=CC"
ro="-ams -bench -struct -rt -save ${save}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-hip-nonum

#HIP with UM and single precision [no run]
co="--with-hip --enable-unified-memory --enable-single --enable-debug --with-MPI-include=${MPICH_DIR}/include --with-MPI-lib-dirs=${MPICH_DIR}/lib --with-MPI-libs=mpi --with-gpu-arch='gfx90a' CC=cc CXX=CC"
./test.sh basic.sh $src_dir -co: $co -mo: $mo
./renametest.sh basic $output_dir/basic-hip-um-single

# run on CPU
co="--with-hip --with-test-using-host --with-memory-tracker --enable-debug --with-MPI-include=${MPICH_DIR}/include --with-MPI-lib-dirs=${MPICH_DIR}/lib --with-MPI-libs=mpi --with-gpu-arch='gfx90a' CC=cc CXX=CC"
ro="-ij-noilu -ams -struct -sstruct -rt -D HYPRE_NO_SAVED"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-hip-cpu

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
