#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

testname=`basename $0 .sh`

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   **** Only run this script on the arcticus cluster ****

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
save="arcticus"

##########
## SYCL ##
##########


############
## OMP4.5 ##
############

# OMP 4.5 without UM in debug mode [struct]
co="--with-device-openmp --enable-debug --enable-fortran=no --with-extra-CXXFLAGS=\\'-Wno-missing-prototype-for-cc\\' --with-extra-CFLAGS=\\'-Wno-missing-prototype-for-cc\\' CC= CXX="
ro="-struct -rt -save ${host}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-deviceomp-nonum-debug-struct

############
## KOKKOS ##
############

# Kokkos without UM in debug mode [struct]
co="--with-device-openmp --with-kokkos --enable-debug --with-kokkos-include=$KOKKOS_HOME/include --with-kokkos-lib=$KOKKOS_HOME/lib64/libkokkoscore.a --with-cxxstandard=17 --with-extra-CXXFLAGS=\\'-fno-exceptions -D__STRICT_ANSI__\\' --enable-fortran=no CC= CXX="
ro="-struct -rt -save ${host}"
./test.sh basic.sh $src_dir -co: $co -mo: $mo -ro: $ro
./renametest.sh basic $output_dir/basic-kokkos-nonum-debug-struct

##########################################################
# Echo to stderr all nonempty error files in $output_dir #
##########################################################
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

