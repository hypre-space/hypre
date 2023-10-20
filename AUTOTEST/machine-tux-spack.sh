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

   **** This script requires certain external libraries. ****

   $0 [-h|-help] {src_dir}

   where: -h|-help   prints this usage information and exits
          {src_dir}  is the hypre source directory
          

   This script runs a number of external library tests.

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

# OpenMPI limits the number of processes available by default - override
export OMPI_MCA_rmaps_base_oversubscribe=1

# Basic build and run tests

# RDF - Add this later when LOBPCG residual diff issues are resolved
#
# spackspec="hypre@develop+debug~superlu-dist"
# spack install $spackspec
# spack load    $spackspec
# spackdir=`spack location -i $spackspec`
# test.sh basic.sh ../src -co: -mo: -spack $spackdir -eo: -spack $spackdir
# ./renametest.sh basic $output_dir/basic-examples

# Use the develop branch for superlu-dist
superludistspec="superlu-dist@develop"
spackspec="hypre@develop~debug+superlu-dist ^$superludistspec"
spack install $spackspec
spack load    $spackspec
spackdir=`spack location -i $spackspec`
test.sh basic.sh ../src -co: -mo: -spack $spackdir -ro: -superlu
./renametest.sh basic $output_dir/basic-dsuperlu

# Clean-up spack build
spack spec --yaml $spackspec > test.yaml
grep ' hash:' test.yaml | sed -e 's/^.*: /\//' | xargs spack mark -e
spack gc -y
grep ' hash:' test.yaml | sed -e 's/^.*: /\//' | xargs spack mark -i
rm -f test.yaml
spack clean --all
spack uninstall -yR $superludistspec

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done
