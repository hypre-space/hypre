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

   $0 [-h|-help] {top_dir}

   where: {top_dir}  is the top-level hypre release directory
          -h|-help   prints this usage information and exits

   This script checks for files without the SPDX license header.

   Example usage: $0 ..

EOF
      exit
      ;;
esac

# Setup
top_dir=`cd $1; pwd`
shift

cd $top_dir

### First check for files that do not have the license, but should

#LicStr='Copyright (c) 1998 Lawrence Livermore'
LicStr='SPDX-License-Identifier: \(Apache-2.0 OR MIT\)'

# Remove these files from the list of all files without 'SPDX'

egrep -LR "$LicStr" . | egrep -v '[.](o|obj|a|filters|pdf|svg|gif|png)$' |
  egrep -v '[.]/[.]git' |
  egrep -v '[.]/.*HYPRE_config[.]h' |
  egrep -v '[.]/src/(blas|lapack)/.*[.]c' |
  egrep -v '[.]/src/examples/docs' |
  egrep -v '[.]/src/test/TEST_.*'    > check-license.files

# Add these file back to the list

egrep -LR "$LicStr" ./src/test/TEST_* |
  egrep '[.](sh|jobs)$'             >> check-license.files

egrep -LR "$LicStr" ./src/test/TEST_* |
  egrep 'TEST_.*/.*[.]in($|[.].*$)' >> check-license.files

# Remove these individual files from the list and echo the result

cat > check-license.remove <<EOF
./check-license.files
./COPYRIGHT
./LICENSE-APACHE
./LICENSE-MIT
./NOTICE
./src/blas/COPYING
./src/cmbuild/README.txt
./src/config/cmake/hypre_CMakeUtilities.cmake
./src/config/compile
./src/config/config.guess
./src/config/config.sub
./src/config/depcomp
./src/config/HYPRE_config.h.in
./src/config/install-sh
./src/config/missing
./src/config/mkinstalldirs
./src/configure
./src/docs/ref-manual/conf.doxygen
./src/docs/usr-manual/Makefile
./src/docs/usr-manual/_static/custom.css
./src/docs/usr-manual/conf.py
./src/docs/usr-manual/zREADME
./src/utilities/cub_allocator.h
./src/lapack/COPYING
./src/nopoe
./src/tarch
./src/test/runtest.valgrind
EOF
egrep -v -f check-license.remove check-license.files >&2
rm -f check-license.remove check-license.files

### Next check for files that should not have the license, but do

# blas and lapack '.c' files should not have an LLNL license
egrep -lR "$LicStr" ./src/blas ./src/lapack | egrep '[.]/src/(blas|lapack)/.*[.]c' >&2
