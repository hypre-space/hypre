#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

currentdir=`pwd`
currentdir=`basename $currentdir`
if [ "$currentdir" != "src" ]; then
  echo "ERROR: Run this script from the 'src' directory (i.e., 'config/update.sh')."
  exit
fi

source config/version.sh

# Check that the version number is not smaller than before
currentnum=`grep "HYPRE_NUMBER=" configure | cut -d= -f 2`
if [ $hypre_number -lt $currentnum ]; then
  echo "ERROR: HYPRE version number is smaller than the current version!"
  exit
elif [ $hypre_number -gt $currentnum ]; then
  echo "HYPRE version number is greater than the current version"
else
  echo "HYPRE version number is the same as the current version"
fi

##### Update release information and configure script for Linux build system

# NOTE: Using '#' as delimiter in sed to allow for '/' in reldate
cat config/configure.in |
sed -e 's#m4_define.*HYPRE_VERS[^)]*#m4_define([M4_HYPRE_VERSION], ['$hypre_version']#' |
sed -e 's#m4_define.*HYPRE_NUMB[^)]*#m4_define([M4_HYPRE_NUMBER],  ['$hypre_number']#'  |
sed -e 's#m4_define.*HYPRE_DATE[^)]*#m4_define([M4_HYPRE_DATE],    ['$hypre_reldate']#' \
> config/configure.in.tmp
mv config/configure.in.tmp config/configure.in

ln -s -f config/configure.in .
rm -rf aclocal.m4 configure autom4te.cache
autoconf --include=config
autoheader configure.in
rm configure.in

cat >> configure <<EOF

mv HYPRE_config.h HYPRE_config.h.tmp
sed 's/FC_FUNC/HYPRE_FC_FUNC/g' < HYPRE_config.h.tmp > HYPRE_config.h
rm -f HYPRE_config.h.tmp

EOF

##### Update release information for CMake build system

# NOTE: Using '#' as delimiter in sed to allow for '/' in reldate
cat CMakeLists.txt |
sed -e 's#set(HYPRE_VERS[^)]*#set(HYPRE_VERSION '$hypre_version'#' |
sed -e 's#set(HYPRE_NUMB[^)]*#set(HYPRE_NUMBER  '$hypre_number'#' |
sed -e 's#set(HYPRE_DATE[^)]*#set(HYPRE_DATE    '$hypre_reldate'#' \
> CMakeLists.txt.tmp
mv CMakeLists.txt.tmp CMakeLists.txt

##### Update release information in documentation

(cd docs; ./update-release.sh)
