#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


usrconf="usr-manual/conf.py"
refconf="ref-manual/conf.doxygen"

version=`../utilities/version -number`
reldate=`../utilities/version -date`
usrdate=`date --date=$reldate +'%B %d, %Y'`

# User manual
sed -e 's/version = .*/version = \x27'$version'\x27/' $usrconf |
sed -e 's/release = .*/release = \x27'$version'\x27/' |
sed -e 's#today = .*#today = \x27'"$usrdate"'\x27#' > $usrconf.tmp
mv $usrconf.tmp $usrconf

# Reference manual
sed -e 's/PROJECT_NUMBER .*=.*/PROJECT_NUMBER = '$version'/' $refconf > $refconf.tmp
mv $refconf.tmp $refconf
