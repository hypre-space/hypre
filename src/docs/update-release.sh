#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


usrconf="usr-manual/conf.py"
refconf="ref-manual/conf.doxygen"

version=`../utilities/version.sh -version`
reldate=`../utilities/version.sh -date`
if type -p gdate > /dev/null; then
    usrdate=`gdate --date=$reldate +'%B %d, %Y'`;
else
    usrdate=`date --date=$reldate +'%B %d, %Y'`
fi

# User manual
sed -e "s/version = .*/version = '$version'/" $usrconf |
sed -e "s/release = .*/release = '$version'/" |
sed -e "s#today = .*#today = '$usrdate'#" > $usrconf.tmp
mv $usrconf.tmp $usrconf

# Reference manual
sed -e "s/PROJECT_NUMBER .*=.*/PROJECT_NUMBER = $version/" $refconf > $refconf.tmp
mv $refconf.tmp $refconf
