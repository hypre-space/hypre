#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2007, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************

testname=`basename $0 .sh`

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   **** Only run this script on the uP machine ****

   $0 [-h|-help] {src_dir}

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script runs a number of tests suitable for the uP machine.

   Example usage: $0 ..

EOF
      exit
      ;;
esac

# Setup
test_dir=`pwd`
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=$1
shift

# Test runtest tests
./test.sh default.sh $src_dir
mv -f default.??? $output_dir

# Test linking for different languages
link_opts="all++ all77"
for opt in $link_opts
do
   output_subdir=$output_dir/link$opt
   mkdir -p $output_subdir
   ./test.sh link.sh $src_dir $opt
   mv -f link.??? $output_subdir
done

# Test other builds
configure_opts="--enable-debug"
for opt in $configure_opts
do
   ./test.sh configure.sh $src_dir $opt 
   output_subdir=$output_dir/build$opt
   mkdir -p $output_subdir
   mv -f configure.??? $output_subdir
   ./test.sh make.sh $src_dir test
   mv -f make.??? $output_subdir
done

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir ! -size 0 -name "*.err" )
do
   echo $errfile >&2
done

# Set some environment variables
# PATH=/usr/local/tools/KCC/kcc4.0f18/KCC_BASE/bin
# PATH=$PATH:/usr/local/tools/guide.assure/guide40.31/bin:/usr/java130/bin
# PATH=$PATH:/usr/local/bin:/usr/bin:/usr/sbin:/usr/ucb
# PATH=$PATH:/usr/bin/X11:/usr/local/totalview/bin:/usr/local/gnu/bin
# PATH=$PATH:/usr/local/scripts:/usr/apps/bin
# PATH=/opt/freeware/bin:$PATH
# PATH=$PATH:.
# export PATH
# LD_LIBRARY_PATH=`pwd`/../hypre/lib
# export LD_LIBRARY_PATH
# MP_RMPOOL=0
# MP_CPU_USE=unique
# MP_EUIDEVICE=css0
# MP_EUILIB=us
# MP_RESD=yes
# MP_HOSTFILE=NULL
# MP_LABELIO=yes
# MP_INFOLEVEL=1
# MP_RETRY=60
# MP_RETRYCOUNT=10
# export MP_RMPOOL MP_CPU_USE MP_EUIDEVICE MP_EUILIB MP_RESD
# export MP_HOSTFILE MP_LABELIO MP_INFOLEVEL
# export MP_RETRY MP_RETRYCOUNT
