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

   $0 [-h] {src_dir} [options for configure]

   where: {src_dir}  is the hypre source directory
          -h|-help   prints this usage information and exits

   This script performs a debug-enabled configure+make+run test in {src_dir}.

   Example usage: $0 ..

EOF
   exit
   ;;
esac

# Setup
output_dir=`pwd`/$testname.dir
rm -fr $output_dir
mkdir -p $output_dir
src_dir=$1
shift

# Call the configure, make and run scripts
./test.sh configure.sh $src_dir --enable-debug $@
mv -f configure.??? $output_dir

./test.sh make.sh $src_dir test
mv -f make.??? $output_dir

./test.sh run.sh $src_dir
mv -f run.??? $output_dir

# Echo to stderr all nonempty error files in $output_dir
for errfile in $( find $output_dir -not -empty -and -name "*.err*" )
do
   echo $errfile >&2
done
