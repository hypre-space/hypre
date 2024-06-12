#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Echo usage information
case $1 in
   -h|-help)
      cat <<EOF

   $0 [-h] {test_name} {new_test_name}

   where: -h|-help      prints this usage information and exits

   This script renames the files/directories associated with a test.

   Example usage: $0 basictest specifictest

EOF
      exit
      ;;
esac

oldname=$1;
newname=$2;

if [ -e $oldname.dir ]; then mv $oldname.dir $newname.dir; fi
if [ -e $oldname.err ]; then mv $oldname.err $newname.err; fi
if [ -e $oldname.out ]; then mv $oldname.out $newname.out; fi
if [ -e $oldname.fil ]; then mv $oldname.fil $newname.fil; fi

# # This code doesn't work when '/' appears in the sed script names
# for i in $oldname.???
# do
#    echo $i $oldname $newname
#    j=`echo $i | sed s/$oldname/$newname/`
#    echo mv $i $j
# done
