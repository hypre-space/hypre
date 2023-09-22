#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Echo usage information
case $1 in
   -h|-help)
cat <<EOF

   $0 [-h|-help] [{test_dir}]

   where: {test_dir} is the name of some number of runtest directories
          -h|-help   prints this usage information and exits

   This script checks the error files for the runtest.sh tests in the specified
   runtest directories.  If no directory is specified, it is assumed that the
   script is being run from within the hypre 'test' directory, and all of the
   'TEST_*' directories are checked.

   Returns 0 if all tests pass, 1 if any test fails.

   Example usage: $0 TEST_struct TEST_ij

EOF
   exit
   ;;
esac

RESET=`shopt -p nullglob`  # Save current nullglob setting
shopt -s nullglob          # Return an empty string for failed wildcard matches 
if [ "x$1" = "x" ]
then
   testdirs=`echo TEST*`   # All TEST directories
else
   testdirs=`echo $*`      # Only the specified test directories
fi
$RESET                     # Restore nullglob setting

error_code=0
echo ""
for testdir in $testdirs
do
   files=`find $testdir -name '*.err' | sort`
   if [ -n "$files" ]
   then
      for file in $files
      do
         SZ=`ls -l $file | awk '{print $5}'`
         if [ $SZ != 0 ]
         then
            echo -e "FAILED : $file  ($SZ)\n"
            cat $file
            error_code=1
         else
            echo "    OK : $file"
         fi
      done
      echo ""
   fi
done
exit $error_code
