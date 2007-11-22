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

# Setup
autotest_dir=`cd ..; pwd`
output_dir="$autotest_dir/AUTOTEST-`date +%m.%d.%Y-%a`"
src_dir="$autotest_dir/linear_solvers"
cvs_opts=""
subject="NEW Autotest Error Summary `date +%D`"
email_list="tzanio@llnl.gov"

while [ "$*" ]
   do
   case $1 in
      -h|-help)
cat <<EOF

   $0 [options] [-checkout | -{test1} -{test2} ... | -summary]

   where: -checkout  Checks the repository and updates the global AUTOTEST directory.
                     Should be called in the beginning of the nightly test cycle.
          -{test}    Runs the corresponding test, which is either  associated with a
                     specific machine (-tux, -up, ...), or is something like -docs.
          -summary   Generates a file with currently pending and failed tests and sends it
                     in an email. Should be called at the end of the nightly test cycle.

   with options:
      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   NOTES:
   - organize the directory structures
   - change the hypre group permissions appropriately
   - checkout the repository before calling 'testsrc.sh'
   - create summary report (option -summary)
   - will have arguments such as '-tux', '-up', etc.

   Example usage: $0 -checkout ; $0 -tux ; $0 -summary

EOF
         exit
         ;;
      -t|-trace)
         set -xv
         shift
         ;;

      # Checkout the repository and update the global AUTOTEST directory
      -checkout)
	 cd $autotest_dir
	 rm -fr linear_solvers AUTOTEST
	 cvs -d /home/casc/repository checkout $cvs_opts linear_solvers
	 # cp -R linear_solvers/AUTOTEST .
	 exit
	 ;;

       # Run remote tests
       -alc|-thunder|-up|-ubgl)
	   host=`echo $1 | awk -F- '{print $2}'`
	   mkdir -p $output_dir
	   rm -f $output_dir/autotest$1-done
	   touch $output_dir/autotest$1-start
	   ./testsrc.sh $src_dir $host:test/$host machine-$host.sh
	   mv machine-$host.??? $output_dir
	   touch $output_dir/autotest$1-done
	   shift
	   ;;

       # Run local tests (modifies $src_dir)
       -tux)
	   mkdir -p $output_dir
	   rm -f $output_dir/autotest$1-done
	   touch $output_dir/autotest$1-start
	   ./test.sh machine$1.sh $src_dir
	   mv machine$1.??? $output_dir
	   touch $output_dir/autotest$1-done
	   shift
	   ;;

       # Test documentation (run after -tux)
       -docs)
	   mkdir -p $output_dir
	   rm -f $output_dir/autotest$1-done
	   touch $output_dir/autotest$1-start
	   ./test.sh docs.sh $src_dir
	   mv docs.??? $output_dir
	   touch $output_dir/autotest$1-done
	   shift
	   ;;

      # Generate a summary file in the output directory
      -summary)
	 cd $output_dir
	 echo "" > summary.txt; echo "[PASSED]" >> summary.txt
 	 for test in $( find .  -maxdepth 1 -empty -and -name "*.err" )
 	 do
	   testname=`basename $test .err`
 	   echo "-${testname#machine-}" >> summary.txt
 	 done
	 echo "" >> summary.txt; echo "[PENDING]" >> summary.txt
	 for test in $( find . -name "*-start" )
	 do
	   testname=`echo $test | awk -F- '{print $2}'`
	   [ ! -e autotest-$testname-done ] && echo "-$testname" >> summary.txt
	 done
	 echo "" >> summary.txt; echo "[FAILED]" >> summary.txt
 	 for test in $( find .  -maxdepth 1 -not -empty -and -name "*.err" )
 	 do
	   testname=`basename $test .err`
 	   echo "-${testname#machine-}" >> summary.txt
 	 done
	 echo "" >> summary.txt; echo "[ERROR FILES]" >> summary.txt
	 for test in $( find $output_dir -not -empty -and -name "*.err" | sort -r )
	 do
	   echo "file://$test" >> summary.txt
	 done
	 cat summary.txt | /usr/bin/Mail -s "$subject" $email_list
	 exit
	 ;;

       *)
	   break
	   ;;
   esac
done
