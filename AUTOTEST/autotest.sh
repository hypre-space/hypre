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
testing_dir=`cd ..; pwd`
autotest_dir="$testing_dir/AUTOTEST"
finished_dir="$testing_dir/AUTOTEST-FINISHED"
output_dir="$testing_dir/AUTOTEST-`date +%Y.%m.%d-%a`"
cvs_opts=""
src_dir="$testing_dir/linear_solvers"
subject="NEW Autotest Error Summary `date +%D`"
email_list="rfalgout@llnl.gov"
# email_list="rfalgout@llnl.gov, tzanio@llnl.gov, umyang@llnl.gov, abaker@llnl.gov, lee123@llnl.gov, chtong@llnl.gov, panayot@llnl.gov"

# Ensure that important directories exist
cd $testing_dir
mkdir -p $autotest_dir
mkdir -p $finished_dir
cd $autotest_dir

# Main loop
while [ "$*" ]
do
   case $1 in
      -h|-help)
         cat <<EOF

   $0 [options] [-checkout | -{test1} -{test2} ... | -summary]

   where: -checkout  Checks out the repository and updates the current AUTOTEST
                     directory.  Should be called before running tests.
          -{test}    Runs the indicated tests in sequence, which are associated
                     with specific machine names (e.g., -tux149, -alc, -up).
          -summary   Generates a summary file with currently pending and failed
                     tests and sends it to developers in an email.

   with options:

      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   The main purpose of this script is to organize the automatic testing process
   and to ensure that all related files have the appropriate permissions.

   Example usage: $0 -checkout
                  $0 -tux149
                  $0 -summary

EOF
         exit
         ;;
      -t|-trace)
         set -xv
         shift
         ;;

      # Checkout the repository and update the global AUTOTEST directory
      -checkout)
         cd $testing_dir
         rm -fr linear_solvers
         cvs -d /home/casc/repository checkout $cvs_opts linear_solvers
         cp -fRp linear_solvers/AUTOTEST .
         break
         ;;

      # Run local tests
      -tux*)
         if [ ! -e autotest-tux-start ]; then
            host=`echo $1 | awk -F- '{print $2}'`
            echo "Test [machine-tux] started at  `date +%T` on `date +%D`" \
               >> autotest-tux-start
            ./testsrc.sh $src_dir $host:hypre/testing/$host machine-tux.sh
            echo "Test [machine-tux] finished at `date +%T` on `date +%D`" \
               >> autotest-tux-start
            mv machine-tux.??? $finished_dir
            touch autotest-tux-done
         fi
         shift
         ;;

      # Run remote tests
      -alc|-thunder|-up|-zeus)
         if [ ! -e autotest$1-start ]; then
            host=`echo $1 | awk -F- '{print $2}'`
            echo "Test [machine-$host] started at  `date +%T` on `date +%D`" \
               >> autotest-$host-start
            ./testsrc.sh $src_dir $host:hypre/testing/$host machine-$host.sh
            echo "Test [machine-$host] finished at `date +%T` on `date +%D`" \
               >> autotest-$host-start
            mv machine-$host.??? $finished_dir
            touch autotest-$host-done
         fi
         shift
         ;;

      # Run tests on a Mac
      -mac)
         if [ ! -e autotest-mac-start ]; then
            host="kolev-mac"
            echo "Test [machine-mac] started at  `date +%T` on `date +%D`" \
               >> autotest-mac-start
            ./testsrc.sh $src_dir $host:hypre/testing/$host machine-mac.sh
            echo "Test [machine-mac] finished at `date +%T` on `date +%D`" \
               >> autotest-mac-start
            mv machine-mac.??? $finished_dir
            touch autotest-mac-done
         fi
         shift
         ;;

      # Generate a summary file in the output directory
      -summary)
         # move the finished logs to todays output directory
         # (using 'cp' then 'rm' produces fewer complaints than using 'mv')
         mkdir -p $output_dir
         cp -fr $finished_dir/* $output_dir
         rm -fr $finished_dir/*

         # all top-level tests with empty error files are reported as "passed",
         # not including the cron autotest logs
         cd $output_dir
         echo "" > Summary.txt; echo "[PASSED]" >> Summary.txt
         for test in $( find . -maxdepth 1 -size 0 -name "*.err" ! -name "*cron*" )
         do
            testname=`basename $test .err`
            echo "-${testname#machine-}" >> Summary.txt
         done

         # active tests without a *-done file are reported as "pending"
         echo "" >> Summary.txt; echo "[PENDING]" >> Summary.txt
         cd $autotest_dir
         for test in $( find . -name "*-start" )
         do
            testname=`echo $test | awk -F- '{print $2}'`
            if [ ! -e autotest-$testname-done ]; then
               echo "-$testname" >> $output_dir/Summary.txt
            else
               mv autotest-*$testname* $output_dir
            fi
         done

         # all top-level tests with non-empty error files are reported as "failed",
         # including the cron autotest logs
         cd $output_dir
         echo "" >> Summary.txt; echo "[FAILED]" >> Summary.txt
         for test in $( find . -maxdepth 1 ! -size 0 -name "*.err" )
         do
            testname=`basename $test .err`
            for prefix in "machine-" "autotest-";
            do
               testname="${testname#$prefix}"
            done
            echo "-$testname" >> Summary.txt
         done

         # list all non-empty error files in todays output directory
         echo "" >> Summary.txt; echo "[ERROR FILES]" >> Summary.txt
         for test in $( find $output_dir ! -size 0 -name "*.err" | sort -r )
         do
            echo "file://$test" >> Summary.txt
         done

         # send the email
         cat Summary.txt | /usr/bin/Mail -s "$subject" $email_list

         break
         ;;

      *)
         shift
         ;;
   esac
done

# Fix permissions
cd $testing_dir
chmod -fR a+rX,ug+w,o-w linear_solvers $autotest_dir $finished_dir $output_dir
chgrp -fR hypre         linear_solvers $autotest_dir $finished_dir $output_dir
