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
src_dir="$testing_dir/linear_solvers"
cvs_opts=""
summary_file="SUMMARY.html"
summary_subject="NEW Autotest Error Summary `date +%D`"
email_list="rfalgout@llnl.gov, tzanio@llnl.gov, umyang@llnl.gov, abaker@llnl.gov, lee123@llnl.gov, chtong@llnl.gov, panayot@llnl.gov"
# email_list="rfalgout@llnl.gov, tzanio@llnl.gov, umyang@llnl.gov, abaker@llnl.gov, lee123@llnl.gov, chtong@llnl.gov, panayot@llnl.gov"

# Ensure that important directories exist
cd $testing_dir
mkdir -p $autotest_dir
mkdir -p $finished_dir
cd $autotest_dir

# Main loop
test_opts=""
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
         cp -fR linear_solvers/AUTOTEST .
         test_opts=""
         break
         ;;

      # Generate a summary file in the output directory
      -summary)
         # move the finished logs to todays output directory
         # (using 'cp' then 'rm' produces fewer complaints than using 'mv')
         mkdir -p $output_dir
         cp -fr $finished_dir/* $output_dir
         rm -fr $finished_dir/*

         cd $output_dir
         echo "<html>"          > $summary_file;
         echo "<head> </head>" >> $summary_file;
         echo "<PRE>"          >> $summary_file;
         echo $summary_subject >> $summary_file


         # all top-level tests with empty error files are reported as "passed",
         # not including the cron autotest logs
         echo ""         >> $summary_file;
         echo "[PASSED]" >> $summary_file
         for test in $( find . -maxdepth 1 -size 0 -name "*.err" ! -name "*cron*" )
         do
            testname=`basename $test .err`
            echo "-${testname#machine-}" >> $summary_file
         done

         # active tests without a *-done file are reported as "pending"
         echo ""          >> $summary_file;
         echo "[PENDING]" >> $summary_file
         cd $autotest_dir
         for test in $( find . -name "autotest-*-start" )
         do
            testbase=`basename $test -start`
            if [ ! -e $testbase-done ]; then
               echo $testbase | sed {s/autotest//g} >> $output_dir/$summary_file
            else
               mv $testbase* $output_dir
            fi
         done
         cd $output_dir

         # all top-level tests with non-empty error files are reported as "failed",
         # including the cron autotest logs
         echo ""         >> $summary_file;
         echo "[FAILED]" >> $summary_file
         for test in $( find . -maxdepth 1 ! -size 0 -name "*.err" )
         do
            testname=`basename $test .err`
            for prefix in "machine-" "autotest-";
            do
               testname="${testname#$prefix}"
            done
            echo "-$testname" >> $summary_file
         done

         # list all non-empty error files in todays output directory
         echo ""              >> $summary_file;
         echo "[ERROR FILES]" >> $summary_file
         for test in $( find $output_dir ! -size 0 -name "*.err" | sort -r )
         do
            echo "<a href=\"file://$test\">$test</a>" >> $summary_file
         done

         echo "</PRE>"  >> $summary_file;
         echo "</html>" >> $summary_file;

         # send the email
         (
            echo To: $email_list
            echo Subject: $summary_subject
            echo Content-Type: text/html
            echo MIME-Version: 1.0

            cat $summary_file

         ) | /usr/sbin/sendmail -t

         test_opts=""
         break
         ;;

      *)
         test_opts="$test_opts $1"
         shift
         ;;
   esac
done

# Run tests
for opt in $test_opts
do
   case $opt in
      -tux[0-9]*-compilers)
         host=`echo $opt | awk -F- '{print $2}'`
         name="tux-compilers"
         ;;

      -tux[0-9]*)
         host=`echo $opt | awk -F- '{print $2}'`
         name="tux"
         ;;

      -alc|-thunder|-up|-zeus)
         host=`echo $opt | awk -F- '{print $2}'`
         name=$host
         ;;

      -mac)
         host="kolev-mac"
         name="mac"
         ;;
   esac

   if [ ! -e autotest-$name-start ]; then
      echo "Test [machine-$name] started at  `date +%T` on `date +%D`" \
         >> autotest-$name-start
      ./testsrc.sh $src_dir $host:hypre/testing/$host machine-$name.sh
      echo "Test [machine-$name] finished at `date +%T` on `date +%D`" \
         >> autotest-$name-start
      mv machine-$name.??? $finished_dir
      touch autotest-$name-done
   fi
done

# Fix permissions
cd $testing_dir
ch_dirs="linear_solvers $autotest_dir $finished_dir"
if [ -e $output_dir ]; then
   ch_dirs="$ch_dirs $output_dir"
fi
chmod -fR a+rX,ug+w,o-w $ch_dirs
chgrp -fR hypre         $ch_dirs
