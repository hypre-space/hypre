#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

# Setup
testing_dir=`cd ..; pwd`
autotest_dir="$testing_dir/AUTOTEST"
finished_dir="$testing_dir/AUTOTEST-FINISHED"
output_dir="$testing_dir/AUTOTEST-`date +%Y.%m.%d-%a`"
src_dir="$testing_dir/hypre/src"
remote_dir="test-hypre"
summary_file="SUMMARY.html"
summary_subject="Autotest Error Summary `date +%Y-%m-%d`"
email_list="rfalgout@llnl.gov, tzanio@llnl.gov, umyang@llnl.gov, schroder2@llnl.gov, oseikuffuor1@llnl.gov, wang84@llnl.gov, li50@llnl.gov"

# Main loop
test_opts=""
while [ "$*" ]
do
   case $1 in
      -h|-help)
         cat <<EOF

   $0 [options] [-checkout | -dist M.mm.rr | -{test1} ... | -summary]

   where:

      -checkout           Checks out the repository and updates the current AUTOTEST
                          directory.  Should be called before running tests.
      -dist               Use the hypre release M.mm.rr (e.g. 2.4.0b). This is an
                          alternative to -checkout and is used by testdist.sh.
      -{test}             Runs the indicated tests in sequence, which are associated
                          with specific machine names (e.g., -tux149, -alc, -up).
      -summary            Generates a summary file of passed, pending, failed tests.
      -summary-email      Same as -summary, but also sends developers an email.
      -summary-copy {dir} Same as -summary, but also copies to remote test {dir}.

   with options:

      -h|-help       prints this usage information and exits
      -t|-trace      echo each command

   The main purpose of this script is to organize the automatic testing process
   and to ensure that all related files have the appropriate permissions.

   Example usage: $0 -checkout
                  $0 -tux149 -alc
                  $0 -summary
                  $0 -summary-copy tux149:/usr/casc/hypre/testing

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
         if [ ! -d hypre ]; then
            echo "Clone the hypre directory in $testing_dir first"
            exit
         else
            cd hypre; git checkout .; git pull; cd ..
         fi
         trap "cp -fR $testing_dir/hypre/AUTOTEST $testing_dir" EXIT
         test_opts=""
         break
         ;;

     -dist)
         shift
         finished_dir="$testing_dir/AUTOTEST-hypre-$1"
         src_dir="$testing_dir/hypre-$1/src"
         remote_dir="test-hypre-$1"
         shift
         ;;

      # Generate a summary file in the output directory
      -summary*)
         # move the finished logs to todays output directory
         # (using 'cp' then 'rm' produces fewer complaints than using 'mv')
         # (the autotest-* files are removed below if not pending)
         # (check first that the files exist to reduce error messages from 'cp')
         mkdir -p $output_dir
         count=$( find $finished_dir -mindepth 1 -name "*" | wc -m )
         if [ $count -ne 0 ]; then
            cp -fr $finished_dir/* $output_dir
            rm -fr $finished_dir/*
         fi
         count=$( find $autotest_dir -mindepth 1 -name "autotest-*" | wc -m )
         if [ $count -ne 0 ]; then
            cp -f  $autotest_dir/autotest-* $output_dir
         fi

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
         for test in $( find . -name "autotest-*-start" )
         do
            testbase=`basename $test -start`
            if [ ! -e $testbase-done ]; then
               echo $testbase | sed {s/autotest//g} >> $output_dir/$summary_file
            else
               rm -f $autotest_dir/$testbase*
            fi
         done

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

         # keep a time stamp of last runs and report if more than 10 days
         echo ""           >> $summary_file;
         echo "[LAST RUN]" >> $summary_file
         for test in $( find . -maxdepth 1 -name "autotest-*-done" )
         do
            testname=`basename $test -done`
            testname="${testname#autotest-}"
            touch $testing_dir/lastrun-$testname
         done
         for test in $( find $testing_dir -maxdepth 1 -name "lastrun-*" -atime +10 )
         do
            testdate=`ls -l $test | awk '{print $6" "$7" "$8}'`
            testname=`basename $test`
            testname="${testname#lastrun-}"
            echo "-$testname  $testdate" >> $summary_file
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

         if [ "$1" = "-summary-email" ]; then
            # send the email
            (
               echo To: $email_list
               echo Subject: $summary_subject
               echo Content-Type: text/html
               echo MIME-Version: 1.0

               cat $summary_file

            ) | /usr/sbin/sendmail -t
         fi

         if [ "$1" = "-summary-copy" ]; then
            # copy output_dir files to the specified remote testing_dir
            rem_finished_dir="$2/AUTOTEST-FINISHED"
            scp -q -r * $rem_finished_dir
         fi

         test_opts=""
         break
         ;;

      *)
         test_opts="$test_opts $1"
         shift
         ;;
   esac
done

# Ensure that important directories exist
if [ -n "$test_opts" ]; then
   cd $testing_dir
   mkdir -p $autotest_dir
   mkdir -p $finished_dir
   cd $autotest_dir
fi

# Run tests
for opt in $test_opts
do
   # TODO: use a "-<testname>:<hostname>" format to avoid this?
   case $opt in
      -tux[0-9]*-compilers)
         host=`echo $opt | awk -F- '{print $2}'`
         name="tux-compilers"
         ;;

      -tux[0-9]*)
         host=`echo $opt | awk -F- '{print $2}'`
         name="tux"
         ;;

      -mac)
         host="parsol"
         name="mac"
         ;;

      *)
         host=`echo $opt | awk -F- '{print $2}'`
         name=$host
         ;;
   esac

   if [ ! -e autotest-$name-start ]; then
      echo "Test [machine-$name] started at  `date +%T` on `date +%D`" \
         >> autotest-$name-start
      ./testsrc.sh $src_dir $host:$remote_dir/$host machine-$name.sh
      echo "Test [machine-$name] finished at `date +%T` on `date +%D`" \
         >> autotest-$name-start
      mv machine-$name.??? $finished_dir
      touch autotest-$name-done
   fi
done

# Fix permissions
cd $testing_dir
ch_dirs="hypre $autotest_dir $finished_dir $output_dir"
for dir in $ch_dirs lastrun-*
do
   if [ -e $dir ]; then
      chmod -fR a+rX,ug+w,o-w $dir
      # chgrp -fR hypre         $dir
   fi
done

# move all but the last 10 autotest results into yearly subdirectories
files=`echo AUTOTEST-2*.*`
count=`echo $files | wc | awk '{print $2}'`
for i in $files
do
   if [ $count -le 10 ]; then
      break;
   fi
   dir=`echo $i | awk -F '.' '{print $1}'`
   if [ ! -d $dir ]; then
      mkdir $dir
      chmod -fR a+rX,ug+w,o-w $dir
      # chgrp -fR hypre         $dir
   fi
   mv $i $dir/$i
   count=`expr $count - 1`
done

