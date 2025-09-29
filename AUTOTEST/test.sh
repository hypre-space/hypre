#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

countJobs=0
rename=0

while [ "$*" ]
do
   case $1 in
      -h|-help)
         cat <<EOF

   $0 [options] {basename}.sh [{basename}.sh args]

    where: {basename} is the user-defined name for the test script

    with options:
       -h|-help             prints this usage information and exits
      -od|-outputdir <val>  output directory used for renaming purposes (with -tn)
      -tn|-testname <val>   label used to identify the test name. Renames
       -t|-trace            echo each command

   This script runs the Bourne shell test '{basename}.sh' and creates output
   files named '{basename}.err' and '{basename}.out' which capture the stderr
   and stdout output from the test.  The test script is run from the current
   directory, which should contain this script, '{test_name}.sh', and any other
   supporting files.

   A test is deemed to have passed when nothing is written to stderr.  A test
   may call other tests.  A test may take arguments, such as directories or
   files.  A test may also create output, which should be collected by the test
   in a directory named '{basename}.dir'.  A test may also require additional
   "filtering" in situations where information is erroneously written to stderr.
   Text identifying lines to be filtered are added to '{basename}.filters'.
   Usage documentation should appear at the top of each test.

   Example usage: $0 configure.sh ../src

EOF
         exit
         ;;
      -t|-trace)
         set -xv
         shift
         ;;
     -od|-outputdir)
         output_dir=$2
         shift 2
         ;;
     -tn|-testname)
         rename=1
         testname=$2
         shift 2
         ;;
      *)
         break
         ;;
   esac
done

# Run the test and capture stdout, stderr
basename=$(basename "$1" .sh)
if [ -n "$testname" ]; then
  # If testname is not an empty string, prepend basename.
  testname="$basename--$testname"
else
  # Otherwise, just use basename.
  testname="$basename"
fi
shift
label="Running test [$testname] "
printf "%s" "$label"
# Fill with dots outside the brackets to a fixed width
leader_width=50
dot_count=$((leader_width - ${#label}))
if [ "$dot_count" -gt 0 ]; then
   printf "%*s" "$dot_count" "" | tr ' ' '.'
fi
SECONDS=0 # Use builtin bash variable for timing
#echo "Args: $@"
./$basename.sh $@ 1>"$basename.out" 2>"$basename.err"
./status.sh $basename.err
hours=$((SECONDS/3600))
mins=$(((SECONDS%3600)/60))
secs=$((SECONDS%60))
jobTotal=-1
if [ "$countJobs" -eq 1 ]; then
   jobTotal=`grep -E '^HYPRE_JOB_TOTAL:' "$basename.out" | tail -n1 | awk '{print $2}'`
fi
if [ "$jobTotal" -gt 0 ]; then
   printf " (jobs: %s, elapsed time: %02dh:%02dm:%02ds)\n" "$jobTotal" "$hours" "$mins" "$secs"
else
   printf " (elapsed time: %02dh:%02dm:%02ds)\n" "$hours" "$mins" "$secs"
fi

# Filter misleading error messages
if [ -e $basename.filters ]; then
    if (egrep -f $basename.filters $basename.err > /dev/null) ; then
       echo "This file contains the original $basename.err before filtering" \
          > $basename.fil
       cat $basename.err >> $basename.fil
       mv $basename.err $basename.tmp
       egrep -v -f $basename.filters $basename.tmp > $basename.err
       rm -f $basename.tmp
    fi
fi

# Rename test?
if [ "$rename" -eq 1 ]; then
    ./renametest.sh $basename $output_dir/$testname
fi
