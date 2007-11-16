
This will be a README file describing the structure of this AUTOTEST directory
and the scripts within.  Some of the below design stuff may appear here...


###############################

This file outlines design changes to autotest to make it more flexible,
maintainable, and extensible.  The current system is not too far away from what
we want (it's at least headed in the right direction), but there are a number of
limitations and issues that this redesign will hopefully address.

=====================

Goals:

- Minimal limitations on the types of tests that are possible.
- Developers should be able to run the tests manually (as with 'runtest.sh').
- Tests should be runable on both the repository and each release.
- Minimal dependence on operating system and software tools (for portability).
- Developers should be able to easily add new tests.
- Simplicity and flexibility.

=====================

Design Overview:

All of the scripts will be written in the Bourne shell for portability.

There will be three main scripts associated with autotest (the names of these
scripts are *all* up for negotiation):

1. 'runtest.sh' - Runs a number of mpi jobs and associated tests (the tests are
   written by the developers).  Can be run manually and is smart enough to deal
   with batch systems.  This script already exists and probably won't change.

2. 'test.sh' - Runs Bourne shell tests (the tests are written by the developers,
   and may in turn call 'runtest.sh').  Can be run manually.  The current
   'autotest_test' script serves a similar role, but needs to be rewritten.

3. 'autotest.sh' - Called by the cron daemon to run nightly regression tests and
   organize the results in some directory structured that is shared and viewable
   by the hypre development team.  As part of this process, the script ensures
   that the results have the right group permissions.  This script should not be
   called directly by developers.

More detailed documentation on each of these scripts is given below.

There will also be a fourth script (currently called 'autotest_create_summary')
run by cron that checks the results of the nightly autotest runs and creates a
summary email that is sent to the hypre developers.

=====================

Script 'runtest.sh':

usage:

  runtest.sh [options] {testpath}/{testname}.sh ...
 
  where: {testpath} is the directory path to the test script (and helper files)
         {testname} is a user-defined name for the test script
 
  with options:
     -h|-help    prints usage information and exits
     -n|-norun   turn off execute mode, echo what would be run
     -t|-trace   echo each command
     -D <var>    define <var> when running tests

This script runs the MPI jobs specified in '{testpath}/{testname}.jobs', then
runs the test script '{testname}.sh' to test the output from the mpi jobs in
some way.  More than one test may be specified on the command line.  The two
{testname} files must both be present.  The format of these files is such that
they may be run as stand-alone scripts in some environments.

This script first copies the needed MPI codes from the current directory to the
directory specified by '{testpath}', then runs the MPI jobs and associated test
script from within the '{testpath}' directory.  Hence, other supporting files
may be located here.  Also, all output is written here and should use naming
conventions that distinguish it from other tests.

A test is deemed to have passed when nothing is written to stderr.  This script
creates an output file named '{testname}.err' which captures the stderr output.
Some post-filtering is also done to remove erroneous stderr output that is
sometimes unavoidable (e.g., insure reports a number of MPI errors that are not
really bugs, so we filter these out).  Filtering should be minimized as much as
possible!

This script knows about the LC machines and batch system, and is able to submit
the MPI jobs in the '.jobs' files through the batch system.  There is also an
additional feature of the '.jobs' file format that allows individual jobs to be
explicitly grouped together in a single batch script.

Questions: 

The current 'runtest.sh' documentation states that output files should have a
specific naming convention.  Is this true?  Is it necessary?

=====================

Script 'test.sh':

usage:

  test.sh [options] {testpath}/{testname}.sh [{testname}.sh arguments]
 
  where: {testpath} is the directory path to the test script (and helper files)
         {testname} is the user-defined name for the test script
 
  with options:
     -h|-help    prints usage information and exits
     -t|-trace   echo each command

This script runs the Bourne shell test '{testname}.sh' and creates an output
file named '{testname}.err' which captures the stderr output from the test.  The
test script is run from within the directory specified by '{testpath}'.

A test is deemed to have passed when nothing is written to stderr.  A test may
call other tests.  A test may take arguments.  Note that care should be taken
with some types of arguments such as files or directories since scripts are run
from within the '{testpath}' directory.  In these cases, it is best to require
absolute path names.  A test may also create output.  It is recommended that all
output be collected by the test in a directory named '{testname}.out'.  Usage
documentation should appear at the top of each test.

=====================

Script 'autotest.sh':

More stuff here...
Must change group permissions.

=====================

Organization of autotest results:

More stuff here...

=====================

Write some examples and plan an initial suite of tests...
