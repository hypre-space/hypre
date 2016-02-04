
This directory contains scripts for running various tests on the hypre library.
They are run automatically as part of hypre's regression testing, and they are
run manually to test new distributions of hypre before releasing them to the
public.  The scripts augment the 'runtest.sh' runtime tests in 'test/TEST_*'.

Every test in this directory may be run manually by developers without fear of
interfering with the auto-testing, as long as they are not run from within the
auto-testing directory (currently '/usr/casc/hypre/testing').

=====================

Organization:

This directory mainly consists of a number of simple Bourne-shell scripts (the
files with a '.sh' extension).  Except for a few "special scripts" (below), each
represents an individual test written by a hypre developer.  The special scripts
are as follows (note that they are the only scripts with "test" in their names):

1. 'test.sh' - Used to run individual tests locally on a machine.
2. 'testsrc.sh' - Used to run individual tests on a remote machine.
3. 'testdist.sh' - Used to test a new distribution before release.
4. 'autotest.sh' - Usually run in an automatic fashion by 'cron', but may also
                   be run manually by developers (useful for debugging).

Usage information for every script (special or individual test) can be obtained
by running it with the '-h' option (e.g., 'test.sh -h' or 'make.sh -h').

The file 'cronfile' encapsulates the current 'cron' entries for auto-testing.
It is possible (and probable) to have multiple developers running 'cron' jobs as
part of the overall auto-testing.  This needs to be coordinated if the output
files are being written to the global auto-testing directory.

=====================

Writing tests:

The rules for writing tests are given in the 'test.sh -h' usage information.
When writing tests, keep in mind the design goals below, especially with respect
to simplicity, flexibility, and portability.

To write a new test, just use an existing test (e.g., 'default.sh') as a
template and make the appropriate modifications.  Try not to use the word "test"
in the name of the script so that we can keep the convention of only the special
scripts having this in their names.  Try not to use absolute directory paths in
the script.  If in doubt, talk to another developer or send an inquiry to
hypre-support@llnl.gov.

=====================

Design goals:

- Minimal limitations on the types of tests that are possible.
- Developers should be able to run the tests manually.
- Tests should be runable on both the repository and each release.
- Minimal dependence on operating system and software tools (for portability).
- Developers should be able to easily add new tests.
- Simplicity and flexibility.
