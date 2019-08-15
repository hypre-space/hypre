# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


This directory contains scripts for running various tests on the hypre library.
The scripts augment the 'runtest.sh' runtime tests in 'test/TEST_*'.

Every test in this directory may be run manually by developers.  Many of the
scripts are also run as part of the nightly regression testing, currently
developed and maintained in a separate git repository called 'hypre/autotest'.

=====================

Organization:

This directory mainly consists of a number of simple Bourne-shell scripts (the
files with a '.sh' extension).  Except for a few "special scripts" (below), each
represents an individual test written by a hypre developer.  The special scripts
are as follows (note that they are the only scripts with "test" in their names):

1. 'test.sh'       - Used to run individual tests.
2. 'cleantest.sh'  - Used to clean up the output from a test (or tests).
3. 'renametest.sh' - Used to rename the output from a test.

Usage information for every script (special or individual test) can be obtained
by running it with the '-h' option (e.g., 'test.sh -h' or 'make.sh -h').

=====================

Writing tests:

The rules for writing tests are given in the 'test.sh -h' usage information.
When writing tests, keep in mind the design goals below, especially with respect
to simplicity, flexibility, and portability.

To write a new test, just use an existing test (e.g., 'default.sh') as a
template and make the appropriate modifications.  Try not to use the word "test"
in the name of the script so that we can keep the convention of only the special
scripts having this in their names.  Try not to use absolute directory paths in
the script.

=====================

Design goals:

- Minimal limitations on the types of tests that are possible.
- Developers should be able to run the tests manually.
- Minimal dependence on operating system and software tools (for portability).
- Developers should be able to easily add new tests.
- Simplicity and flexibility.
