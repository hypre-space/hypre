#!/bin/ksh 
#=============================================================================
# sstruct_fac: Tests the sstruct_fac solver
#    for each test save the results for comparison with the baseline case
#=============================================================================

tail -3 sstruct_fac.out.0 > fac.testdata
tail -3 sstruct_fac.out.1 > fac.testdata.temp
diff -bI"time" fac.testdata fac.testdata.temp >&2

cat fac.testdata > sstruct_fac.tests
cat fac.testdata.temp >> sstruct_fac.tests
#=============================================================================
tail -3 sstruct_fac.out.2 > fac.testdata
tail -3 sstruct_fac.out.3 > fac.testdata.temp
diff -bI"time" fac.testdata fac.testdata.temp >&2

cat fac.testdata >> sstruct_fac.tests
cat fac.testdata.temp >> sstruct_fac.tests
#=============================================================================
tail -3 sstruct_fac.out.4 > fac.testdata
tail -3 sstruct_fac.out.5 > fac.testdata.temp
diff -bI"time" fac.testdata fac.testdata.temp >&2

cat fac.testdata >> sstruct_fac.tests
cat fac.testdata.temp >> sstruct_fac.tests
#=============================================================================
tail -3 sstruct_fac.out.6 > fac.testdata
tail -3 sstruct_fac.out.7 > fac.testdata.temp
diff -bI"time" fac.testdata fac.testdata.temp >&2

cat fac.testdata >> sstruct_fac.tests
cat fac.testdata.temp >> sstruct_fac.tests
#=============================================================================
tail -3 sstruct_fac.out.8 > fac.testdata
tail -3 sstruct_fac.out.9 > fac.testdata.temp
diff -bI"time" fac.testdata fac.testdata.temp >&2

cat fac.testdata >> sstruct_fac.tests
cat fac.testdata.temp >> sstruct_fac.tests
#=============================================================================
tail -3 sstruct_fac.out.10 > fac.testdata
tail -3 sstruct_fac.out.11 > fac.testdata.temp
diff -bI"time" fac.testdata fac.testdata.temp >&2

cat fac.testdata >> sstruct_fac.tests
cat fac.testdata.temp >> sstruct_fac.tests

#=============================================================================
#  compare with baseline test case
#=============================================================================
diff -bI"time" sstruct_fac.saved sstruct_fac.tests >&2

#=============================================================================
#   remove temporary files
#=============================================================================
rm -f fac.testdata fac.testdata.temp sstruct_fac.tests
