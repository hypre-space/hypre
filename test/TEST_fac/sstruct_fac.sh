#!/bin/ksh 
#=============================================================================
# sstruct_fac: Tests the sstruct_fac solver
#=============================================================================

tail -3 fac.out.0 > fac.testdata
tail -3 fac.out.1 > fac.testdata.temp
diff fac.testdata fac.testdata.temp >&2

tail -3 fac.out.2 > fac.testdata
tail -3 fac.out.3 > fac.testdata.temp
diff fac.testdata fac.testdata.temp >&2

tail -3 fac.out.4 > fac.testdata
tail -3 fac.out.5 > fac.testdata.temp
diff fac.testdata fac.testdata.temp >&2

tail -3 fac.out.6 > fac.testdata
tail -3 fac.out.7 > fac.testdata.temp
diff fac.testdata fac.testdata.temp >&2

tail -3 fac.out.8 > fac.testdata
tail -3 fac.out.9 > fac.testdata.temp
diff fac.testdata fac.testdata.temp >&2

tail -3 fac.out.10 > fac.testdata
tail -3 fac.out.11 > fac.testdata.temp
diff fac.testdata fac.testdata.temp >&2


rm -f fac.testdata fac.testdata.temp
