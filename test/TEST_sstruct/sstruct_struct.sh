#!/bin/ksh 
#=============================================================================
# sstruct: Tests the struct solvers called from the sstruct interface
#=============================================================================

tail -3 struct.out.1 > struct.testdata
tail -3 struct.out.201 > struct.testdata.temp
diff struct.testdata struct.testdata.temp >&2

tail -3 struct.out.0 > struct.testdata
tail -3 struct.out.200 > struct.testdata.temp
diff struct.testdata struct.testdata.temp >&2

rm -f struct.testdata struct.testdata.temp
