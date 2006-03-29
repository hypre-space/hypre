#=============================================================================
# sstruct: Test addtovalue routine. Compares the solutions obtained using
# different formations of the matrix- one with setvalues and the other with
# addtovalues
#=============================================================================

tail -3 addtovalues.out.0 > addtovalues.testdata
tail -3 addtovalues.out.1 > addtovalues.testdata.temp
diff addtovalues.testdata addtovalues.testdata.temp >&2

tail -3 addtovalues.out.2 > addtovalues.testdata
tail -3 addtovalues.out.3 > addtovalues.testdata.temp
diff addtovalues.testdata addtovalues.testdata.temp >&2

rm -f addtovalues.testdata addtovalues.testdata.temp
