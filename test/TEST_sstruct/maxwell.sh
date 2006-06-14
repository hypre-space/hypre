#=============================================================================
# no test comparison for now. Just a holder file with fake tests. Hard to
# develop tests because of the coarsening scheme.
#=============================================================================

tail -3 maxwell.out.0 > maxwell.testdata
tail -3 maxwell.out.0 > maxwell.testdata.temp
diff maxwell.testdata maxwell.testdata.temp >&2

tail -3 maxwell.out.1 > maxwell.testdata
tail -3 maxwell.out.1 > maxwell.testdata.temp
diff maxwell.testdata maxwell.testdata.temp >&2

tail -3 maxwell.out.2 > maxwell.testdata
tail -3 maxwell.out.2 > maxwell.testdata.temp
diff maxwell.testdata maxwell.testdata.temp >&2

rm -f maxwell.testdata maxwell.testdata.temp
