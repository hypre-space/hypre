#=============================================================================
# struct: Test parallel and blocking by diffing against base "true" 2d case
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.0 > vcpfmgRedBlackGS.testdata

tail -3 vcpfmgRedBlackGS.out.1 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.2 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.3 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.4 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.5 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2


#=============================================================================
# struct: symmetric GS
#=============================================================================

tail -3 vcpfmgRedBlackGS.out.6 > vcpfmgRedBlackGS.testdata

tail -3 vcpfmgRedBlackGS.out.7 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.8 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.9 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.10 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

tail -3 vcpfmgRedBlackGS.out.11 > vcpfmgRedBlackGS.testdata.temp
diff vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp >&2

rm -f vcpfmgRedBlackGS.testdata vcpfmgRedBlackGS.testdata.temp
