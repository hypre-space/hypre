#=============================================================================
# struct: Test parallel and blocking by diffing against base "true" 2d case
#=============================================================================

tail -3 vdpfmgRedBlackGS.out.0 > vdpfmgRedBlackGS.testdata

tail -3 vdpfmgRedBlackGS.out.1 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.2 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.3 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.4 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.5 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2


#=============================================================================
# struct: symmetric GS
#=============================================================================

tail -3 vdpfmgRedBlackGS.out.6 > vdpfmgRedBlackGS.testdata

tail -3 vdpfmgRedBlackGS.out.7 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.8 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.9 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.10 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

tail -3 vdpfmgRedBlackGS.out.11 > vdpfmgRedBlackGS.testdata.temp
diff vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp >&2

rm -f vdpfmgRedBlackGS.testdata vdpfmgRedBlackGS.testdata.temp
