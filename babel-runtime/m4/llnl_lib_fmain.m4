
dnl 
dnl @synopsis LLNL_LIB_FMAIN
dnl 
dnl Finds the "main" function if the driver is written in fortran
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
dnl
AC_DEFUN([LLNL_LIB_FMAIN],
[AC_REQUIRE([AC_PROG_F77])dnl
AC_CACHE_CHECK(if $CC linker needs a special library for $F77 main, llnl_lib_fmain, [
echo "      END" > conftest.f
foutput=`${F77} -v -o conftest conftest.f 2>&1`
xlf_p=`echo $foutput | grep xlfentry`
if test -n "$xlf_p"; then
  foutput=`echo $foutput | sed 's/,/ /g'`
fi
fmain=no
for arg in $foutput; do
  case "$arg" in
    *for_main.o)
      if test -e $arg; then 
        found=true
        fmain="$arg"
      fi
    ;;
  esac
done
llnl_lib_fmain="$fmain"
if test "X$llnl_lib_fmain" != "Xno" ; then 
  FMAIN="$llnl_lib_fmain"
else
  FMAIN=
fi
rm -f conftest.f conftest
])
AC_SUBST(FMAIN)
])
