
dnl 
dnl @synopsis LLNL_LIB_FCMAIN
dnl 
dnl Finds the "main" function if the driver is written in fortran
dnl
dnl @version 
dnl @author 
dnl
dnl Note:  Clone of F77 version but tailored to pgf90 needs.
dnl

AC_DEFUN([LLNL_LIB_FCMAIN],
[AC_REQUIRE([AC_PROG_FC])dnl
AC_REQUIRE([LLNL_F90_LIBRARY_LDFLAGS])
AC_CACHE_CHECK(if $CC linker needs a special library for $FC main, llnl_lib_f90main, [
echo "END" > conftest.f90
foutput=`${FC} ${ac_cv_prog_fc_v} -o conftest conftest.f90 2>&1`
fmain=`echo $foutput | grep f90main`
if test -n "$fmain"; then
  foutput=`echo $foutput | sed 's/,/ /g'`
fi
f90main=no
for arg in $foutput; do
  case "$arg" in
    *f90main.o)
      if test -e $arg; then 
        found=true
        f90main="$arg"
      fi
   ;;
    *fj90rt0.o)
      if test -e $arg; then
	found=true
	f90main="$arg"
      fi
    ;;
    *for_main.o)
      if test -e $arg; then
	found=true
	f90main="$arg"
      fi
    ;;
  esac
done
llnl_lib_f90main="$f90main"
if test "X$llnl_lib_f90main" != "Xno" ; then 
  FCMAIN="$llnl_lib_f90main"
else
  FCMAIN=
fi
rm -f conftest.f90 conftest
])
AC_SUBST(FCMAIN)
])
