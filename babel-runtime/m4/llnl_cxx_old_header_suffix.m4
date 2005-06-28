
dnl 
dnl @synopsis LLNL_CXX_OLD_HEADER_SUFFIX
dnl 
dnl If the CXX compiler requires *.h includes define `LLNL_CXX_OLD_HEADER_SUFFIX'
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
AC_DEFUN([LLNL_CXX_OLD_HEADER_SUFFIX],
[
AC_MSG_CHECKING([if ${CXX} requires requires old .h-style header includes])
AC_CACHE_VAL(llnl_cv_old_cxx_header_suffix,
[
AC_LANG_PUSH([C++])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <iostream>]], [[ using namespace std; cout;]])],[llnl_cv_old_cxx_header_suffix=no],[llnl_cv_old_cxx_header_suffix=yes])
AC_LANG_POP([])
])
AC_MSG_RESULT($llnl_cv_old_cxx_header_suffix)
if test "$llnl_cv_old_cxx_header_suffix" = yes; then
AC_DEFINE(REQUIRE_OLD_CXX_HEADER_SUFFIX,,[define if C++ requires old .h-style header includes])
fi])	
