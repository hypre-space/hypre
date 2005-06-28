

dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_numeric_limits.m4***
dnl @synopsis AC_CXX_HAVE_NUMERIC_LIMITS
dnl
dnl If the compiler has numeric_limits<T>, define HAVE_NUMERIC_LIMITS.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_NUMERIC_LIMITS],
[AC_CACHE_CHECK(whether the compiler has numeric_limits<T>,
ac_cv_cxx_have_numeric_limits,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <limits>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[double e = numeric_limits<double>::epsilon(); return 0;]])],[ac_cv_cxx_have_numeric_limits=yes],[ac_cv_cxx_have_numeric_limits=no])
 AC_LANG_POP([])
])
if test "$ac_cv_cxx_have_numeric_limits" = yes; then
  AC_DEFINE(HAVE_NUMERIC_LIMITS,,[define if the compiler has numeric_limits<T>])
fi
])


