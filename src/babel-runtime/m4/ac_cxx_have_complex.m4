

dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_complex.m4***
dnl @synopsis AC_CXX_HAVE_COMPLEX
dnl
dnl If the compiler has complex<T>, define HAVE_COMPLEX.
dnl
dnl @version $Id: ac_cxx_have_complex.m4,v 1.6 2006/08/29 22:29:23 painter Exp $
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_COMPLEX],
[AC_CACHE_CHECK(whether the compiler has complex<T>,
ac_cv_cxx_have_complex,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[complex<float> a; complex<double> b; return 0;]])],[ac_cv_cxx_have_complex=yes],[ac_cv_cxx_have_complex=no])
 AC_LANG_POP([])
])
if test "$ac_cv_cxx_have_complex" = yes; then
  AC_DEFINE(HAVE_COMPLEX,,[define if the compiler has complex<T>])
fi
])


