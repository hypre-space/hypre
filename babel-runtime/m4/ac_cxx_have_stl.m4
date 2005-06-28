

dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_stl.m4***
dnl @synopsis AC_CXX_HAVE_STL
dnl
dnl If the compiler supports the Standard Template Library, define HAVE_STL.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_STL],
[AC_CACHE_CHECK(whether the compiler supports Standard Template Library,
ac_cv_cxx_have_stl,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <list>
#include <deque>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[list<int> x; x.push_back(5);
list<int>::iterator iter = x.begin(); if (iter != x.end()) ++iter; return 0;]])],[ac_cv_cxx_have_stl=yes],[ac_cv_cxx_have_stl=no])
 AC_LANG_POP([])
])
if test "$ac_cv_cxx_have_stl" = yes; then
  AC_DEFINE(HAVE_STL,,[define if the compiler supports Standard Template Library])
fi
])


