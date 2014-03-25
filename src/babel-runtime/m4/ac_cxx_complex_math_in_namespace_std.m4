

dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_complex_math_in_namespace_std.m4***
dnl @synopsis AC_CXX_COMPLEX_MATH_IN_NAMESPACE_STD
dnl
dnl If the C math functions are in the cmath header file and std:: namespace,
dnl define HAVE_MATH_FN_IN_NAMESPACE_STD.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_COMPLEX_MATH_IN_NAMESPACE_STD],
[AC_CACHE_CHECK(whether complex math functions are in std::,
ac_cv_cxx_complex_math_in_namespace_std,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <complex>
namespace S { using namespace std;
              complex<float> pow(complex<float> x, complex<float> y)
              { return std::pow(x,y); }
            };
]], [[using namespace S; complex<float> x = 1.0, y = 1.0; S::pow(x,y); return 0;]])],[ac_cv_cxx_complex_math_in_namespace_std=yes],[ac_cv_cxx_complex_math_in_namespace_std=no])
 AC_LANG_POP([])
])
if test "$ac_cv_cxx_complex_math_in_namespace_std" = yes; then
  AC_DEFINE(HAVE_COMPLEX_MATH_IN_NAMESPACE_STD,,
            [define if complex math functions are in std::])
fi
])


