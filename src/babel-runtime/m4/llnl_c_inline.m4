dnl @synopsis LLNL_C_HAS_INLINE
dnl
dnl Define SIDL_C_HAS_INLINE if the C compiler supports the inline keyword
dnl
dnl @author Tom Epperly
AC_DEFUN([LLNL_C_HAS_INLINE],
[AC_REQUIRE([AC_C_INLINE])dnl
if test "$ac_cv_c_inline" != no; then
  AC_DEFINE(SIDL_C_HAS_INLINE,1,
	[Define to 1 if the C compiler supports inline.])
fi
])
