dnl
dnl Check if the regression tests should be enabled
dnl
dnl
dnl @synopsis LLNL_ENABLE_REGRESSION
dnl
dnl LLNL_ENABLE_REGRESSION defines a conditional that allows the regression
dnl tests to be skipped.
dnl 
dnl @author Tom Epperly
dnl @version $Id$

AC_DEFUN([LLNL_ENABLE_REGRESSION],[
AC_ARG_ENABLE([regression],
    [AC_HELP_STRING([--enable-regression],
        [enable regression test suite @<:@default=yes@:>@])],
	[enable_regression="$enableval"],
	[enable_regression=yes])
  AM_CONDITIONAL(ENABLE_REGRESSION, test "X$enable_regression" != "Xno")
])

AC_DEFUN([LLNL_ENABLE_DOCUMENTATION],[
AC_ARG_ENABLE([documentation],
    [AC_HELP_STRING([--enable-documentation],
        [enable documentation generation @<:@default=yes@:>@])],
	[enable_documentation="$enableval"],
	[enable_documentation=yes])
  AM_CONDITIONAL(ENABLE_DOCUMENTATION, test "X$enable_documentation" != "Xno")
])
