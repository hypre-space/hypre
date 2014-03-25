
dnl @synopsis LLNL_PREVENT_CROSS_COMPILATION
dnl  
dnl If compilers fail, they assume cross compilation.
dnl This macro makes turns that assumption to a failure
dnl
dnl @author Gary Kumfert
AC_DEFUN([LLNL_PREVENT_CROSS_COMPILATION],
[if test "X$cross_compiling" = "Xyes"; then
  AC_MSG_ERROR([Compiler installation problem - could not run compilers...])
fi
])
