#
# Define a configuration macro to see if VOLATILE (a Lahey F90 extension)
# is available.

AC_DEFUN([LLNL_F90_VOLATILE],
[AC_REQUIRE([AC_PROG_FC])dnl
AC_CACHE_CHECK([whether $F90 supports volatile],
               [llnl_cv_f90_volatile],
[AC_LANG_PUSH(Fortran)dnl
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [
  integer(selected_int_kind(18)) :: tin
  volatile :: tin
])],[
 llnl_cv_f90_volatile=["volatile ::"]
],[
 llnl_cv_f90_volatile=["! no volatile"]
])
AC_LANG_POP(Fortran)dnl
])
AC_DEFINE_UNQUOTED(F90_VOLATILE,$llnl_cv_f90_volatile,
[A macro for making F90 variables volatile (i.e., subject to changes the
compiler cannot predict. This is necessary for array access via
the access method.])
])
