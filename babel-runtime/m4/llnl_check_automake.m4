
dnl
dnl @synopsis LLNL_CHECK_AUTOMAKE(VERSION)
dnl
dnl Check whether automake is the specified version.
dnl
dnl @author Gary Kumfert
AC_DEFUN([LLNL_CHECK_AUTOMAKE],
[
  AC_MSG_CHECKING(for automake/aclocal version)
  changequote(,)
  llnl_automake_version=`automake --version 2>/dev/null | sed '1s/.* \([^ ]*\)$/\1/g;1q'`
  changequote([,])
  if test "X$1" = "X$llnl_automake_version"; then
    AC_MSG_RESULT([$llnl_automake_version (enabled)])
  else
    AUTOMAKE="$SHELL $am_aux_dir/disabled automake"
    ACLOCAL="$SHELL $am_aux_dir/disabled autoconf"
    AC_MSG_RESULT([$llnl_automake_version (disabled)])
  fi
])
