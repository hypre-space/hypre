
dnl
dnl @synopsis LLNL_CHECK_AUTOCONF(VERSION)
dnl
dnl Check whether autoconf is the specified version.
dnl
dnl @author Gary Kumfert
AC_DEFUN([LLNL_CHECK_AUTOCONF], 
[
  AC_MSG_CHECKING(for autoconf/autoheader version)
  changequote(,)
  llnl_autoconf_version=`autoconf --version 2>/dev/null | sed '1s/.* \([^ ]*\)$/\1/g;1q'`
  changequote([,])
  if test "X$1" = "X$llnl_autoconf_version"; then
    AC_MSG_RESULT([$llnl_autoconf_version (enabled)])
  else
    AUTOCONF="$SHELL $am_aux_dir/disabled autoconf"
    AUTOHEADER="$SHELL $am_aux_dir/disabled autoheader"
    AC_MSG_RESULT([$llnl_autoconf_version (disabled)])
  fi
])
