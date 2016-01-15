
dnl @synopsis LLNL_PROG_PYTHON
dnl
dnl LLNL_PROG_PYTHON tests for an existing Python interpreter. It uses
dnl the environment variable PYTHON and then tries to find python
dnl in standard places.
dnl
dnl @author ?
AC_DEFUN([LLNL_PROG_PYTHON],[
  AC_REQUIRE([AC_EXEEXT])dnl
  AC_CHECK_PROGS(PYTHON, python$EXEEXT)
  if test "x$PYTHON" = x; then
    AC_MSG_WARN([Not building Python support - unable to find Python executable])
  else
    AC_MSG_CHECKING([if $PYTHON is executable])
    if AC_TRY_COMMAND($PYTHON -c "import sys; print sys.version") >/dev/null 2>&1; then
      AC_MSG_RESULT([yes])
    else
      AC_MSG_RESULT([no])
      AC_MSG_WARN([Not building Python support - $PYTHON does not run])
      unset PYTHON
    fi
  fi
])
