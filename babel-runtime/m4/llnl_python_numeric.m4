dnl @synopsis LLNL_PYTHON_NUMERIC
dnl
dnl @author ?
AC_DEFUN([LLNL_PYTHON_NUMERIC],[
  AC_REQUIRE([LLNL_PROG_PYTHON])dnl
  AC_REQUIRE([LLNL_PYTHON_LIBRARY])dnl
  AC_CACHE_CHECK(for Numerical Python, llnl_cv_python_numerical, [
    llnl_cv_python_numerical=no
    if test "X$PYTHON" != "X"; then
      if AC_TRY_COMMAND($PYTHON -c "import Numeric") > /dev/null 2>&1; then
        if test -f $llnl_cv_python_include/Numeric/arrayobject.h; then
          llnl_cv_python_numerical=yes
        fi
      fi
    fi
  ])
])
