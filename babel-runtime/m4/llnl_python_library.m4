dnl @synopsis LLNL_PYTHON_LIBRARY 
dnl
dnl @author ?
AC_DEFUN([LLNL_PYTHON_LIBRARY],[
  AC_REQUIRE([LLNL_PROG_PYTHON])dnl

  if test "X$PYTHON" != "X"; then
    AC_CACHE_CHECK(for Python version, llnl_cv_python_version, [
      llnl_cv_python_version=`$PYTHON -c 'import sys; print sys.version' | sed '1s/^\(...\).*/\1/g;1q' 2>/dev/null`
    ])
    AC_CACHE_CHECK(for Python library path, llnl_cv_python_library, [
      llnl_python_prefix=`$PYTHON -c 'import sys; print sys.prefix' 2>/dev/null`
      llnl_cv_python_library="$llnl_python_prefix/"`$PYTHON -c "import sys; print sys.__dict__.get('lib','lib')"`"/python$llnl_cv_python_version"
    ])
    AC_CACHE_CHECK(for Python include path, llnl_cv_python_include, [
      llnl_python_prefix=`$PYTHON -c 'import sys; print sys.prefix' 2>/dev/null`
      llnl_cv_python_include="$llnl_python_prefix/include/python$llnl_cv_python_version"
    ])
  fi

  AC_DEFINE_UNQUOTED(PYTHON_VERSION,"$llnl_cv_python_version",[A string indicating the Python version number])
  PYTHONLIB="$llnl_cv_python_library"
  PYTHONINC="-I$llnl_cv_python_include"
  PYTHON_VERSION="$llnl_cv_python_version"
  AC_SUBST(PYTHONLIB)
  AC_SUBST(PYTHONINC)
  AC_SUBST(PYTHON_VERSION)
])
