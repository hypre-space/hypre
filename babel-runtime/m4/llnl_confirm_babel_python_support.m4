dnl
dnl @synopsis LLNL_CONFIRM_BABEL_PYTHON_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for PYTHON is enabled:
dnl     the cpp macro PYTHON_DISABLED is undefined
dnl     the automake conditional SUPPORT_PYTHON is true
dnl
dnl  If Babel support for PYTHON is disabled:
dnl     the cpp macro PYTHON_DISABLED is defined as true
dnl     the automake conditional SUPPORT_PYTHON is false
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_PYTHON_SUPPORT],
 [AC_REQUIRE([LLNL_LIBXML_CONFIG])dnl
  AC_ARG_VAR([PYTHON],[Python interpreter])
  AC_ARG_ENABLE([python],
        AS_HELP_STRING(--enable-python@<:@=PYTHON@:>@,python language bindings @<:@default=yes@:>@),
               [enable_python="$enableval"],
               [enable_python=yes])
  test -z "$enable_python" && enable_python=yes
  if test $enable_python != no; then
    if test $enable_python != yes; then 
      PYTHON=$enable_python
      enable_python=yes
    fi
  fi

  if test "X$enable_python" != "Xno"; then
    LLNL_PYTHON_LIBRARY
    LLNL_PYTHON_NUMERIC
    LLNL_PYTHON_SHARED_LIBRARY
    LLNL_PYTHON_AIX
    if test "X$llnl_cv_python_numerical" != "Xyes" -o "X$enable_shared" = "Xno" -o "X$XML2_CONFIG" = "Xno"; then
       enable_python=no;
       AC_MSG_WARN([Configuration for Python failed.  Support for Python disabled!])
       if test "X$XML2_CONFIG" = "Xno"; then
          AC_MSG_WARN([Python requires libxml $LIBXML_REQUIRED_VERSION (or later)])
       fi
       msgs="$msgs
  	  Python support disabled against request, shared libs disabled or NumPy not found."
    elif test "X$llnl_python_shared_library_found" != "Xyes"; then
       AC_MSG_WARN([No Python shared library found.  Support for server-side Python disabled!])
       msgs="$msgs
  	  Server-side Python support disabled against request, can only do client side when no libpython.so found".
    else
       msgs="$msgs
  	  Python enabled.";
    fi
  else
    msgs="$msgs 
  	  Python support disabled by request"
  fi
  # support python in general?
  AM_CONDITIONAL(SUPPORT_PYTHON, test "X$enable_python" != "Xno")
  if test "X$enable_python" = "Xno"; then
    AC_DEFINE(PYTHON_DISABLED, 1, [If defined, Python support was disabled at configure time])
  fi 

  # support server-side python in particular
  AM_CONDITIONAL(SERVER_PYTHON, (test "X$enable_python" != "Xno") && (test "X$llnl_python_shared_library_found" = "Xyes"))
  if (test "X$enable_python" = "Xno") || (test "X$llnl_python_shared_library_found" != "Xyes"); then
    AC_DEFINE(PYTHON_SERVER_DISABLED, 1, [If defined, server-side Python support was disabled at configure time])
  fi;

  LLNL_WHICH_PROG(WHICH_PYTHON)
])
