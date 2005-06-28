dnl
dnl @synopsis LLNL_CONFIRM_BABEL_CXX_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for CXX is enabled:
dnl     the cpp macro CXX_DISABLED is undefined
dnl     the automake conditional SUPPORT_CXX is true
dnl
dnl  If Babel support for CXX is disabled:
dnl     the cpp macro CXX_DISABLED is defined as true
dnl     the automake conditional SUPPORT_CXX is false
dnl
dnl  @author Gary Kumfert

dnl this is broken into two tests 'cause ac_cxx_namespaces
dnl consistently gets placed *before* ac_prog_cxx otherwise.
dnl We have to prevent this at all costs!

AC_DEFUN([LLNL_CONFIRM_BABEL_CXX_SUPPORT],[
  if test -z "$CCC"; then
    CCC="g++ KCC CC xlC"
  fi

  AC_ARG_ENABLE([cxx],
        AS_HELP_STRING(--enable-cxx@<:@=C++@:>@,C++ language bindings @<:@default=yes@:>@),
               [enable_cxx="$enableval"],
               [enable_cxx=yes])
  test -z "$enable_cxx" && enable_cxx=yes
  if test "$enable_cxx" != no; then
    if test $enable_cxx != yes; then 
      CCC=$enable_cxx
      enable_cxx=yes
    fi
  fi
  if test "X$enable_cxx" = "Xno"; then
    AC_MSG_ERROR([Sorry, this package cannot work without C++ enabled.])
  fi
  AC_PROG_CXX
  # confirm that that C++ compiler can compile a trivial file issue146
  AC_MSG_CHECKING([if C++ compiler works])
  AC_LANG_PUSH([C++])
  AC_TRY_COMPILE([],[],AC_MSG_RESULT([yes]),[
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([The C++ compiler $CXX fails to compile a trivial program (see config.log)])])
  AC_LANG_POP([])
])

AC_DEFUN([LLNL_CONFIRM_BABEL_CXX_SUPPORT2], [
  AC_REQUIRE([LLNL_CONFIRM_BABEL_CXX_SUPPORT])
  if test -n "$CXX"; then
    # 6.a. Libraries (existence) 
    LLNL_CXX_LIBRARY_LDFLAGS
    # 6.b. Header Files
    LLNL_CXX_OLD_HEADER_SUFFIX
    AC_CXX_HAVE_STD
    AC_CXX_HAVE_STL
    AC_CXX_HAVE_NUMERIC_LIMITS
    AC_CXX_COMPLEX_MATH_IN_NAMESPACE_STD
    AC_CXX_HAVE_COMPLEX
    AC_CXX_HAVE_COMPLEX_MATH1
    AC_CXX_HAVE_COMPLEX_MATH2
    AC_CXX_HAVE_IEEE_MATH
  fi
  AM_CONDITIONAL(SUPPORT_CXX, test "X$enable_cxx" != "Xno")
  if test "X$enable_cxx" = "Xno"; then
    AC_DEFINE(CXX_DISABLED, 1, [If defined, C++ support was disabled at configure time])
    msgs="$msgs 
  	  C++ disabled by request"
  else
    msgs="$msgs
  	  C++ enabled.";
  fi
  LLNL_WHICH_PROG(WHICH_CXX)
])
