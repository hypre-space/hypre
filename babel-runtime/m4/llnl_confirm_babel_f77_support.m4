dnl
dnl @synopsis LLNL_CONFIRM_BABEL_F77_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for F77 is enabled:
dnl     the cpp macro FORTRAN_DISABLED is undefined
dnl     the automake conditional SUPPORT_FORTRAN is true
dnl
dnl  If Babel support for F77 is disabled:
dnl     the cpp macro FORTRAN_DISABLED is defined as true
dnl     the automake conditional SUPPORT_FORTRAN is false
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_F77_SUPPORT], [
  AC_ARG_ENABLE([fortran77],
        AC_HELP_STRING([--enable-fortran77@<:@=F77@:>@],
                       [fortran 77 language bindings @<:@default=yes@:>@]),
               [enable_fortran77="$enableval"],
               [enable_fortran77=yes])
  test -z "$enable_fortran77" && enable_fortran77=yes
  if test $enable_fortran77 != no; then
    if test $enable_fortran77 != yes; then 
      F77=$enable_fortran77
      enable_fortran77=yes
    fi
  fi

  if test "X$enable_fortran77" != "Xno"; then
    AC_PROG_F77
    # 5.a. Libraries (existence)
    LLNL_LIB_FMAIN
    LLNL_F77_LIBRARY_LDFLAGS
    LLNL_F77_DUMMY_MAIN
    LLNL_SORT_FLIBS
    LLNL_F77_NAME_MANGLING
    LLNL_F77_C_CONFIG
  fi
  AM_CONDITIONAL(SUPPORT_FORTRAN77, test "X$enable_fortran77" != "Xno")
  if test "X$enable_fortran77" = "Xno"; then
    AC_DEFINE(FORTRAN77_DISABLED, 1, [If defined, Fortran support was disabled at configure time])
    msgs="$msgs
	  Fortran77 disabled by request";
  else
    msgs="$msgs
	  Fortran77 enabled.";
  fi
  LLNL_WHICH_PROG(WHICH_F77) 
])
