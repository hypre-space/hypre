dnl
dnl @synopsis LLNL_CONFIRM_BABEL_F90_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for F90 is enabled:
dnl     the cpp macro FORTRAN90_DISABLED is undefined
dnl     the automake conditional SUPPORT_FORTRAN90 is true
dnl
dnl  If Babel support for F90 is disabled:
dnl     the cpp macro FORTRAN90_DISABLED is defined as true
dnl     the automake conditional SUPPORT_FORTRAN90 is false
dnl
dnl  @author 
dnl
dnl  Note:  Clone of F77 version.

AC_DEFUN([LLNL_CONFIRM_BABEL_F90_SUPPORT], [
  #begin LLNL_CONFIRM_BABEL_F90_SUPPORT
  if test \( -z "$FC" \) -a \( -n "$F90" \); then
	AC_MSG_WARN([FC environment variable is preferred over F90.  compensating])
	FC="$F90"
  fi
  if test \( -z "$FC" \) -a \( -n "$F90" \); then
	AC_MSG_WARN([FCFLAGS environment variable is preferred over F90FLAGS.  compensating])
	FCFLAGS="$F90FLAGS"
  fi

  AC_ARG_ENABLE([fortran90],
        AC_HELP_STRING([--enable-fortran90@<:@=FC@:>@],
                       [fortran 90 language bindings @<:@default=yes@:>@]),
               [enable_fortran90="$enableval"],
               [enable_fortran90=yes])
  test -z "$enable_fortran90" && enable_fortran90=yes
  if test $enable_fortran90 != no; then
    if test $enable_fortran90 != yes; then 
      FC=$enable_fortran90
      enable_fortran90=yes
    fi
  fi

  if test "X$enable_fortran90" != "Xno"; then
    AC_PROG_FC(,1990)dnl was AC_PROG_F90
   AC_LANG_PUSH(Fortran) dnl gkk Do I need this?
    AC_FC_SRCEXT([f90],[])
   AC_LANG_POP(Fortran) dnl gkk Do I need this?
    LLNL_LIB_CHASM
  else
    FC=
  fi
  #end LLNL_CONFIRM_BABEL_F90_SUPPORT
])

AC_DEFUN([LLNL_CONFIRM_BABEL_F90_SUPPORT2],[
  AC_REQUIRE([LLNL_CONFIRM_BABEL_F90_SUPPORT])
  AC_REQUIRE([LLNL_F90_LIBRARY_LDFLAGS]) dnl get the order right
    #begin LLNL_CONFIRM_BABEL_F90_SUPPORT2
    if test \( -n "$FC" \) -a \( "X$enable_chasm" = "Xyes" \); then 
	F90="$FC"
        # 5.a. Libraries (existence)
	dnl LLNL_F90_LIBRARY_LDFLAGS dnl slight mod to AC_FC_LIBRARY_LDFLAGS
	LLNL_FC_MAIN dnl changed to requie LLNL_FC_LIBRARY_LDFLAGS
	LLNL_LIB_FCMAIN dnl needed to define the lib to include
        AC_FC_DUMMY_MAIN 
        LLNL_SORT_FCLIBS
	AC_FC_WRAPPERS dnl        LLNL_F90_NAME_MANGLING
	LLNL_F90_NAME_MANGLING dnl required for LLNL_F90_C_CONFIG
        LLNL_F90_C_CONFIG
	LLNL_F90_POINTER_SIZE
	LLNL_F90_VOLATILE
    else
	AC_WARN([Disabling F90 Support])
	if test \( -n "$FC" \); then
          enable_fortran90="no_chasm"
        else
          enable_fortran90="broken"	
        fi
    fi
  #end LLNL_CONFIRM_BABEL_F90_SUPPORT2
])

AC_DEFUN([LLNL_CONFIRM_BABEL_F90_SUPPORT3],[
  #begin LLNL_CONFIRM_BABEL_F90_SUPPORT3
  if test "X$enable_fortran90" = "Xno"; then
    msgs="$msgs
	  Fortran90 disabled by request.";
  elif test "X$enable_fortran90" = "Xyes"; then
    msgs="$msgs
	  Fortran90 enabled.";
  elif test "X$enable_fortran90" = "Xno_chasm"; then
    msgs="$msgs
	  Fortran90 disabled against user request: no CHASM installation found.";
  else
    msgs="$msgs
	  Fortran90 disabled against user request: no viable compiler found.";
  fi 
  if test "X$enable_fortran90" != "Xyes"; then
    AC_DEFINE(FORTRAN90_DISABLED, 1, [If defined, F90 support was disabled at configure time])
  fi
  AM_CONDITIONAL(SUPPORT_FORTRAN90, test "X$enable_fortran90" = "Xyes")
  LLNL_WHICH_PROG(WHICH_FC)
  #end LLNL_CONFIRM_BABEL_F90_SUPPORT3
])
