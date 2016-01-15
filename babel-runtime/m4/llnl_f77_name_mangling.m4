
# LLNL_F77_NAME_MANGLING
# ---------------------
# Test for the name mangling scheme used by the Fortran 77 compiler.
#
# Sets ac_cv_f77_mangling. The value contains three fields, separated
# by commas:
#
# lower case / upper case:
#    case translation of the Fortran 77 symbols
# underscore / no underscore:
#    whether the compiler appends "_" to symbol names
# extra underscore / no extra underscore:
#    whether the compiler appends an extra "_" to symbol names already
#    containing at least one underscore
#
AC_DEFUN([LLNL_F77_NAME_MANGLING],
[AC_REQUIRE([LLNL_F77_LIBRARY_LDFLAGS])dnl
AC_REQUIRE([LLNL_F77_DUMMY_MAIN])dnl
AC_CACHE_CHECK([for Fortran 77 name-mangling scheme],
               ac_cv_f77_mangling,
[AC_LANG_PUSH(Fortran 77)dnl
AC_COMPILE_IFELSE(
[      subroutine Foobar()
      return
      end
      subroutine Foo_bar()
      return
      end],
[mv conftest.$ac_objext cf77_test.$ac_objext

  AC_LANG_PUSH(C)dnl

  ac_save_LIBS=$LIBS
  LIBS="cf77_test.$ac_objext $LIBS $FLIBS"

  ac_success=no
  for ac_foobar in foobar Foobar FOOBAR; do
    for ac_underscore in "" "_"; do
      ac_func="$ac_foobar$ac_underscore"
      AC_TRY_LINK_FUNC($ac_func,
         [ac_success=yes; break 2])
    done
  done

  if test "$ac_success" = "yes"; then
     case $ac_foobar in
        foobar)
	   sidl_cv_f77_case="lower"
           ac_foo_bar=foo_bar
           ;;
        FOOBAR)
	   sidl_cv_f77_case="upper"
           ac_foo_bar=FOO_BAR
           ;;
        Foobar)
	   sidl_cv_f77_case="mixed"
           ac_foo_bar=Foo_bar
           ;;
     esac

     ac_success_extra=no
     for ac_extra in "" "_"; do
        ac_func="$ac_foo_bar$ac_underscore$ac_extra"
        AC_TRY_LINK_FUNC($ac_func,
        [ac_success_extra=yes; break])
     done

     if test "$ac_success_extra" = "yes"; then
	ac_cv_f77_mangling="$sidl_cv_f77_case case"
        if test -z "$ac_underscore"; then
           ac_cv_f77_mangling="$ac_cv_f77_mangling, no underscore"
	   sidl_cv_f77_number_underscores=0
	else
           ac_cv_f77_mangling="$ac_cv_f77_mangling, underscore"
	   sidl_cv_f77_number_underscores=1
        fi
        if test -z "$ac_extra"; then
           ac_cv_f77_mangling="$ac_cv_f77_mangling, no extra underscore"
	else
           ac_cv_f77_mangling="$ac_cv_f77_mangling, extra underscore"
	   sidl_cv_f77_number_underscores=`expr $sidl_cv_f77_number_underscores + 1`
        fi
      else
	ac_cv_f77_mangling="unknown"
      fi
  else
     ac_cv_f77_mangling="unknown"
  fi

  if test "$ac_cv_f77_mangling" = "unknown"; then
    AC_MSG_ERROR([Failed to determine how F77 mangles linker symbols.])
  fi
  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f cf77_test* conftest*])
AC_LANG_POP(Fortran 77)dnl
])
])# LLNL_F77_NAME_MANGLING
