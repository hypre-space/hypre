
# LLNL_F90_NAME_MANGLING
# ---------------------
# Test for the name mangling scheme used by the Fortran 90 compiler.
#
# Sets ac_cv_f90_mangling. The value contains three fields, separated
# by commas:
#
# lower case / upper case:
#    case translation of the Fortran 90 symbols
# underscore / no underscore:
#    whether the compiler appends "_" to symbol names
# extra underscore / no extra underscore:
#    whether the compiler appends an extra "_" to symbol names already
#    containing at least one underscore
#
# Note:  Clone of F77 version.
#
AC_DEFUN([LLNL_F90_NAME_MANGLING],
[AC_REQUIRE([LLNL_F90_LIBRARY_LDFLAGS])dnl
AC_REQUIRE([AC_FC_DUMMY_MAIN])dnl
AC_CACHE_CHECK([for Fortran 90 name-mangling scheme],
               ac_cv_f90_mangling,
[AC_LANG_PUSH(Fortran)dnl
AC_COMPILE_IFELSE(
[subroutine Foobar()
return
end subroutine Foobar
subroutine Foo_bar()
return
end subroutine Foo_bar],
[mv conftest.$ac_objext cf90_test.$ac_objext

  AC_LANG_PUSH(C)dnl

  ac_save_LIBS=$LIBS
  LIBS="cf90_test.$ac_objext $LIBS $FCLIBS"

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
	   sidl_cv_f90_case="lower"
           ac_foo_bar=foo_bar
           ;;
        FOOBAR)
	   sidl_cv_f90_case="upper"
           ac_foo_bar=FOO_BAR
           ;;
        Foobar)
	   sidl_cv_f90_case="mixed"
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
	ac_cv_f90_mangling="$sidl_cv_f90_case case"
        if test -z "$ac_underscore"; then
           ac_cv_f90_mangling="$ac_cv_f90_mangling, no underscore"
	   sidl_cv_f90_number_underscores=0
	else
           ac_cv_f90_mangling="$ac_cv_f90_mangling, underscore"
	   sidl_cv_f90_number_underscores=1
        fi
        if test -z "$ac_extra"; then
           ac_cv_f90_mangling="$ac_cv_f90_mangling, no extra underscore"
	else
           ac_cv_f90_mangling="$ac_cv_f90_mangling, extra underscore"
	   sidl_cv_f90_number_underscores=`expr $sidl_cv_f90_number_underscores + 1`
        fi
      else
	ac_cv_f90_mangling="unknown"
      fi
  else
     ac_cv_f90_mangling="unknown"
  fi

  if test "$ac_cv_f90_mangling" = "unknown"; then
    AC_MSG_ERROR([Failed to determine how F90 mangles linker symbols.])
  fi
  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f cf90_test* conftest*])
AC_LANG_POP(Fortran)dnl
])
])# LLNL_F90_NAME_MANGLING
