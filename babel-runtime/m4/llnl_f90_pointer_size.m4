# LLNL_F90_POINTER_SIZE
# ---------------------
# Try to determine the size of a F90 pointer to a derived type
# Note the size of a pointer may depend on the type of thing
# it is pointing to. :-(
#
AC_DEFUN([LLNL_F90_POINTER_SIZE],
[AC_REQUIRE([AC_PROG_FC])dnl
AC_REQUIRE([LLNL_F90_LIBRARY_LDFLAGS])dnl
AC_CACHE_CHECK([for Fortran 90 pointer to derived type size],
	ac_cv_f90_pointer_size,
[AC_LANG_PUSH(C)dnl
  case $ac_cv_f90_mangling in
    "lower case, no underscore"*)
       ac_cv_f90_pointer_func="pointerdiff";;
    "lower case, underscore"*)
       ac_cv_f90_pointer_func="pointerdiff_";;
    "upper case, no underscore"*)
       ac_cv_f90_pointer_func="POINTERDIFF";;
    "upper case, underscore"*)
       ac_cv_f90_pointer_func="POINTERDIFF_";;
    "mixed case, no underscore"*)
       ac_cv_f90_pointer_func="pointerdiff";;
    "mixed case, underscore"*)
       ac_cv_f90_pointer_func="pointerdiff_";;
     *)
  	AC_MSG_ERROR([unknown Fortran 90 name-mangling scheme])
	;;
  esac
  AC_COMPILE_IFELSE(
   [#include <stdio.h>
#ifdef __cplusplus
extern "C" 
#endif
void $ac_cv_f90_pointer_func(char *p1, char *p2)
{
  printf("%ld\n", (long)(p2 - p1));
  fflush(stdout); /* needed for gfortran */
}
],
   [mv conftest.$ac_objext cf90_test.$ac_objext

    AC_LANG_PUSH(Fortran)dnl
    ac_save_LIBS=$LIBS
    LIBS="cf90_test.$ac_objext $LIBS"
    AC_LINK_IFELSE([AC_LANG_PROGRAM([[]], [[
  implicit none
  type foo 
    sequence
    integer :: data
  end type foo

  type foowrap
    sequence
    type(foo), pointer :: foop
  end type foowrap

  external pointerdiff
  type(foowrap), dimension(2) :: fooa
  call pointerdiff(fooa(1), fooa(2))
]])],[dnl
      ac_cv_f90_pointer_size=`./conftest$ac_exeext`
      if test -z "$ac_cv_f90_pointer_size"; then
	AC_MSG_ERROR([Unable to determine pointer size (running ./conftest$ac_exeext produced no output)])
      fi
],[
      AC_MSG_ERROR([Unable to determine pointer size])
    ])
    LIBS=$ac_save_LIBS
    AC_LANG_POP(Fortran)dnl
    rm -f cf90_test* conftest*
   ])
  AC_LANG_POP(C)dnl
  ])

AC_DEFINE_UNQUOTED(SIDL_F90_POINTER_SIZE, $ac_cv_f90_pointer_size,
[Size in bytes for a F90 pointer to a derived type])
])
# LLNL_F90_POINTER_SIZE
