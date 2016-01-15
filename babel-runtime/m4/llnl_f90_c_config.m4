dnl
dnl @synopsis LLNL_F90_C_CONFIG
dnl
dnl
dnl @author
dnl
dnl Note:  Clone of F77 version.

AC_DEFUN([LLNL_F90_C_CONFIG],[
AC_REQUIRE([AC_CANONICAL_TARGET])dnl
AC_REQUIRE([AC_PROG_FC])dnl
AC_REQUIRE([AC_FC_WRAPPERS])dnl
dnl set some resonable defaults
sidl_cv_f90_char="as_string"
dnl sidl_cv_f90_true=1
dnl sidl_cv_f90_false=0
sidl_cv_f90_str_minsize=512

AC_CACHE_CHECK([the integer value of F90's .true.], sidl_cv_f90_true,[dnl
ac_save_ext=$ac_ext
ac_ext=f90
AC_LANG_PUSH(Fortran)dnl
AC_LINK_IFELSE([AC_LANG_PROGRAM([],[
  logical log
  integer value
  equivalence (log, value)
  log = .true.
  write (*,*) value
])],[dnl
sidl_cv_f90_true=`./conftest$ac_exeext`
llnl_status=$?
if test -z "$sidl_cv_f90_true"; then
  AC_MSG_ERROR([Unable to determine integer value of F90 .true. (running ./conftest$ac_exeext produced no output)])
fi
],[
AC_MSG_ERROR([Unable to determine integer value of F90 .true.])])
AC_LANG_POP(Fortran)dnl])
ac_ext=$ac_save_ext

AC_CACHE_CHECK([the integer value of F90's .false.], sidl_cv_f90_false,[dnl
AC_LANG_PUSH(Fortran)dnl
AC_LINK_IFELSE([AC_LANG_PROGRAM([],[
  logical log
  integer value
  equivalence (log, value)
  log = .false.
  write (*,*) value
])],[dnl
sidl_cv_f90_false=`./conftest$ac_exeext`
if test -z "$sidl_cv_f90_false"; then
  AC_MSG_ERROR([Unable to determine integer value of F90 .false. (running ./conftest$ac_exeext produced no output)])
fi
],[echo "the program generates"  `./conftest$ac_exeext`
   AC_MSG_ERROR([Unable to determine integer value of F90 .false.])])
AC_LANG_POP(Fortran)dnl
])

dnl set number of underscores
if test -z "$sidl_cv_f90_number_underscores"; then
   AC_MSG_ERROR([Number of F90 underscores not determined])
elif test $sidl_cv_f90_number_underscores -eq 2; then
   AC_DEFINE(SIDL_F90_TWO_UNDERSCORE,,[two underscores after Fortran 90 symbols])
elif test $sidl_cv_f90_number_underscores -eq 1; then
   AC_DEFINE(SIDL_F90_ONE_UNDERSCORE,,[one underscore after Fortran 90 symbols])
else
  if test $sidl_cv_f90_number_underscores -ne 0; then
     AC_MSG_WARN([number of underscores after Fortran 90 symbols undetermined, assuming zero])
  fi;
   AC_DEFINE(SIDL_F90_ZERO_UNDERSCORE,,[no underscores after Fortran 90 symbols])
fi;
dnl set case
if test "$sidl_cv_f90_case" = "mixed"; then
   AC_DEFINE(SIDL_F90_MIXED_CASE,,[Fortran 90 symbols are mixed case])
elif test "$sidl_cv_f90_case" = "upper"; then
   AC_DEFINE(SIDL_F90_UPPER_CASE,,[Fortran 90 symbols are upper case])
else
   if test "$sidl_cv_f90_case" != "lower"; then
      AC_MSG_WARN([case of Fortran 90 symbols undetermined, assuming lower case])
   fi  
   AC_DEFINE(SIDL_F90_LOWER_CASE,,[Fortran 90 symbols are lower case])
fi;
AC_DEFINE_UNQUOTED(SIDL_F90_TRUE,$sidl_cv_f90_true,[Fortran 90 logical true value])
AC_DEFINE_UNQUOTED(SIDL_F90_FALSE,$sidl_cv_f90_false,[Fortran 90 logical false value])
LLNL_FORTRAN_STRING_TEST(Fortran,F90,$FCLIBS)
if test "$sidl_cv_f90_char" = "as_string"; then
   AC_DEFINE(SIDL_F90_CHAR_AS_STRING,,[Fortran 90 char args are strings])
fi; 
AC_DEFINE_UNQUOTED(SIDL_F90_STR_MINSIZE,$sidl_cv_f90_str_minsize,[Minimum size for out strings])
])
