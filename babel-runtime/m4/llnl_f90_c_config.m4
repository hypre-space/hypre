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
sidl_cv_f90_str="str" dnl can also be "struct"
sidl_cv_f90_str_len="far" dnl can also be "near", only meaningful if $sidl_cv_str="str"
sidl_cv_f90_str_struct="str_len" dnl can also be "len_str", only meaningful if $sidl_cv_str="struct"
sidl_cv_f90_char="as_string"
dnl sidl_cv_f90_true=1
dnl sidl_cv_f90_false=0
sidl_cv_f90_str_minsize=512

AC_CACHE_CHECK([the integer value of F90's .true.], sidl_cv_f90_true,[dnl
ac_save_ext=$ac_ext
ac_ext=f90
AC_LANG_PUSH(Fortran)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK([],[
  logical log
  integer value
  equivalence (log, value)
  log = .true.
  write (*,*) value
],[dnl
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
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(dnl
,[
  logical log
  integer value
  equivalence (log, value)
  log = .false.
  write (*,*) value
],[dnl
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
dnl strings
if test "$sidl_cv_f90_str" = "struct"; then 
   if test "$sidl_cv_f90_str_struct" = "len_str"; then
      AC_DEFINE(SIDL_F90_STR_STRUCT_LEN_STR,,[Fortran 90 strings as length-char* structs])
   else 
      if test "$sidl_cv_f90_str_struct" != "str_len"; then
         AC_MSG_WARN([string structs as length-charptr or char_ptr/length undetermined, assuming the latter])   
      fi;
      AC_DEFINE(SIDL_F90_STR_STRUCT_STR_LEN,,[Fortran 90 strings as char*-length structs])
   fi;
else
   if test "$sidl_cv_f90_str" != "str"; then 
      AC_MSG_WARN([strings passed as structs or char*/length undetermined, assumming the latter])
   fi;
   if test "$sidl_cv_f90_str_len" = "near"; then
      AC_DEFINE(SIDL_F90_STR_LEN_NEAR,,[Fortran 90 strings lengths at end])
   else
      if test "$sidl_cv_f90_str_len" != "far"; then
         AC_MSG_WARN([string length immediately following char* or at end undetermined, assuming at end])
      fi
      AC_DEFINE(SIDL_F90_STR_LEN_FAR,,[Fortran 90 strings lengths immediately follow string])
   fi;
fi;
if test "$sidl_cv_f90_char" = "as_string"; then
   AC_DEFINE(SIDL_F90_CHAR_AS_STRING,,[Fortran 90 char args are strings])
fi; 
AC_DEFINE_UNQUOTED(SIDL_F90_TRUE,$sidl_cv_f90_true,[Fortran 90 logical true value])
AC_DEFINE_UNQUOTED(SIDL_F90_FALSE,$sidl_cv_f90_false,[Fortran 90 logical false value])
AC_DEFINE_UNQUOTED(SIDL_F90_STR_MINSIZE,$sidl_cv_f90_str_minsize,[Minimum size for out strings])
])
