dnl
dnl @synopsis LLNL_F77_C_CONFIG
dnl
dnl
dnl @author
dnl
AC_DEFUN([LLNL_F77_C_CONFIG],[
AC_REQUIRE([AC_CANONICAL_TARGET])dnl
AC_REQUIRE([AC_PROG_F77])dnl
AC_REQUIRE([LLNL_F77_NAME_MANGLING])
dnl set some resonable defaults
sidl_cv_f77_str="str" dnl can also be "struct"
sidl_cv_f77_str_len="far" dnl can also be "near", only meaningful if $sidl_cv_str="str"
sidl_cv_f77_str_struct="str_len" dnl can also be "len_str", only meaningful if $sidl_cv_str="struct"
sidl_cv_f77_char="as_string"
dnl sidl_cv_f77_true=1
dnl sidl_cv_f77_false=0
sidl_cv_f77_str_minsize=512

AC_CACHE_CHECK([the integer value of F77's .true.], sidl_cv_f77_true,[dnl
AC_LANG_PUSH(Fortran 77)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(dnl
,[
        logical log
        integer value
        equivalence (log, value)
        log = .true.
        write (*,*) value
],[
sidl_cv_f77_true=`./conftest$ac_exeext`
llnl_status=$?
if test -z "$sidl_cv_f77_true"; then
  AC_MSG_ERROR([Unable to determine integer value of F77 .true. (running ./conftest$ac_exeext produced no output)])
fi
],[echo "the program generates"  `./conftest$ac_exeext`
   AC_MSG_ERROR([Unable to determine integer value of F77 .true.])])
AC_LANG_POP(Fortran 77)dnl])

AC_CACHE_CHECK([the integer value of F77's .false.], sidl_cv_f77_false,[dnl
AC_LANG_PUSH(Fortran 77)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(,[
        logical log
        integer value
        equivalence (log, value)
        log = .false.
	write (*,*) value
],[
sidl_cv_f77_false=`./conftest$ac_exeext`
if test -z "$sidl_cv_f77_false"; then
  AC_MSG_ERROR([Unable to determine integer value of F77 .false. (running ./conftest$ac_exeext produced no output)])
fi
],[echo "the program generates"  `./conftest$ac_exeext`
   AC_MSG_ERROR([Unable to determine integer value of F77 .false.])])
AC_LANG_POP(Fortran 77)dnl
])

dnl set number of underscores
if test -z "$sidl_cv_f77_number_underscores"; then
   AC_MSG_ERROR([Number of underscores not determined])
elif test $sidl_cv_f77_number_underscores -eq 2; then
   AC_DEFINE(SIDL_F77_TWO_UNDERSCORE,,[two underscores after F77 symbols])
elif test $sidl_cv_f77_number_underscores -eq 1; then
   AC_DEFINE(SIDL_F77_ONE_UNDERSCORE,,[one underscore after F77 symbols])
else
  if test $sidl_cv_f77_number_underscores -ne 0; then
     AC_WARN([number of underscores after F77 symbols undetermined, assuming zero])
  fi;
   AC_DEFINE(SIDL_F77_ZERO_UNDERSCORE,,[no underscores after F77 symbols])
fi;
dnl set case
if test "$sidl_cv_f77_case" = "mixed"; then
   AC_DEFINE(SIDL_F77_MIXED_CASE,,[F77 symbols are mixed case])
elif test "$sidl_cv_f77_case" = "upper"; then
   AC_DEFINE(SIDL_F77_UPPER_CASE,,[F77 symbols are upper case])
else
   if test "$sidl_cv_f77_case" != "lower"; then
      AC_WARN([case of f77 symbols undetermined, assuming lower case])
   fi  
   AC_DEFINE(SIDL_F77_LOWER_CASE,,[F77 symbols are lower case])
fi;
dnl strings
if test "$sidl_cv_f77_str" = "struct"; then 
   if test "$sidl_cv_f77_str_struct" = "len_str"; then
      AC_DEFINE(SIDL_F77_STR_STRUCT_LEN_STR,,[F77 strings as length-char* structs])
   else 
      if test "$sidl_cv_f77_str_struct" != "str_len"; then
         AC_WARN([string structs as length-charptr or char_ptr/length undetermined, assuming the latter])   
      fi;
      AC_DEFINE(SIDL_F77_STR_STRUCT_STR_LEN,,[F77 strings as char*-length structs])
   fi;
else
   if test "$sidl_cv_f77_str" != "str"; then 
      AC_WARN([strings passed as structs or char*/length undetermined, assumming the latter])
   fi;
   if test "$sidl_cv_f77_str_len" = "near"; then
      AC_DEFINE(SIDL_F77_STR_LEN_NEAR,,[F77 strings lengths at end])
   else
      if test "$sidl_cv_f77_str_len" != "far"; then
         AC_WARN([string length immediately following char* or at end undetermined, assuming at end])
      fi
      AC_DEFINE(SIDL_F77_STR_LEN_FAR,,[F77 strings lengths immediately follow string])
   fi;
fi;
if test "$sidl_cv_f77_char" = "as_string"; then
   AC_DEFINE(SIDL_F77_CHAR_AS_STRING,,[F77 char args are strings])
fi; 
AC_DEFINE_UNQUOTED(SIDL_F77_TRUE,$sidl_cv_f77_true,[F77 logical true value])
AC_DEFINE_UNQUOTED(SIDL_F77_FALSE,$sidl_cv_f77_false,[F77 logical false value])
AC_DEFINE_UNQUOTED(SIDL_F77_STR_MINSIZE,$sidl_cv_f77_str_minsize,[Minimum size for out strings])
])
