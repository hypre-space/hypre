
dnl @synopsis LLNL_FIND_64BIT_SIGNED_INT
dnl
dnl @author Gary Kumfert, LLNL
AC_DEFUN([LLNL_FIND_64BIT_SIGNED_INT],
[AC_CACHE_CHECK(for 64 bit signed integer,
 llnl_cv_find_64bit_signed_int, 
 [AC_REQUIRE([LLNL_CHECK_LONG_LONG])
  if test $ac_cv_sizeof_int -eq 8; then
    llnl_cv_find_64bit_signed_int=int;
  elif test $ac_cv_sizeof_short -eq 8; then
    llnl_cv_find_64bit_signed_int=short;
  elif test $ac_cv_sizeof_long -eq 8; then 
    llnl_cv_find_64bit_signed_int=long;
  elif test $ac_cv_sizeof_long_long -eq 8; then
    llnl_cv_find_64bit_signed_int="long long";
  else
    llnl_cv_find_64bit_signed_int="unresolved";
  fi
])
 if test "$llnl_cv_find_64bit_signed_int" = "unresolved"; then
   AC_MSG_WARN([Could not identify a suitable 8 byte signed integer type])
 fi 
])
