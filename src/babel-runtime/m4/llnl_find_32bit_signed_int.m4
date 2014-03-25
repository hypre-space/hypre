
dnl @synopsis LLNL_FIND_32BIT_SIGNED_INT
dnl
dnl @author Gary Kumfert, LLNL
AC_DEFUN([LLNL_FIND_32BIT_SIGNED_INT],
[AC_CACHE_CHECK(for 32 bit signed int,
 llnl_cv_find_32bit_signed_int, 
 [AC_REQUIRE([LLNL_CHECK_LONG_LONG])
  if test $ac_cv_sizeof_int -eq 4; then
    llnl_cv_find_32bit_signed_int=int;
  elif test $ac_cv_sizeof_short -eq 4; then
    llnl_cv_find_32bit_signed_int=short;
  elif test $ac_cv_sizeof_long -eq 4; then 
    llnl_cv_find_32bit_signed_int=long;
  elif test $ac_cv_sizeof_long_long -eq 4; then
    llnl_cv_find_32bit_signed_int="long long";
  else
    llnl_cv_find_32bit_signed_int=unresolved
  fi
])
 if test "$llnl_cv_find_32bit_signed_int" = "unresolved"; then
   AC_MSG_WARN([Could not identify a suitable 4 byte signed integer type])
 fi
])
