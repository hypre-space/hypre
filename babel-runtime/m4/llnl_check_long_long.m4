
dnl @synopsis LLNL_CHECK_LONG_LONG
dnl
dnl checks for a `long long' type
dnl 
dnl @version 
dnl @author Gary Kumfert, LLNL
AC_DEFUN([LLNL_CHECK_LONG_LONG],
[AC_CACHE_CHECK(for type long long,
 ac_cv_c_long_long,
 AC_RUN_IFELSE([AC_LANG_SOURCE([[int main() {
 exit(sizeof(long long) < sizeof(long)); }]])],[ac_cv_c_long_long=yes],[ac_cv_c_long_long=no],[])
 if test "$ac_cv_c_long_long" = yes; then
   AC_DEFINE(HAVE_LONG_LONG,,[define if long long is a built in type])
 fi
)])


