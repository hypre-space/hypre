
dnl 
dnl @synopsis LLNL_FUNC_DRAND_FORTYEIGHT
dnl 
dnl If the C compiler has drand48() define `LLNL_HAVE_FUNC_DRAND_FORTYEIGHT'
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
AC_DEFUN([LLNL_FUNC_DRAND_FORTYEIGHT],
[
AC_MSG_CHECKING([if drand48 is available])
AC_CACHE_VAL(llnl_cv_have_drand_fortyeight,
[
AC_LANG_SAVE
AC_LANG_C
AC_TRY_COMPILE([#include <stdlib.h>],
[ double d = drand48();],
llnl_cv_have_drand_fortyeight=yes,
llnl_cv_have_drand_fortyeight=no)
AC_LANG_RESTORE
])
AC_MSG_RESULT($llnl_cv_have_drand_fortyeight)
if test "$llnl_cv_have_drand_fortyeight" = yes; then
AC_DEFINE(HAVE_FUNCTION_DRAND48,,[define if drand48() is available])
fi])	
