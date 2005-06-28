
dnl @synopsis LLNL_CHECK_INT64_T
dnl  
dnl Checks for a int64_t in inttypes.h, systypes.h,
dnl stdlib.h and stddef.h.  If none is found int64_t
dnl is defined as some 64bit signed integer searched
dnl by alternative means.
dnl
dnl @author Gary Kumfert

AC_DEFUN([LLNL_CHECK_INT64_T], 
[AC_REQUIRE([LLNL_FIND_64BIT_SIGNED_INT])dnl
 AC_REQUIRE([AC_HEADER_STDC])dnl
 AC_CACHE_CHECK(for int64_t, llnl_cv_int64_t,
 [
  AC_LANG_PUSH([C])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#include <sys/types.h>
#if STDC_HEADERS
#include <stdlib.h>
#include <stddef.h>
#endif
]], [[int64_t t]])],[llnl_cv_int64_t=yes],[llnl_cv_int64_t=no])
  AC_LANG_POP([])
])
if test "$llnl_cv_int64_t" = "no"; then 
  if test "$llnl_cv_find_64bit_signed_int" = "unresolved"; then
    AC_MSG_ERROR([Cannot find int64_t or an alternative 8 byte integer])
  else 
    AC_MSG_WARN([Using $llnl_cv_find_64bit_signed_int instead of int64_t])
    AC_DEFINE_UNQUOTED(int64_t, $llnl_cv_find_64bit_signed_int,
	  [used when a compiler does not recognize int64_t])
  fi
fi
])
