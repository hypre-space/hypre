
dnl @synopsis LLNL_CHECK_INT32_T
dnl  
dnl Checks for a int32_t in inttypes.h, systypes.h,
dnl stdlib.h and stddef.h.  If none is found int32_t
dnl is defined as some 32bit signed integer searched
dnl by alternative means.
dnl
dnl @author Gary Kumfert

AC_DEFUN([LLNL_CHECK_INT32_T], 
[AC_REQUIRE([LLNL_FIND_32BIT_SIGNED_INT])dnl
 AC_REQUIRE([AC_HEADER_STDC])dnl
 AC_CACHE_CHECK(for int32_t, llnl_cv_int32_t,
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
]], [[int32_t t]])],[llnl_cv_int32_t=yes],[llnl_cv_int32_t=no])
  AC_LANG_POP([])
])
if test "$llnl_cv_int32_t" = no; then 
  if test "$llnl_cv_find_32bit_signed_int" = "unresolved"; then
    AC_MSG_ERROR([Cannot find int32_t or an alternative 4 byte integer])
  else 
    AC_MSG_WARN([Using $llnl_cv_find_32bit_signed_int instead of int32_t])
    AC_DEFINE_UNQUOTED(int32_t, $llnl_cv_find_32bit_signed_int,
	  [used when a compiler does not recognize int32_t])
  fi
fi
])
