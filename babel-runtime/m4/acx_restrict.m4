

dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C_Support/acx_restrict.m4***
dnl @synopsis ACX_C_RESTRICT
dnl
dnl This macro determines whether the C compiler supports the "restrict"
dnl keyword introduced in ANSI C99, or an equivalent.  Does nothing if
dnl the compiler accepts the keyword.  Otherwise, if the compiler supports
dnl an equivalent (like gcc's __restrict__) defines "restrict" to be that.
dnl Otherwise, defines "restrict" to be empty.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>

AC_DEFUN([ACX_C_RESTRICT],
[AC_CACHE_CHECK([for C restrict keyword], acx_cv_c_restrict,
[acx_cv_c_restrict=unsupported
 AC_LANG_PUSH([C])
 # Try the official restrict keyword, then gcc's __restrict__, then
 # SGI's __restrict.  __restrict has slightly different semantics than
 # restrict (it's a bit stronger, in that __restrict pointers can't
 # overlap even with non __restrict pointers), but I think it should be
 # okay under the circumstances where restrict is normally used.
 for acx_kw in restrict __restrict__ __restrict; do
   AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[float * $acx_kw x;]])],[acx_cv_c_restrict=$acx_kw; break],[])
 done
 AC_LANG_POP([])
])
 if test "$acx_cv_c_restrict" != "restrict"; then
   acx_kw="$acx_cv_c_restrict"
   if test "$acx_kw" = unsupported; then acx_kw=""; fi
   AC_DEFINE_UNQUOTED(restrict, $acx_kw, [Define to equivalent of C99 restrict keyword, or to nothing if this is not supported.  Do not define if restrict is supported directly.])
 fi
])


