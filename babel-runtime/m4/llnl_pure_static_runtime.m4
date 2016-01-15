dnl
dnl @synopsis LLNL_PURE_STATIC_RUNTIME
dnl
dnl There is a desire to be able to turn off dynamic loading in
dnl libsidl.a (the static runtime library) regardless of whether
dnl dlopen is available or not. This file adds a command line argument
dnl a pure static runtime with no reference to dlopen.
dnl
dnl @author Tom Epperly <epperly2@llnl.gov>

AC_DEFUN([LLNL_PURE_STATIC_RUNTIME],[dnl
  AC_ARG_ENABLE([pure-static-runtime],
	AS_HELP_STRING(--enable-pure-static-runtime@<:@=yes@:>@,disable dlopen in libsidl.a @<:@default=no@:>@),
	[enable_pure_static_runtime="$enableval"],
	[enable_pure_static_runtime=no])
  if test "X$enable_pure_static_runtime" = "Xyes"; then
    AC_DEFINE(SIDL_PURE_STATIC_RUNTIME,1,
	[Define to prevent the static runtime from using dlopen.])
  fi
])
