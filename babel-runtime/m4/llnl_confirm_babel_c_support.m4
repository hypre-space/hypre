dnl
dnl @synopsis LLNL_CONFIRM_BABEL_C_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_C_SUPPORT], [
  AC_REQUIRE([AC_LTDL_SHLIBPATH])dnl
  ############################################################
  #
  # C Compiler
  #
  # AC_PROG_CC
  # Verify C compiler can compile trivial C program issue146
  AC_MSG_CHECKING([if C compiler works])
  AC_LANG_PUSH([C])
  AC_TRY_COMPILE([],[],AC_MSG_RESULT([yes]),[
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([The C compiler $CC fails to compile a trivial program (see config.log)])])
  AC_LANG_POP([])
  AC_DEFINE(SIDL_CAST_INCREMENTS_REFCOUNT,,[This should always be defined for Babel 0.11.0 and beyond])
  LLNL_WHICH_PROG(WHICH_CC)
  # a. Libraries (existence)
  # b. Header Files.
  AC_HEADER_DIRENT
  AC_HEADER_STDC
  AC_HEADER_STDBOOL
  AC_CHECK_HEADERS([argz.h float.h limits.h malloc.h memory.h netinet/in.h sched.h stddef.h stdlib.h string.h strings.h sys/socket.h unistd.h ctype.h sys/stat.h sys/time.h])
  AC_HEADER_TIME
  # c. Typedefs, Structs, Compiler Characteristics
  AC_C_CONST
  AC_TYPE_SIZE_T
  AC_TYPE_PID_T
  AC_CHECK_TYPES([ptrdiff_t])
  AC_CHECK_SIZEOF(short,2)
  AC_CHECK_SIZEOF(int,4)
  AC_CHECK_SIZEOF(long,8)
  LLNL_CHECK_LONG_LONG
  AC_CHECK_SIZEOF(long long,8)
  LLNL_FIND_32BIT_SIGNED_INT
  LLNL_CHECK_INT32_T
  LLNL_FIND_64BIT_SIGNED_INT
  LLNL_CHECK_INT64_T
  AC_CHECK_SIZEOF(void *,4)
  AC_C_INLINE
  AC_C_RESTRICT
  AC_C_VOLATILE
  LLNL_C_HAS_INLINE
  # d. Specific Library Functions.
  AC_FUNC_MALLOC
  AC_FUNC_REALLOC
  AC_FUNC_MEMCMP 
  AC_FUNC_STAT
  AC_FUNC_CLOSEDIR_VOID
  AC_FUNC_ERROR_AT_LINE
  AC_FUNC_FORK
  AC_FUNC_SELECT_ARGTYPES
  AC_LANG_C
  AC_LANG_PUSH([C])
  if test "$ac_compiler_gnu" = yes; then
    CFLAGS="$CFLAGS -fno-strict-aliasing"
  fi
  AC_LANG_POP([])
  AC_CHECK_FUNCS([atexit bzero getcwd memset socket strchr strdup strrchr])
  SHARED_LIB_VAR=${libltdl_cv_shlibpath_var}
  AC_SUBST(SHARED_LIB_VAR)
])
