dnl
dnl @synopsis LLNL_CONFIRM_BABEL_C_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_C_SUPPORT], [
  ############################################################
  #
  # C Compiler
  #
  AC_PROG_CC
  LLNL_WHICH_PROG(WHICH_CC)
  # a. Libraries (existence)
  # b. Header Files.
  AC_HEADER_DIRENT
  AC_HEADER_STDC
  AC_HEADER_STDBOOL
  AC_CHECK_HEADERS([float.h inttypes.h limits.h malloc.h memory.h netinet/in.h stddef.h stdlib.h string.h strings.h sys/socket.h unistd.h ctype.h sys/stat.h sys/types.h])
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
  AC_C_RESTRICT
  AC_C_VOLATILE
  # d. Specific Library Functions.
  AC_FUNC_MEMCMP 
  AC_FUNC_STAT
  AC_FUNC_CLOSEDIR_VOID
  AC_FUNC_ERROR_AT_LINE
  AC_LANG_SAVE
  AC_LANG_C
  if test "$ac_compiler_gnu" = yes; then
    CFLAGS="$CFLAGS -fno-strict-aliasing"
  fi
  AC_LANG_RESTORE
  AC_CHECK_FUNCS([atexit bzero getcwd memset socket strchr strdup strrchr])
])
