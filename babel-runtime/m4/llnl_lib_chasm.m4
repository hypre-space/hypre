dnl autoconf macros for LANL's chasm
dnl
dnl @synopsis LLNL_LIB_CHASM
dnl
dnl Make sure CHASM is installed, configured,
dnl and that the include files are located.
dnl

AC_DEFUN([LLNL_LIB_CHASM], 
[ AC_ARG_ENABLE([chasm],
    AC_HELP_STRING([--enable-chasm@<:@=prefix@:>@],
                   [chasm F90 array descriptor library @<:@default=yes@:>@]),
	 	   [enable_chasm="$enableval"],
		   [enable_chasm=yes])
  AC_ARG_VAR([CHASMPREFIX],[Directory where chasm's include/ and lib/ are installed.])
  CHASM_CFLAGS=""
  CHASM_LIBS=""
  llnl_cv_chasm_fortran_vendor=""
  chasm_prefix="$CHASMPREFIX"
  if test  ! \( \( -z "$enable_chasm" \) -o \( "$enable_chasm" = yes \) \); then
    if test $enable_chasm != no; then
      chasm_prefix="$enable_chasm"
      enable_chasm=yes
    fi
  fi
  chasm_prefix=`echo "$chasm_prefix" | sed -e 's,//*$,,g'`
  if test "X$enable_chasm" = "Xyes"; then
    AC_MSG_CHECKING([checking for chasm...])
    AC_LANG_SAVE
    save_LIBS=$LIBS
    save_CFLAGS=$CFLAGS
    if test -n "$chasm_prefix"; then
      CHASM_LIBS="-L$chasm_prefix/lib -lchasm"
      CHASM_CFLAGS="-I$chasm_prefix/include"
    fi
    AC_LANG_C
    LIBS="$LIBS $CHASM_LIBS"
    CFLAGS="$CFLAGS $CHASM_CFLAGS"
    AC_TRY_LINK([#include <CompilerCharacteristics.h>
#include <F90ArrayDataType.h>
#include <F90Compiler.h>
#include <stdio.h>],[
  int i, size, maxSize = 0;
  F90_CompilerCharacteristics cc;
  if (i = F90_SetCompilerCharacteristics(&cc, FORTRAN_COMPILER)) {
    return i;
  }
  for(i = 1; i <= 7; ++i) {
    size = (cc.getArrayDescSize)(i);
    if (size > maxSize) maxSize = size;
  }
  printf("%d\n", maxSize);
  return 0;
], [
  chasm_max_descriptor_size=`./conftest$ac_exeext 2>/dev/null`
  if ./conftest$ac_exeext > /dev/null 2>&1; then
    AC_MSG_RESULT([yes])
    AC_MSG_CHECKING([checking the maximum F90 array description...])
    AC_MSG_RESULT($chasm_max_descriptor_size)
    AC_DEFINE_UNQUOTED(SIDL_MAX_F90_DESCRIPTOR, $chasm_max_descriptor_size,
       [the maximum size in bytes of a F90 array descriptor])
    AC_MSG_CHECKING([The compiler type Chasm is configured for])
  llnl_cv_chasm_fortran_vendor=`grep 'define FORTRAN_COMPILER' $chasm_prefix/include/*.h | sed 's/.*\"\(.*\)\"/\1/;'`
    AC_MSG_RESULT([$llnl_cv_chasm_fortran_vendor])
    CHASM_FORTRAN_VENDOR=$llnl_cv_chasm_fortran_vendor
    AC_MSG_CHECKING([The Fortran compiler option for specifying a module search path (from Chasm)])
    llnl_cv_chasm_fortran_mflag=`grep 'CHASM_F90MFLAG' $chasm_prefix/include/MakeIncl.chasm | sed 's/ *CHASM_F90MFLAG *= *//g'`
    AC_MSG_RESULT([$llnl_cv_chasm_fortran_mflag])
    CHASM_FORTRAN_MFLAG="$llnl_cv_chasm_fortran_mflag"
  else
    AC_MSG_RESULT([no])
    AC_MSG_WARN([Unable to determine maximum array descriptor size $chasm_max_descriptor_size])
    AC_MSG_WARN([Disabling chasm])
    
    enable_chasm=no
  fi
], [
  AC_MSG_RESULT([no])
  AC_MSG_WARN([Unable to compile and link to chasm -- disabling chasm])
  enable_chasm=no
])

    LIBS=$save_LIBS
    CFLAGS=$save_CFLAGS
    AC_LANG_RESTORE
  fi
  AC_SUBST(CHASM_CFLAGS)
  AC_SUBST(CHASM_LIBS)
  AC_SUBST(CHASM_FORTRAN_VENDOR)
  AC_SUBST(CHASM_FORTRAN_MFLAG)
])
