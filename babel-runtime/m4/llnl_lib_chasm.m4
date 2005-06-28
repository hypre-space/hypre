#####################################
#
# LLNL_LIB_CHASM
#
# Check for CHASM_PREFIX (if set) and commandline.
# 


AC_DEFUN([LLNL_LIB_CHASM], 
[ 
  AC_MSG_CHECKING([if CHASM is requested])
  # declare CHASM_PREFIX as important
  AC_ARG_VAR([CHASMPREFIX],[Directory where chasm's include/ and lib/ are installed.])
  ac_arg_with_chasm=no
  test "${CHASMPREFIX+set}" = set &&  ac_arg_with_chasm=yes
  # set --with-chasm flag
  AC_ARG_WITH([chasm],
      AS_HELP_STRING(--with-chasm@<:@=prefix@:>@,chasm F90 array descriptor library @<:@default=yes@:>@),
      [ case $withval in
          no) ac_arg_with_chasm=no ;;
          yes) ac_arg_with_chasm=yes ;;
          *) ac_arg_with_chasm=yes;
             CHASMPREFIX="$withval" ;;
        esac]) # end AC_ARG_WITH
  AC_MSG_RESULT([$ac_arg_with_chasm])

  llnl_cv_chasm_fortran_vendor=""
  CHASM_CFLAGS=""
  CHASM_LIBS=""
  chasm_prefix=`echo "$CHASMPREFIX" | sed -e 's,//*$,,g'`

  if test $ac_arg_with_chasm = yes; then
    AC_MSG_CHECKING([for working CHASM home...])
    save_LIBS=$LIBS
    save_CFLAGS=$CFLAGS
    if test -n "$chasm_prefix"; then
      CHASM_LIBS="-L$chasm_prefix/lib -lchasm"
      CHASM_CFLAGS="-I$chasm_prefix/include"
    fi
    AC_LANG_PUSH([C])
    LIBS="$LIBS $CHASM_LIBS"
    CFLAGS="$CFLAGS $CHASM_CFLAGS"
    AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <CompilerCharacteristics.h>
#include <F90ArrayDataType.h>
#include <F90Compiler.h>
#include <stdio.h>]], [[
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
]])],[
  chasm_max_descriptor_size=`./conftest$ac_exeext 2>/dev/null`
  if ./conftest$ac_exeext > /dev/null 2>&1; then
    enable_chasm=yes
    AC_MSG_RESULT($chasm_prefix)
    AC_MSG_CHECKING([the maximum F90 array description...])
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
    enable_chasm=no
    AC_MSG_RESULT([no])
    AC_MSG_WARN([Unable to determine maximum array descriptor size $chasm_max_descriptor_size])
    AC_MSG_WARN([Disabling chasm])    
  fi
],[
  enable_chasm=no
  AC_MSG_RESULT([no])
  AC_MSG_WARN([Unable to compile and link to chasm -- disabling chasm])
])

    LIBS=$save_LIBS
    CFLAGS=$save_CFLAGS
    AC_LANG_POP([])
  fi
  AC_SUBST(CHASM_CFLAGS)
  AC_SUBST(CHASM_LIBS)
  AC_SUBST(CHASM_FORTRAN_VENDOR)
  AC_SUBST(CHASM_FORTRAN_MFLAG)
]) # end LLNL_LIB_CHASM
