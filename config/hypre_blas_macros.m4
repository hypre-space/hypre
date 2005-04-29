dnl @synopsis HYPRE_FIND_BLAS([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the BLAS
dnl linear-algebra interface (see http://www.netlib.org/blas/).
dnl On success, it sets the BLASLIBS output variable to
dnl hold the requisite library linkages.
dnl
dnl To link with BLAS, you should link with:
dnl
dnl 	$BLASLIBS $LIBS $FLIBS
dnl
dnl in that order.  FLIBS is the output variable of the
dnl AC_F77_LIBRARY_LDFLAGS macro, and is sometimes necessary in order to link
dnl with F77 libraries.
dnl
dnl Many libraries are searched for, from ATLAS to CXML to ESSL.
dnl The user may specify a BLAS library by using the --with-blas-libs=<lib>
dnl and --with-blas-lib-dirs=<dir> options.  In order to link successfully,
dnl however, be aware that you will probably need to use the same Fortran
dnl compiler (which can be set via the F77 env. var.) as was used to compile
dnl the BLAS library.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a BLAS
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands
dnl to run it if it is not found. 
dnl
dnl This macro requires autoconf 2.50 or later.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl
AC_DEFUN([HYPRE_FIND_BLAS],
[
  AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

#***************************************************************
#   Initialize return variables
#***************************************************************
  BLASLIBS="null"
  BLASLIBDIRS="null"

  AC_ARG_WITH(blas,
	[AS_HELP_STRING([  --with-blas], [Find a system-provided BLAS library])])

  case $with_blas in
      yes) ;;
        *) BLASLIBS="internal" ;;
  esac

#***************************************************************
#   Save incoming LIBS and LDFLAGS values to be restored 
#***************************************************************
  hypre_save_LIBS="$LIBS"
  hypre_save_LDFLGS="$LDFLAGS"
  LIBS="$LIBS $FLIBS"

#***************************************************************
#   Get fortran linker names for a BLAS function
#***************************************************************
  AC_F77_FUNC(dgemm)

#***************************************************************
#   Set possible BLAS library names
#***************************************************************
  BLAS_LIB_NAMES="blas essl dxml cxml mkl scs atlas complib.sgimath sunmath"

#***************************************************************
#   Set search paths for BLAS library
#***************************************************************
  temp_FLAGS="-L/usr/lib -L/usr/local/lib -L/lib -L/opt/intel/mkl70/lib/32"
  LDFLAGS="$temp_FLAGS $LDFLAGS"

#***************************************************************
#   Check for function dgemm in BLAS_LIB_NAMES
#***************************************************************
  for lib in $BLAS_LIB_NAMES; do
     if test "$BLASLIBS" = "null"; then
        AC_CHECK_LIB($lib, $dgemm, [BLASLIBS=$lib])
     fi
  done

#***************************************************************
#   Set path to selected BLAS library 
#***************************************************************
  BLAS_SEARCH_DIRS="/usr/lib /usr/local/lib /lib /opt/intel/mkl70/lib/32"

  if test "$BLASLIBS" != "null"; then
     for dir in $BLAS_SEARCH_DIRS; do
         if test "$BLASLIBDIRS" = "null" -a -f $dir/lib$BLASLIBS.a; then
            BLASLIBDIRS=$dir
         fi

         if test "$BLASLIBDIRS" = "null" -a -f $dir/lib$BLASLIBS.so; then
            BLASLIBDIRS=$dir
         fi
     done
  fi

#***************************************************************
#   Set variables if ATLAS or DMXL libraries are used 
#***************************************************************
  if test "$BLASLIBS" = "dxml"; then
     AC_DEFINE(HYPRE_USING_DXML, 1, [Using dxml for Blas])
  fi

  if test "$BLASLIBS" = "essl"; then
     AC_DEFINE(HYPRE_USING_ESSL, 1, [Using essl for Blas])
  fi

#***************************************************************
#   Add -L and -l prefixes if values found
#***************************************************************
  if test "$BLASLIBS" != "null" -a "$BLASLIBS" != "internal"; then
     BLASLIBS="-l$BLASLIBS"
  fi

  if test "$BLASLIBDIRS" != "null"; then
     BLASLIBDIRS="-L$BLASLIBDIRS"
  fi

#***************************************************************
#   Restore incoming LIBS and LDFLAGS values
#***************************************************************
  LIBS="$hypre_save_LIBS"
  LDFLAGS="$hypre_save_LDFLGS"

])dnl HYPRE_FIND_BLAS


dnl @synopsis HYPRE_SET_BLAS_FILES([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro defines the BLAS routines needed internally by hypre.  The source
dnl code is in the blas subdirectory of linear_solvers.
dnl
AC_DEFUN([HYPRE_SET_BLAS_FILES],
[

#*******************************************************
#   Source files
#*******************************************************
 FL1="HYPRE_TOP_SRC_DIR/blas/blas_utils.c HYPRE_TOP_SRC_DIR/blas/dasum.c"
 FL2="HYPRE_TOP_SRC_DIR/blas/daxpy.c HYPRE_TOP_SRC_DIR/blas/dcopy.c"
 FL3="HYPRE_TOP_SRC_DIR/blas/ddot.c HYPRE_TOP_SRC_DIR/blas/dgemm.c"
 FL4="HYPRE_TOP_SRC_DIR/blas/dgemv.c HYPRE_TOP_SRC_DIR/blas/dger.c"
 FL5="HYPRE_TOP_SRC_DIR/blas/dnrm2.c HYPRE_TOP_SRC_DIR/blas/drot.c"
 FL6="HYPRE_TOP_SRC_DIR/blas/dscal.c HYPRE_TOP_SRC_DIR/blas/dswap.c"
 FL7="HYPRE_TOP_SRC_DIR/blas/dsymm.c HYPRE_TOP_SRC_DIR/blas/dsymv.c"
 FL8="HYPRE_TOP_SRC_DIR/blas/dsyr2.c HYPRE_TOP_SRC_DIR/blas/dsyr2k.c"
 FL9="HYPRE_TOP_SRC_DIR/blas/dsyrk.c HYPRE_TOP_SRC_DIR/blas/dtrmm.c"
 F10="HYPRE_TOP_SRC_DIR/blas/dtrmv.c HYPRE_TOP_SRC_DIR/blas/dtrsm.c"
 F11="HYPRE_TOP_SRC_DIR/blas/dtrsv.c HYPRE_TOP_SRC_DIR/blas/idamax.c"

 HYPREBLASFILES="$FL1 $FL2 $FL3 $FL4 $FL5 $FL6 $FL7 $FL8 $FL9 $F10 $F11"

])dnl HYPRE_SET_BLAS_FILES
