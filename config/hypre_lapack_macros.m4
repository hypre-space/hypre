dnl @synopsis HYPRE_FIND_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the LAPACK
dnl linear-algebra interface (see http://www.netlib.org/lapack/).
dnl On success, it sets the LAPACKLIBS output variable to
dnl hold the requisite library linkages.
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl     $LAPACKLIBS $BLASLIBS $LIBS $FLIBS
dnl
dnl in that order.  BLASLIBS is either the output variable of the HYPRE_FIND_BLAS
dnl macro (which is called by configure before this macro) or the user-defined 
dnl blas library.  FLIBS is the output variable of the AC_F77_LIBRARY_LDFLAGS 
dnl macro, which is sometimes necessary in order to link with F77 libraries. 
dnl
dnl The user may use --with-lapack-libs=<lib> and --with-lapack-lib-dirs=<dir>
dnl in order to use a specific LAPACK library <lib>.  In order to link successfully,
dnl however, be aware that you will probably need to use the same Fortran compiler
dnl (which can be set via the F77 env. var.) as was used to compile the LAPACK and
dnl BLAS libraries.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a LAPACK
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands
dnl to run it if it is not found.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>

AC_DEFUN([HYPRE_FIND_LAPACK], 
[
  AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

#***************************************************************
#   Initialize return variables
#***************************************************************
  LAPACKLIBS="null"
  LAPACKLIBDIRS="null"

  AC_ARG_WITH(lapack,
        [AS_HELP_STRING([  --with-lapack], [Find a system-provided LAPACK library])])

  case $with_lapack in
      yes) ;;
        *) LAPACKLIBS="internal" ;;
  esac

#***************************************************************
#   Save incoming LIBS and LDFLAGS values to be restored
#***************************************************************
  hypre_save_LIBS="$LIBS"
  hypre_save_LDFLGS="$LDFLAGS"
  LIBS="$LIBS $FLIBS"

#***************************************************************
# Get fortran linker name of LAPACK function to check for.
#***************************************************************
  AC_F77_FUNC(dsygv)

#***************************************************************
#   Set possible LAPACK library names
#***************************************************************
  LAPACK_LIB_NAMES="lapack lapack_rs6k"

#***************************************************************
#   Set search paths for LAPACK library 
#***************************************************************
  temp_FLAGS="-L/usr/lib -L/usr/local/lib -L/lib"
  LDFLAGS="$temp_FLAGS $LDFLAGS"

#***************************************************************
#   Check for function dsygv in LAPACK_LIB_NAMES
#***************************************************************
  for lib in $LAPACK_LIB_NAMES; do
     if test "$LAPACKLIBS" = "null"; then
        AC_CHECK_LIB($lib, $dsygv, [LAPACKLIBS=$lib], [], [-lblas])
     fi
  done

#***************************************************************
#   Set path to selected LAPACK library
#***************************************************************
  LAPACK_SEARCH_DIRS="/usr/lib /usr/local/lib /lib"

  if test "$LAPACKLIBS" != "null"; then
     for dir in $LAPACK_SEARCH_DIRS; do
         if test "$LAPACKLIBDIRS" = "null" -a -f $dir/lib$LAPACKLIBS.a; then
            LAPACKLIBDIRS=$dir
         fi

         if test "$LAPACKLIBDIRS" = "null" -a -f $dir/lib$LAPACKLIBS.so; then
            LAPACKLIBDIRS=$dir
         fi
     done
  fi

#***************************************************************
#   Add -L and -l prefixes if values found
#***************************************************************
  if test "$LAPACKLIBS" != "null" -a "$LAPACKLIBS" != "internal"; then
     LAPACKLIBS="-l$LAPACKLIBS"
  fi

  if test "$LAPACKLIBDIRS" != "null"; then
     LAPACKLIBDIRS="-L$LAPACKLIBDIRS"
  fi

#***************************************************************
#   Restore incoming LIBS and LDFLAGS values
#***************************************************************
  LIBS="$hypre_save_LIBS"
  LDFLAGS="$hypre_save_LDFLGS"

])dnl HYPRE_FIND_LAPACK


dnl @synopsis HYPRE_SET_LAPACK_FILES([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro defines the LAPACK routiens needed internally by hypre.  The
dnl source code is inthe lapack subdirectory of linear_solvers.

AC_DEFUN([HYPRE_SET_LAPACK_FILES], 
[

#***************************************************************
#  Source files 
#***************************************************************
  FL1="HYPRE_TOP_SRC_DIR/lapack/dbdsqr.c HYPRE_TOP_SRC_DIR/lapack/dgebd2.c"
  FL2="HYPRE_TOP_SRC_DIR/lapack/dgebrd.c HYPRE_TOP_SRC_DIR/lapack/dgelq2.c"
  FL3="HYPRE_TOP_SRC_DIR/lapack/dgelqf.c HYPRE_TOP_SRC_DIR/lapack/dgels.c"
  FL4="HYPRE_TOP_SRC_DIR/lapack/dgeqr2.c HYPRE_TOP_SRC_DIR/lapack/dgeqrf.c"
  FL5="HYPRE_TOP_SRC_DIR/lapack/dgesvd.c HYPRE_TOP_SRC_DIR/lapack/dlabad.c"
  FL6="HYPRE_TOP_SRC_DIR/lapack/dlabrd.c HYPRE_TOP_SRC_DIR/lapack/dlacpy.c"
  FL7="HYPRE_TOP_SRC_DIR/lapack/dlae2.c HYPRE_TOP_SRC_DIR/lapack/dlaev2.c"
  FL8="HYPRE_TOP_SRC_DIR/lapack/dlamch.c HYPRE_TOP_SRC_DIR/lapack/dlange.c"
  FL9="HYPRE_TOP_SRC_DIR/lapack/dlanst.c HYPRE_TOP_SRC_DIR/lapack/dlansy.c"
  F10="HYPRE_TOP_SRC_DIR/lapack/dlapy2.c HYPRE_TOP_SRC_DIR/lapack/dlarfb.c"
  F11="HYPRE_TOP_SRC_DIR/lapack/dlarf.c HYPRE_TOP_SRC_DIR/lapack/dlarfg.c"
  F12="HYPRE_TOP_SRC_DIR/lapack/dlarft.c HYPRE_TOP_SRC_DIR/lapack/dlartg.c"
  F13="HYPRE_TOP_SRC_DIR/lapack/dlas2.c HYPRE_TOP_SRC_DIR/lapack/dlascl.c"
  F14="HYPRE_TOP_SRC_DIR/lapack/dlaset.c HYPRE_TOP_SRC_DIR/lapack/dlasq1.c"
  F15="HYPRE_TOP_SRC_DIR/lapack/dlasq2.c HYPRE_TOP_SRC_DIR/lapack/dlasq3.c"

  F16="HYPRE_TOP_SRC_DIR/lapack/dlasq4.c HYPRE_TOP_SRC_DIR/lapack/dlasq5.c"
  F17="HYPRE_TOP_SRC_DIR/lapack/dlasq6.c HYPRE_TOP_SRC_DIR/lapack/dlasr.c"
  F18="HYPRE_TOP_SRC_DIR/lapack/dlasrt.c HYPRE_TOP_SRC_DIR/lapack/dlassq.c"
  F19="HYPRE_TOP_SRC_DIR/lapack/dlasv2.c HYPRE_TOP_SRC_DIR/lapack/dlatrd.c"
  F20="HYPRE_TOP_SRC_DIR/lapack/dorg2l.c HYPRE_TOP_SRC_DIR/lapack/dorg2r.c"
  F21="HYPRE_TOP_SRC_DIR/lapack/dorgbr.c HYPRE_TOP_SRC_DIR/lapack/dorgl2.c"
  F22="HYPRE_TOP_SRC_DIR/lapack/dorglq.c HYPRE_TOP_SRC_DIR/lapack/dorgql.c"
  F23="HYPRE_TOP_SRC_DIR/lapack/dorgqr.c HYPRE_TOP_SRC_DIR/lapack/dorgtr.c"
  F24="HYPRE_TOP_SRC_DIR/lapack/dorm2r.c HYPRE_TOP_SRC_DIR/lapack/dormbr.c"
  F25="HYPRE_TOP_SRC_DIR/lapack/dorml2.c HYPRE_TOP_SRC_DIR/lapack/dormlq.c"
  F26="HYPRE_TOP_SRC_DIR/lapack/dormqr.c HYPRE_TOP_SRC_DIR/lapack/dpotf2.c"
  F27="HYPRE_TOP_SRC_DIR/lapack/dpotrf.c HYPRE_TOP_SRC_DIR/lapack/dpotrs.c"
  F28="HYPRE_TOP_SRC_DIR/lapack/dsteqr.c HYPRE_TOP_SRC_DIR/lapack/dsterf.c"
  F29="HYPRE_TOP_SRC_DIR/lapack/dsyev.c HYPRE_TOP_SRC_DIR/lapack/dsygs2.c"
  F30="HYPRE_TOP_SRC_DIR/lapack/dsygst.c HYPRE_TOP_SRC_DIR/lapack/dsygv.c"

  F31="HYPRE_TOP_SRC_DIR/lapack/dsytd2.c HYPRE_TOP_SRC_DIR/lapack/dsytrd.c"
  F32="HYPRE_TOP_SRC_DIR/lapack/ieeeck.c HYPRE_TOP_SRC_DIR/lapack/ilaenv.c"
  F33="HYPRE_TOP_SRC_DIR/lapack/lapack_utils.c"
  F34="HYPRE_TOP_SRC_DIR/lapack/lsame.c HYPRE_TOP_SRC_DIR/lapack/xerbla.c"

  Fil0="$FL1 $FL2 $FL3 $FL4 $FL5 $FL6 $FL7 $FL8 $FL9 $F10"
  Fil1="$F11 $F12 $F13 $F14 $F15"
  Fil2="$F16 $F17 $F18 $F19 $F20 $F21 $F22 $F23 $F24 $F25"
  Fil3="$F26 $F27 $F28 $F29 $F30"
  Fil4="$F31 $F32 $F33 $F34"

  HYPRELAPACKFILS1="$Fil0 $Fil1"
  HYPRELAPACKFILS2="$Fil2 $Fil3"
  HYPRELAPACKFILS3="$Fil4"

])dnl HYPRE_SET_LAPACK_FILES
