dnl @synopsis HYPRE_FIND_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the LAPACK
dnl linear-algebra interface (see http://www.netlib.org/lapack/).
dnl On success, it sets the LAPACK_LIBS output variable to
dnl hold the requisite library linkages.
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl     $LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS
dnl
dnl in that order.  BLAS_LIBS is the output variable of the HYPRE_FIND_BLAS
dnl macro, called automatically.  FLIBS is the output variable of the
dnl AC_F77_LIBRARY_LDFLAGS macro (called if necessary by HYPRE_FIND_BLAS),
dnl and is sometimes necessary in order to link with F77 libraries.
dnl Users will also need to use AC_F77_DUMMY_MAIN (see the autoconf
dnl manual), for the same reason.
dnl
dnl The user may also use --with-lapack_liband --with-lapack_dirs in order
dnl to use a specific LAPACK library <lib>.  In order to link successfully,
dnl however, be aware that you will probably need to use the same
dnl Fortran compiler (which can be set via the F77 env. var.) as
dnl was used to compile the LAPACK and BLAS libraries.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a LAPACK
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands
dnl to run it if it is not found.  If ACTION-IF-FOUND is not specified,
dnl the default action will define HAVE_LAPACK.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>

AC_DEFUN([HYPRE_FIND_LAPACK], 
[
  AC_REQUIRE([HYPRE_FIND_BLAS])

  hypre_lapack_ok=no
  LAPACKLIBS=""

  AC_ARG_WITH(lapack,
        [AS_HELP_STRING([  --with-lapack], [Find a system-provided LAPACK library])])

  case $with_lapack in
      yes | "") ;;
             *) LAPACKLIBS="internal";
                hypre_lapack_ok=internal ;;
  esac

# Get fortran linker name of LAPACK function to check for.
  AC_F77_FUNC(dsygv)

  hypre_lapack_save_LIBS="$LIBS"

# First, check LAPACKLIBS environment variable
  if test $hypre_lapack_ok = no; then
    if test "x$LAPACKLIBS" != x; then
        save_LIBS="$LIBS"; LIBS="$LAPACKLIBS $BLASLIBS $LIBS $FLIBS"
        AC_MSG_CHECKING([for $dsygv in $LAPACKLIBS])
        AC_TRY_LINK_FUNC($dsygv, [hypre_lapack_ok=yes], [LAPACKLIBS=""])
        AC_MSG_RESULT($hypre_lapack_ok)
        LIBS="$save_LIBS"
    fi
  fi

# LAPACK linked to by default?  (is sometimes included in BLAS lib)
  if test $hypre_lapack_ok = no; then
        save_LIBS="$LIBS"; LIBS="$LIBS $BLASLIBS $FLIBS"
        AC_CHECK_FUNC($dsygv, [hypre_lapack_ok=yes])
        LIBS="$save_LIBS"
  fi

# Generic LAPACK library?
  for lapack in lapack lapack_rs6k; do
        if test $hypre_lapack_ok = no; then
                save_LIBS="$LIBS"; LIBS="$BLASLIBS $LIBS"
                AC_CHECK_LIB($lapack, $dsygv,
                    [hypre_lapack_ok=yes; LAPACKLIBS="-l$lapack"], [], [$FLIBS])
                LIBS="$save_LIBS"
        fi
  done

### if no lapack library is found; set to force configuring without-lapack.
if test $hypre_lapack_ok = no; then
         LAPACKLIBS="no"
fi

])dnl HYPRE_FIND_LAPACK
