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

  hypre_lapack_ok=no

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

# Is LAPACKLIBS environment variable set?
  if test $hypre_lapack_ok = no; then
    if test "x$LAPACKLIBS" != x; then
        save_LIBS="$LIBS"; LIBS="$LAPACKLIBS $BLASLIBS $LIBS $FLIBS"
        AC_MSG_CHECKING([for $dsygv in $LAPACKLIBS])
        AC_TRY_LINK_FUNC($dsygv, [hypre_lapack_ok=yes], [LAPACKLIBS=""])
        AC_MSG_RESULT($hypre_lapack_ok)
        LIBS="$save_LIBS"
    fi
  fi

# LAPACK included in BLAS lib?
  if test $hypre_lapack_ok = no; then
        save_LIBS="$LIBS"; LIBS="$LIBS $BLASLIBS $FLIBS"
        AC_CHECK_FUNC($dsygv, [hypre_lapack_ok=yes; LAPACKLIBS="$BLASLIBS"])
        LIBS="$save_LIBS"
  fi

# LAPACK linked to by default? 
  if test $hypre_lapack_ok = no; then
        save_LIBS="$LIBS"; LIBS="$LIBS"
        AC_CHECK_FUNC($dsygv, [hypre_lapack_ok=yes; LAPACKLIBS="$LIBS"])
        LIBS="$save_LIBS"
  fi

# Generic LAPACK library
  if test $hypre_lapack_ok = no; then
     save_LIBS="$LIBS"; LIBS="$LIBS $FLIBS"
     save_LDFLAGS="$LDFLAGS"
     LDFLAGS="-L/usr/lib -L/usr/local/lib $LDFLAGS"
     AC_CHECK_LIB(lapack, $dsygv, [hypre_lapack_ok=yes; LAPACKLIBS="-llapack"],
                              [], [-lblas])
     LIBS="$save_LIBS"
     LDFLAGS="$save_LDFLAGS"
  fi

# Generic LAPACK_RS6K library
  if test $hypre_lapack_ok = no; then
     save_LIBS="$LIBS"; LIBS="$LIBS $FLIBS"
     save_LDFLAGS="$LDFLAGS"
     LDFLAGS="-L/usr/lib -L/usr/local/lib $LDFLAGS"
     AC_CHECK_LIB(lapack_rs6k, $dsygv, [hypre_lapack_ok=yes; LAPACKLIBS="-llapack_rs6k"],
                              [], [-lblas])
     LIBS="$save_LIBS"
     LDFLAGS="$save_LDFLAGS"
  fi

  LIBS="$hypre_lapack_save_libs"

### if no lapack library is found; set to force configuring without-lapack.
if test $hypre_lapack_ok = no; then
         LAPACKLIBS="no"
fi

])dnl HYPRE_FIND_LAPACK
