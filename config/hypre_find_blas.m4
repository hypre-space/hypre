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
dnl AC_F77_LIBRARY_LDFLAGS macro (called if necessary by ACX_BLAS),
dnl and is sometimes necessary in order to link with F77 libraries.
dnl Users will also need to use AC_F77_DUMMY_MAIN (see the autoconf
dnl manual), for the same reason.
dnl
dnl Many libraries are searched for, from ATLAS to CXML to ESSL.
dnl The user may specify a BLAS library by using the --with-blas-libs and
dnl --with-blas-dirs options.  In order to link successfully,
dnl however, be aware that you will probably need to use the same
dnl Fortran compiler (which can be set via the F77 env. var.) as
dnl was used to compile the BLAS library.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a BLAS
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands
dnl to run it if it is not found.  If ACTION-IF-FOUND is not specified,
dnl the default action will define HAVE_BLAS.
dnl
dnl This macro requires autoconf 2.50 or later.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl
AC_DEFUN([HYPRE_FIND_BLAS],
[
  AC_PREREQ(2.50)
  AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

  hypre_blas_ok=no
  BLASLIBS=""

  AC_ARG_WITH(blas,
	[AS_HELP_STRING([  --with-blas], [Find a system-provided BLAS library])])

  case $with_blas in
      yes | "") ;;
	     *) BLASLIBS="internal";
                hypre_blas_ok=internal ;;
  esac

# Get fortran linker names of BLAS functions to check for.
  AC_F77_FUNC(dgemm)

  hypre_blas_save_LIBS="$LIBS"
  LIBS="$LIBS $FLIBS"

# First, check BLASLIBS environment variable
  if test $hypre_blas_ok = no; then
    if test "x$BLASLIBS" != x; then
	save_LIBS="$LIBS"; LIBS="$BLASLIBS $LIBS"
	AC_MSG_CHECKING([for $dgemm in $BLASLIBS])
	AC_TRY_LINK_FUNC($dgemm, [hypre_blas_ok=yes], [BLASLIBS=""])
	AC_MSG_RESULT($hypre_blas_ok)
	LIBS="$save_LIBS"
    fi
  fi

# BLAS linked to by default?  (happens on some supercomputers)
  if test $hypre_blas_ok = no; then
	save_LIBS="$LIBS"; LIBS="$LIBS"
	AC_CHECK_FUNC($dgemm, [hypre_blas_ok=yes])
	LIBS="$save_LIBS"
  fi

# BLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(atlas, ATL_xerbla,
		[AC_CHECK_LIB(f77blas, $dgemm,
		[AC_CHECK_LIB(cblas, cblas_dgemm,
			[hypre_blas_ok=yes
			 BLASLIBS="-lcblas -lf77blas -latlas"],
			[], [-lf77blas -latlas])],
			[], [-latlas])])
  fi

# BLAS in Alpha CXML library?
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(cxml, $dgemm, [hypre_blas_ok=yes;BLASLIBS="-lcxml"])
  fi

# BLAS in Alpha DXML library? (now called CXML, see above)
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(dxml, $dgemm, [hypre_blas_ok=yes;BLASLIBS="-ldxml"
	AC_DEFINE(HYPRE_USING_DXML, 1, [Using dxml for Blas])])
  fi

# BLAS in Sun Performance library?
  if test $hypre_blas_ok = no; then
	if test "x$GCC" != xyes; then # only works with Sun CC
		AC_CHECK_LIB(sunmath, acosp,
			[AC_CHECK_LIB(sunperf, $dgemm,
        			[BLASLIBS="-xlic_lib=sunperf -lsunmath"
                                 hypre_blas_ok=yes],[],[-lsunmath])])
	fi
  fi

# BLAS in SCSL library?  (SGI/Cray Scientific Library)
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(scs, $dgemm, [hypre_blas_ok=yes; BLASLIBS="-lscs"])
  fi

# BLAS in SGIMATH library?
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(complib.sgimath, $dgemm,
		     [hypre_blas_ok=yes; BLASLIBS="-lcomplib.sgimath"])
  fi

# BLAS in IBM ESSL library? 
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(essl, $dgemm,
			[hypre_blas_ok=yes; BLASLIBS="-lessl"
	AC_DEFINE(HYPRE_USING_ESSL, 1, [Using essl for Blas])])
  fi

# Generic BLAS library? 
  if test $hypre_blas_ok = no; then
	AC_CHECK_LIB(blas, $dgemm, [hypre_blas_ok=yes; BLASLIBS="-lblas"])
  fi

  LIBS="$hypre_blas_save_LIBS"

### if no blas library was found; set to force configuring without-blas
  if test $hypre_blas_ok = no; then
         BLASLIBS="no"
  fi

])dnl HYPRE_FIND_BLAS
