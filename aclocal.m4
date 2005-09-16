# aclocal.m4 generated automatically by aclocal 1.6.3 -*- Autoconf -*-

# Copyright 1996, 1997, 1998, 1999, 2000, 2001, 2002
# Free Software Foundation, Inc.
# This file is free software; the Free Software Foundation
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.

dnl **********************************************************************
dnl * ACX_CHECK_MPI
dnl *
dnl try to determine what the MPI flags should be
dnl ACX_CHECK_MPI([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl ACTION-IF-FOUND is a list of shell commands to run 
dnl   if an MPI library is found, and
dnl ACTION-IF-NOT-FOUND is a list of commands to run it 
dnl   if it is not found. If ACTION-IF-FOUND is not specified, 
dnl   the default action will define HAVE_MPI. 
dnl
AC_DEFUN([ACX_CHECK_MPI],
[AC_PREREQ(2.57)dnl
AC_PREREQ(2.50) dnl for AC_LANG_CASE

if test x = x"$MPILIBS"; then
  AC_LANG_CASE([C], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
    [C++], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
    [Fortran 77], [AC_MSG_CHECKING([for MPI_Init])
      AC_TRY_LINK([],[      call MPI_Init], [MPILIBS=" "
        AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)])])
fi

if test x = x"$MPILIBS"; then
  AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
fi

if test x = x"$MPILIBS"; then
  AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
fi

dnl We have to use AC_TRY_COMPILE and not AC_CHECK_HEADER because the
dnl latter uses $CPP, not $CC (which may be mpicc).
AC_LANG_CASE([C], [if test x != x"$MPILIBS"; then
  AC_MSG_CHECKING([for mpi.h])
  AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [MPILIBS=""
                     AC_MSG_RESULT(no)])
fi],
[C++], [if test x != x"$MPILIBS"; then
  AC_MSG_CHECKING([for mpi.h])
  AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [MPILIBS=""
                     AC_MSG_RESULT(no)])
fi])

AC_SUBST(MPILIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x = x"$MPILIBS"; then
  $2
  :
else
  AC_DEFINE(HAVE_MPI,1,[Found the MPI library.])
  $1
  :
fi
])

dnl **********************************************************************
dnl * ACX_TIMING
dnl *
dnl determine timing routines to use
dnl
AC_DEFUN([ACX_TIMING],
[AC_PREREQ(2.57)dnl
AC_ARG_WITH(timing,
AC_HELP_STRING([--with-timing],[use HYPRE timing routines]),
[if test "$withval" = "yes"; then
  AC_DEFINE(HYPRE_TIMING,1,[HYPRE timing routines are being used])
fi])
])

dnl **********************************************************************
dnl * ACX_OPENMP
dnl *
dnl compile with OpenMP
dnl
AC_DEFUN([ACX_OPENMP],
[AC_PREREQ(2.57)dnl
AC_ARG_WITH(openmp,
AC_HELP_STRING([--with-openmp],
[use openMP--this may affect which compiler is chosen.
Supported using guidec on IBM and Compaq.]),
[case "${withval}" in
  yes) casc_using_openmp=yes
    AC_DEFINE([HYPRE_USING_OPENMP], 1, [Enable OpenMP support]) ;;
  no)  casc_using_openmp=no;;
  *) AC_MSG_ERROR(bad value ${withval} for --with-openmp) ;;
esac],[casc_using_openmp=no])
])

dnl **********************************************************************
dnl * HYPRE_FIND_G2C
dnl *  try to find libg2c.a
dnl **********************************************************************
AC_DEFUN([HYPRE_FIND_G2C],
[
 AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

  hypre_save_LIBS="$LIBS"
  LIBS="$LIBS $FLIBS"

  found_g2c=no

  g2c_GCC_PATH="-L/usr/lib/gcc-lib/i386-redhat-linux/3.2.3"
  g2c_SEARCH_PATHS="$g2c_GCC_PATH -L/usr/lib -L/usr/local/lib -L/usr/apps/lib -L/lib"

  LDFLAGS="$g2c_SEARCH_PATHS $LDFLAGS"

  AC_CHECK_LIB(g2c, e_wsfe, [found_g2c=yes])

  if test "$found_g2c" = "yes"; then
     LIBS="-lg2c $hypre_save_LIBS"
  else
     LIBS="$hypre_save_LIBS"
  fi

])


dnl **********************************************************************
dnl * HYPRE_REVERSE_FLIBS
dnl *   reverse the order of -lpmpich and -lmpich ONLY when using insure
dnl *   Search FLIBS to find -lpmpich, when found reverse the order with
dnl *      mpich; ignore the -lmpich when found; save all other FLIBS 
dnl *      values
dnl **********************************************************************
AC_DEFUN([HYPRE_REVERSE_FLIBS],
[
 AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

  hypre_save_FLIBS="$FLIBS"
  FLIBS=

  for lib_list in $hypre_save_FLIBS; do
     tmp_list="$lib_list"
     if test "$lib_list" = "-lpmpich"
     then
        tmp_list="-lmpich"
     fi

     if test "$lib_list" = "-lmpich"
     then
        tmp_list="-lpmpich"
     fi

     FLIBS="$FLIBS $tmp_list"
  done

])

dnl **********************************************************************
dnl * ACX_OPTIMIZATION_FLAGS
dnl *
dnl try and determine what the optimized compile FLAGS
dnl
AC_DEFUN([ACX_OPTIMIZATION_FLAGS],
[AC_PREREQ(2.57)dnl
if test "x${casc_user_chose_cflags}" = "xno"
then
  if test "x${GCC}" = "xyes"
  then
    dnl **** default settings for gcc
    CFLAGS="-O2"
  else
    case "${CC}" in
      kcc|mpikcc)
        CFLAGS="-fast +K3"
        ;;
      KCC|mpiKCC)
        CFLAGS="--c -fast +K3"
        ;;
      icc)
        CFLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -openmp"
        fi
        ;;
      pgcc|mpipgcc)
        CFLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -mp"
        fi
        ;;
      cc|c89|mpcc|mpiicc|xlc|ccc)
        case "${host}" in
          alpha*-dec-osf4.*)
            CFLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            CFLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -omp"
            fi
            ;;
          hppa*-hp-hpux*)
            CFLAGS="-Aa -D_HPUX_SOURCE -O"
            ;;
          mips-sgi-irix6.[[4-9]]*)
            CFLAGS="-Ofast -64 -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            CFLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            CFLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            CFLAGS="-O3 -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -qsmp=omp"
            fi
            ;;
          *)
            CFLAGS="-O"
            ;;
        esac
        ;;
      *)
        CFLAGS="-O"
        ;;
    esac
  fi
fi
if test "x${casc_user_chose_cxxflags}" = "xno"
then
  if test "x${GXX}" = "xyes"
  then
    dnl **** default settings for gcc
    CXXFLAGS="-O2"
  else
    case "${CXX}" in
      KCC|mpiKCC)
        CXXFLAGS="-fast +K3"
        ;;
      icc)
        CXXFLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -openmp"
        fi
        ;;
      pgCC|mpipgCC)
        CXXFLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -mp"
        fi
        ;;
      CC|aCC|mpCC|mpiicc|xlC|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            CXXFLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            CXXFLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -omp"
            fi
            ;;
          hppa*-hp-hpux*)
            CXXFLAGS="-D_HPUX_SOURCE -O"
            ;;
          mips-sgi-irix6.[[4-9]]*)
            CXXFLAGS="-Ofast -64 -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            CXXFLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            CXXFLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            CXXFLAGS="-O3 -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -qsmp=omp"
            fi
            ;;
          *)
            CXXFLAGS="-O"
            ;;
        esac
        ;;
      *)
        CXXFLAGS="-O"
        ;;
    esac
  fi
fi
if test "x${casc_user_chose_fflags}" = "xno"
then
  if test "x${G77}" = "xyes"
  then
    FFLAGS="-O"
  else
    case "${F77}" in
      kf77|mpikf77)
        FFLAGS="-fast +K3"
        ;;
      ifc)
        FFLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -openmp"
        fi
        ;;
      pgf77|mpipgf77)
        FFLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -mp"
        fi
        ;;
      f77|f90|mpxlf|mpif77|mpiifc|xlf|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            FFLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            FFLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -omp"
            fi
            ;;
          mips-sgi-irix6.[[4-9]]*)
            FFLAGS="-Ofast -64  -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            FFLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            FFLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            FFLAGS="-O3 -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -qsmp=omp"
            fi
            ;;
          sparc-sun-solaris2*)
            FFLAGS="-silent -O"
            ;;
          *)
            FFLAGS="-O"
            ;;
        esac
        ;;
      *)
        FFLAGS="-O"
        ;;
    esac
  fi
fi])
      
dnl **********************************************************************
dnl * ACX_DEBUG_FLAGS
dnl *
dnl try and determine what the debuging compile FLAGS
dnl
AC_DEFUN([ACX_DEBUG_FLAGS],
[AC_PREREQ(2.57)dnl
if test "x${casc_user_chose_cflags}" = "xno"
then
  if test "x${GCC}" = "xyes"
  then
    dnl **** default settings for gcc
    CFLAGS="-g"
    CFLAGS="$CFLAGS -Wall"
  else
    case "${CC}" in
      kcc|mpikcc)
        CFLAGS="-g +K3"
        ;;
      KCC|mpiKCC)
        CFLAGS="--c -g +K3"
        ;;
      icc)
        CFLAGS="-g -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -openmp"
        fi
        ;;
      pgcc|mpipgcc)
        CFLAGS="-g"
        if test "$casc_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -mp"
        fi
        ;;
      cc|c89|mpcc|mpiicc|xlc|ccc)
        case "${host}" in
          alpha*-dec-osf4.*)
            CFLAGS="-std1 -w0 -g"
            ;;
          alpha*-dec-osf5.*)
            CFLAGS="-g"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -omp"
            fi
            ;;
          hppa*-hp-hpux*)
            CFLAGS="-Aa -D_HPUX_SOURCE -g"
            ;;
          mips-sgi-irix6.[[4-9]]*)
            CFLAGS="-g -64 -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            CFLAGS="-fullwarn -woff 835 -g -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            CFLAGS="-D_ALL_SOURCE -g"
            ;;
          powerpc-ibm-aix*)
            CFLAGS="-g -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -qsmp=omp"
            fi
            ;;
          *)
            CFLAGS="-g"
            ;;
        esac
        ;;
      *)
        CFLAGS="-g"
        ;;
    esac
  fi
fi
if test "x${casc_user_chose_cxxflags}" = "xno"
then
  if test "x${GXX}" = "xyes"
  then
    dnl **** default settings for gcc
    CXXFLAGS="-g -Wall"
  else
    case "${CXX}" in
      KCC|mpiKCC)
        CXXFLAGS="-g +K3"
        ;;
      icc)
        CXXFLAGS="-g -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -openmp"
        fi
        ;;
      pgCC|mpipgCC)
        CXXFLAGS="-g"
        if test "$casc_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -mp"
        fi
        ;;
      CC|aCC|mpCC|mpiicc|xlC|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            CXXFLAGS="-std1 -w0 -g"
            ;;
          alpha*-dec-osf5.*)
            CXXFLAGS="-g"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -omp"
            fi
            ;;
          hppa*-hp-hpux*)
            CXXFLAGS="-D_HPUX_SOURCE -g"
            ;;
          mips-sgi-irix6.[[4-9]]*)
            CXXFLAGS="-g -64 -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            CXXFLAGS="-fullwarn -woff 835 -g -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            CXXFLAGS="-D_ALL_SOURCE -g"
            ;;
          powerpc-ibm-aix*)
            CXXFLAGS="-g -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -qsmp=omp"
            fi
            ;;
          *)
            CXXFLAGS="-g"
            ;;
        esac
        ;;
      *)
        CXXFLAGS="-g"
        ;;
    esac
  fi
fi
if test "x${casc_user_chose_fflags}" = "xno"
then
  if test "x${G77}" = "xyes"
  then
    FFLAGS="-g -Wall"
  else
    case "${F77}" in
      kf77|mpikf77)
        FFLAGS="-g +K3"
        ;;
      ifc)
        FFLAGS="-g -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -openmp"
        fi
        ;;
      pgf77|mpipgf77)
        FFLAGS="-g"
        if test "$casc_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -mp"
        fi
        ;;
      f77|f90|mpxlf|mpif77|mpiifc|xlf|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            FFLAGS="-std1 -w0 -g"
            ;;
          alpha*-dec-osf5.*)
            FFLAGS="-g"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -omp"
            fi
            ;;
          mips-sgi-irix6.[[4-9]]*)
            FFLAGS="-g -64 -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            FFLAGS="-fullwarn -woff 835 -g -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            FFLAGS="-D_ALL_SOURCE -g"
            ;;
          powerpc-ibm-aix*)
            FFLAGS="-g -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -qsmp=omp"
            fi
            ;;
          sparc-sun-solaris2*)
            FFLAGS="-silent -g"
            ;;
          *)
            FFLAGS="-g"
            ;;
        esac
        ;;
      *)
        FFLAGS="-g"
        ;;
    esac
  fi
fi]) dnl

dnl **********************************************************************
dnl * HYPRE_SET_ARCH
dnl * Defines the architecture of the platform on which the code is to run.
dnl * Cross-compiling is indicated by the host and build platforms being 
dnl * different values, which are usually user supplied on the command line.
dnl * When cross-compiling is detected the values supplied will be used
dnl * directly otherwise the needed values will be determined as follows:
dnl *
dnl * Find the hostname and assign it to an exported macro $HOSTNAME.
dnl * Guesses a one-word name for the current architecture, unless ARCH
dnl * has been preset.  This is an alternative to the built-in macro
dnl * AC_CANONICAL_HOST, which gives a three-word name.  Uses the utility
dnl * 'tarch', which is a Bourne shell script that should be in the same  
dnl * directory as the configure script.  If tarch is not present or if it
dnl * fails, ARCH is set to the value, if any, of shell variable HOSTTYPE,
dnl * otherwise ARCH is set to "unknown".
dnl **********************************************************************

AC_DEFUN([HYPRE_SET_ARCH],
[
   if test $host_alias = $build_alias
   then

      AC_MSG_CHECKING(the hostname)
      casc_hostname=hostname
      HOSTNAME="`$casc_hostname`"

      dnl * if $HOSTNAME is still empty, give it the value "unknown".
      if test -z "$HOSTNAME" 
      then
         HOSTNAME=unknown
         AC_MSG_WARN(hostname is unknown)
      else
         AC_MSG_RESULT($HOSTNAME)
      fi

      AC_MSG_CHECKING(the architecture)

      dnl * the environment variable $ARCH may already be set; if so use its
      dnl * value, otherwise go through this procedure
      if test -z "$ARCH"; then

         dnl * search for the tool "tarch".  It should be in the same 
         dnl * directory as configure.in, but a couple of other places will
         dnl * be checked.  casc_tarch stores a relative path for "tarch".
         casc_tarch_dir=
         for casc_dir in $srcdir $srcdir/.. $srcdir/../.. $srcdir/config; do
            if test -f $casc_dir/tarch; then
               casc_tarch_dir=$casc_dir
               casc_tarch=$casc_tarch_dir/tarch
               break
            fi
         done

         dnl * if tarch was not found or doesn't work, try using env variable
         dnl * $HOSTTYPE
         if test -z "$casc_tarch_dir"; then
            AC_MSG_WARN(cannot find tarch, using \$HOSTTYPE as the architecture)
            HYPRE_ARCH=$HOSTTYPE
         else
            HYPRE_ARCH="`$casc_tarch`"

            if test -z "$HYPRE_ARCH" || test "$HYPRE_ARCH" = "unknown"; then
               HYPRE_ARCH=$HOSTTYPE
            fi
         fi

         dnl * if $HYPRE_ARCH is still empty, give it the value "unknown".
         if test -z "$HYPRE_ARCH"; then
            HYPRE_ARCH=unknown
            AC_MSG_WARN(architecture is unknown)
         else
            AC_MSG_RESULT($HYPRE_ARCH)
         fi    
      else
         HYPRE_ARCH = $ARCH
         AC_MSG_RESULT($HYPRE_ARCH)
      fi

   else
      HYPRE_ARCH=$host_alias
      HOSTNAME=unknown
   fi
dnl *
dnl *    define type of architecture
   case $HYPRE_ARCH in
      alpha)
         AC_DEFINE(HYPRE_ALPHA)
         ;;
      sun* | solaris*)
         AC_DEFINE(HYPRE_SOLARIS)
         ;;
      hp* | HP*)
         AC_DEFINE(HYPRE_HPPA)
         ;;
      rs6000 | RS6000 | *bgl* | *BGL* | ppc64*)
         AC_DEFINE(HYPRE_RS6000)
         ;;
      IRIX64)
         AC_DEFINE(HYPRE_IRIX64)
         ;;
      Linux | linux | LINUX)
         case $HOSTNAME in
            mcr* | thunder* | ilx*)
               AC_DEFINE(HYPRE_LINUX_CHAOS2)
               ;;
            *)
               AC_DEFINE(HYPRE_LINUX)
               ;;
         esac
         ;;
   esac
     
dnl *
dnl *    return architecture and host name values
   AC_SUBST(HYPRE_ARCH)
   AC_SUBST(HOSTNAME)

])dnl

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

