dnl #BHEADER**********************************************************************
dnl # Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
dnl # Produced at the Lawrence Livermore National Laboratory.
dnl # This file is part of HYPRE.  See file COPYRIGHT for details.
dnl #
dnl # HYPRE is free software; you can redistribute it and/or modify it under the
dnl # terms of the GNU Lesser General Public License (as published by the Free
dnl # Software Foundation) version 2.1 dated February 1999.
dnl #
dnl # $Revision: 1.25 $
dnl #EHEADER**********************************************************************

dnl **********************************************************************
dnl * AC_HYPRE_CHECK_MPI
dnl *
dnl try to determine what the MPI flags should be
dnl AC_HYPRE_CHECK_MPI([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl ACTION-IF-FOUND is a list of shell commands to run 
dnl   if an MPI library is found, and
dnl ACTION-IF-NOT-FOUND is a list of commands to run it
dnl   if it is not found. If ACTION-IF-FOUND is not specified,
dnl   the default action will define HAVE_MPI.
dnl **********************************************************************
AC_DEFUN([AC_HYPRE_CHECK_MPI],
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
  AC_DEFINE(HYPRE_HAVE_MPI,1,[Found the MPI library.])
  $1
  :
fi
])

dnl **********************************************************************
dnl * AC_HYPRE_FIND_G2C
dnl *  try to find libg2c.a
dnl **********************************************************************
AC_DEFUN([AC_HYPRE_FIND_G2C],
[
dnl AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

  hypre_save_LIBS="$LIBS"
  LIBS="$LIBS $FLIBS"

  found_g2c=no

dnl * This setting of LDFLAGS is not the right way to go (RDF)
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
dnl * AC_HYPRE_OPTIMIZATION_FLAGS
dnl *
dnl * Set compile FLAGS for optimization
dnl **********************************************************************
AC_DEFUN([AC_HYPRE_OPTIMIZATION_FLAGS],
[AC_PREREQ(2.57)dnl

if test "x${hypre_user_chose_cflags}" = "xno"
then
   case "${CC}" in
      gcc|mpicc)
        CFLAGS="-O2"
        ;;
      icc|mpiicc)
        CFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -openmp"
          LDFLAGS="$LDFLAGS -openmp"
        fi
        ;;
      pgcc|mpipgcc)
        CFLAGS="-fast"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -mp"
          LDFLAGS="$LDFLAGS -mp"
        fi
        ;;
      KCC|mpiKCC)
        CFLAGS="-fast +K3"
        ;;
      cc|mpcc|xlc|mpxlc|mpixlc_r)
        CFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -qsmp=omp"
          LDFLAGS="$LDFLAGS -qsmp=omp"
        fi
        ;;
      *)
        CFLAGS="-O"
        ;;
   esac
fi

if test "x${hypre_user_chose_cxxflags}" = "xno"
then
   case "${CXX}" in
      gCC|mpiCC)
        CXXFLAGS="-O2"
        ;;
      icpc|icc|mpiicpc|mpiicc)
        CXXFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -openmp"
        fi
        ;;
      pgCC|mpipgCC)
        CXXFLAGS="-fast"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -mp"
        fi
        ;;
      KCC|mpiKCC)
        CXXFLAGS="-fast +K3"
        ;;
      CC|mpCC|mpxlC|xlC|cxx)
        CXXFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -qsmp=omp"
        fi
        ;;
      *)
        CXXFLAGS="-O"
        ;;
   esac
fi

if test "x${hypre_user_chose_fflags}" = "xno"
then
   case "${F77}" in
      g77)
        FFLAGS="-O2"
        ;;
      ifort|mpiifort)
        FFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -openmp"
        fi
        ;;
      pgf77|mpipgf77)
        FFLAGS="-fast"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -mp"
        fi
        ;;
      kf77|mpikf77)
        FFLAGS="-fast +K3"
        ;;
      f77|f90|mpxlf|mpif77|xlf)
        FFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -qsmp=omp"
        fi
        ;;
      *)
        FFLAGS="-O"
        ;;
   esac
fi])

dnl **********************************************************************
dnl * AC_HYPRE_DEBUG_FLAGS
dnl *
dnl * Set compile FLAGS for debug
dnl **********************************************************************
AC_DEFUN([AC_HYPRE_DEBUG_FLAGS],
[AC_PREREQ(2.57)dnl

if test "x${hypre_user_chose_cflags}" = "xno"
then
   case "${CC}" in
      gcc|mpicc)
        CFLAGS="-g -Wall"
        ;;
      KCC|mpiKCC)
        CFLAGS="--c -g +K3"
        ;;
      icc|mpiicc)
        CFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -openmp"
          LDFLAGS="$LDFLAGS -openmp"
        fi
        ;;
      pgcc|mpipgcc)
        CFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -mp"
          LDFLAGS="$LDFLAGS -mp"
        fi
        ;;
      cc|mpcc|xlc|mpxlc|mpixlc_r)
        CFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -qsmp=omp"
          LDFLAGS="$LDFLAGS -qsmp=omp"
        fi
        ;;
      *)
        CFLAGS="-g"
        ;;
   esac
fi

if test "x${hypre_user_chose_cxxflags}" = "xno"
then
   case "${CXX}" in
      g++|mpig++)
        CXXFLAGS="-g -Wall"
        ;;
      KCC|mpiKCC)
        CXXFLAGS="-g +K3"
        ;;
      icpc|icc|mpiicpc|mpiicc)
        CXXFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -openmp"
        fi
        ;;
      pgCC|mpipgCC)
        CXXFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -mp"
        fi
        ;;
      CC|mpCC|mpxlC|xlC|cxx)
        CXXFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -qsmp=omp"
        fi
        ;;
      *)
        CXXFLAGS="-g"
        ;;
   esac
fi

if test "x${hypre_user_chose_fflags}" = "xno"
then
   case "${F77}" in
      g77|mpig77)
        FFLAGS="-g -Wall"
        ;;
      kf77|mpikf77)
        FFLAGS="-g +K3"
        ;;
      ifort|mpiifort)
        FFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -openmp"
        fi
        ;;
      pgf77|mpipgf77)
        FFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -mp"
        fi
        ;;
      f77|f90|mpxlf|mpif77|xlf)
        FFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -qsmp=omp"
        fi
        ;;
      *)
        FFLAGS="-g"
        ;;
   esac
fi]) dnl

dnl **********************************************************************
dnl * AC_HYPRE_SET_ARCH
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

AC_DEFUN([AC_HYPRE_SET_ARCH],
[
   if test $host_alias = $build_alias
   then

      AC_MSG_CHECKING(the hostname)
      hypre_hostname=hostname
      HOSTNAME="`$hypre_hostname`"

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
         dnl * be checked.  hypre_tarch stores a relative path for "tarch".
         hypre_tarch_dir=
         for hypre_dir in $srcdir $srcdir/.. $srcdir/../.. $srcdir/config; do
            if test -f $hypre_dir/tarch; then
               hypre_tarch_dir=$hypre_dir
               hypre_tarch=$hypre_tarch_dir/tarch
               break
            fi
         done

         dnl * if tarch was not found or doesn't work, try using env variable
         dnl * $HOSTTYPE
         if test -z "$hypre_tarch_dir"; then
            AC_MSG_WARN(cannot find tarch, using \$HOSTTYPE as the architecture)
            HYPRE_ARCH=$HOSTTYPE
         else
            HYPRE_ARCH="`$hypre_tarch`"

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
         if test -r /etc/home.config
         then
            systemtype=`grep ^SYS_TYPE /etc/home.config | cut -d" " -f2`
            case $systemtype in 
               chaos*)
                  AC_DEFINE(HYPRE_LINUX_CHAOS)
                  ;;
               *)
                  AC_DEFINE(HYPRE_LINUX)
                  ;;
            esac
         else
            AC_DEFINE(HYPRE_LINUX)
         fi
         ;;
   esac
     
dnl *
dnl *    return architecture and host name values
   AC_SUBST(HYPRE_ARCH)
   AC_SUBST(HOSTNAME)

])dnl
