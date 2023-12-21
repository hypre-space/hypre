dnl Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
dnl HYPRE Project Developers. See the top-level COPYRIGHT file for details.
dnl
dnl SPDX-License-Identifier: (Apache-2.0 OR MIT)

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
[AC_PREREQ([2.69])dnl
AC_PREREQ([2.69]) dnl for AC_LANG_CASE

if test x = x"$MPILIBS"; then
  AC_LANG_CASE([C], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
    [C++], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
    [Fortran 77], [AC_MSG_CHECKING([for MPI_Init])
      AC_LINK_IFELSE([AC_LANG_PROGRAM([[]], [[      call MPI_Init]])],[MPILIBS=" "
        AC_MSG_RESULT(yes)],[AC_MSG_RESULT(no)])])
fi

if test x = x"$MPILIBS"; then
  AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
fi

if test x = x"$MPILIBS"; then
  AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
fi

dnl We have to use AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[]])],[],[]) and not
dnl AC_CHECK_HEADER because the latter uses $CPP, not $CC (which may be mpicc).
AC_LANG_CASE([C], [if test x != x"$MPILIBS"; then
  AC_MSG_CHECKING([for mpi.h])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <mpi.h>]], [[]])],[AC_MSG_RESULT(yes)],[MPILIBS=""
                     AC_MSG_RESULT(no)])
fi],
[C++], [if test x != x"$MPILIBS"; then
  AC_MSG_CHECKING([for mpi.h])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <mpi.h>]], [[]])],[AC_MSG_RESULT(yes)],[MPILIBS=""
                     AC_MSG_RESULT(no)])
fi])

AC_SUBST(MPILIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x = x"$MPILIBS"; then
  $2
  :
else
  AC_DEFINE(HYPRE_HAVE_MPI,1,[Define to 1 if an MPI library is found])
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
dnl AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])

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
[AC_PREREQ([2.69])dnl

if test "x${hypre_user_chose_cflags}" = "xno"
then
   case `basename ${CC}` in
      gcc|mpigcc|mpicc)
        CFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -fopenmp"
          LDFLAGS+=" -fopenmp"
        fi
        ;;
      icc|mpiicc|icx|mpiicx)
        CFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -qopenmp"
          LDFLAGS+=" -qopenmp"
        fi
        ;;
      pgcc|mpipgcc|mpipgicc)
        CFLAGS="-fast"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -mp"
          LDFLAGS+=" -mp"
        fi
        ;;
      cc|xlc|xlc_r|mpxlc|mpixlc|mpixlc_r|mpixlc-gpu|mpcc)
        CFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -qsmp=omp"
          LDFLAGS+=" -qsmp=omp"
        fi
        ;;
      KCC|mpiKCC)
        CFLAGS="-fast +K3"
        ;;
      *)
        CFLAGS="-O"
        ;;
   esac
fi

if test "x${hypre_user_chose_cxxflags}" = "xno"
then
   case `basename ${CXX}` in
      g++|gCC|mpig++|mpicxx|mpic++|mpiCC)
        CXXFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -fopenmp"
        fi
        ;;
      icpc|icc|mpiicpc|mpiicc|icpx|mpiicpx)
        CXXFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -qopenmp"
        fi
        ;;
      pgCC|mpipgCC|pgc++|mpipgic++)
        CXXFLAGS="-fast"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -mp"
        fi
        ;;
      CC|cxx|xlC|xlC_r|mpxlC|mpixlC|mpixlC-gpu|mpixlcxx|mpixlcxx_r|mpCC)
        CXXFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -qsmp=omp"
        fi
        ;;
      KCC|mpiKCC)
        CXXFLAGS="-fast +K3"
        ;;
      *)
        CXXFLAGS="-O"
        ;;
   esac
fi

if test "$hypre_using_fortran" = "yes" -a "x${hypre_user_chose_fflags}" = "xno"
then
   case `basename ${FC}` in
      g77|gfortran|mpigfortran|mpif77)
        FFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS+=" -fopenmp"
        fi
        ;;
      ifort|mpiifort)
        FFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS+=" -qopenmp"
        fi
        ;;
      pgf77|mpipgf77|pgfortran|mpipgifort)
        FFLAGS="-fast"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS+=" -mp"
        fi
        ;;
      f77|f90|xlf|xlf_r|mpxlf|mpixlf77|mpixlf77_r)
        FFLAGS="-O2"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS+=" -qsmp=omp"
        fi
        ;;
      kf77|mpikf77)
        FFLAGS="-fast +K3"
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
[AC_PREREQ([2.69])dnl

if test "x${hypre_user_chose_cflags}" = "xno"
then
   case `basename ${CC}` in
      gcc|mpigcc|mpicc)
        CFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -fopenmp"
          LDFLAGS+=" -fopenmp"
        fi
        ;;
      icc|mpiicc|icx|mpiicx)
        CFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -qopenmp"
          LDFLAGS+=" -qopenmp"
        fi
        ;;
      pgcc|mpipgcc|mpipgicc)
        CFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -mp"
          LDFLAGS+=" -mp"
        fi
        ;;
      cc|xlc|mpxlc|mpixlc|mpcc)
        CFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CFLAGS+=" -qsmp=omp"
          LDFLAGS+=" -qsmp=omp"
        fi
        ;;
      KCC|mpiKCC)
        CFLAGS="--c -g +K3"
        ;;
      *)
        CFLAGS="-g"
        ;;
   esac
fi

if test "x${hypre_user_chose_cxxflags}" = "xno"
then
   case `basename ${CXX}` in
      g++|gCC|mpig++|mpicxx|mpic++|mpiCC)
        CXXFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -fopenmp"
        fi
        ;;
      icpc|icc|mpiicpc|mpiicc|icpx|mpiicpx)
        CXXFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -qopenmp"
        fi
        ;;
      pgCC|mpipgCC|pgc++|mpipgic++)
        CXXFLAGS="-O0 -g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -mp"
        fi
        ;;
      CC|cxx|xlC|mpxlC|mpixlcxx|mpCC)
        CXXFLAGS="-O0 -g"
        if test "$hypre_using_openmp" = "yes" ; then
          CXXFLAGS+=" -qsmp=omp"
        fi
        ;;
      KCC|mpiKCC)
        CXXFLAGS="-g +K3"
        ;;
      *)
        CXXFLAGS="-g"
        ;;
   esac
fi

if test "$hypre_using_fortran" = "yes" -a "x${hypre_user_chose_fflags}" = "xno"
then
   case `basename ${FC}` in
      g77|gfortran|mpigfortran|mpif77)
        FFLAGS="-g -Wall"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -fopenmp"
        fi
        ;;
      ifort|mpiifort|ifx|mpiifx)
        FFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -qopenmp"
        fi
        ;;
      pgf77|mpipgf77|pgfortran|mpipgifort)
        FFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -mp"
        fi
        ;;
      f77|f90|xlf|mpxlf|mpixlf77)
        FFLAGS="-g"
        if test "$hypre_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -qsmp=omp"
        fi
        ;;
      kf77|mpikf77)
        FFLAGS="-g +K3"
        ;;
      *)
        FFLAGS="-g"
        ;;
   esac
fi])

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
         HYPRE_ARCH=$ARCH
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
         AC_DEFINE(HYPRE_ALPHA,1,[Define to 1 for Alpha platforms])
         ;;
      sun* | solaris*)
         AC_DEFINE(HYPRE_SOLARIS,1,[Define to 1 for Solaris.])
         ;;
      hp* | HP*)
         AC_DEFINE(HYPRE_HPPA,1,[Define to 1 for HP platforms])
         ;;
      rs6000 | RS6000 | *bgl* | *BGL* | ppc64*)
         AC_DEFINE(HYPRE_RS6000,1,[Define to 1 for RS6000 platforms])
         ;;
      IRIX64)
         AC_DEFINE(HYPRE_IRIX64,1,[Define to 1 for IRIX64 platforms])
         ;;
      Linux | linux | LINUX)
         if test -r /etc/home.config
         then
            systemtype=`grep ^SYS_TYPE /etc/home.config | cut -d" " -f2`
            case $systemtype in
               chaos*)
                  AC_DEFINE(HYPRE_LINUX_CHAOS,1,[Define to 1 for Linux on platforms running any version of CHAOS])
                  ;;
               *)
                  AC_DEFINE(HYPRE_LINUX,1,[Define to 1 for Linux platform])
                  ;;
            esac
         else
            AC_DEFINE(HYPRE_LINUX,1,[Define to 1 for Linux platform])
         fi
         ;;
   esac

dnl *
dnl *    return architecture and host name values
   AC_SUBST(HYPRE_ARCH)
   AC_SUBST(HOSTNAME)

])dnl
