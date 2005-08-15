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
dnl * HYPRE_GUESS_ARCH
dnl * First find the hostname and assigns it to an exported macro $HOSTNAME.
dnl * Guesses a one-word name for the current architecture, unless ARCH
dnl * has been preset.  This is an alternative to the built-in macro
dnl * AC_CANONICAL_HOST, which gives a three-word name.  Uses the utility
dnl * 'tarch', which is a Bourne shell script that should be in the same  
dnl * directory as the configure script.  If tarch is not present or if it
dnl * fails, ARCH is set to the value, if any, of shell variable HOSTTYPE,
dnl * otherwise ARCH is set to "unknown".
dnl **********************************************************************

AC_DEFUN([HYPRE_GUESS_ARCH],
[
   AC_MSG_CHECKING(the hostname)
   casc_hostname=hostname
   HOSTNAME="`$casc_hostname`"

   if test -z "$HOSTNAME" 
   then
   dnl * if $HOSTNAME is still empty, give it the value "unknown".
      HOSTNAME=unknown
      AC_MSG_WARN(hostname is unknown)
   else
      AC_MSG_RESULT($HOSTNAME)
   fi
   

   AC_MSG_CHECKING(the architecture)

   dnl * $ARCH could already be set in the environment or earlier in configure
   dnl * Use the preset value if it exists, otherwise go through the procedure
   if test -z "$ARCH"; then

      dnl * configure searches for the tool "tarch".  It should be in the
      dnl * same directory as configure.in, but a couple of other places
      dnl * will be checked.  casc_tarch stores a relative path for "tarch".
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

   AC_SUBST(HYPRE_ARCH)
   AC_SUBST(HOSTNAME)

])dnl
