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



# ---------------------------------------- #
# 4d. Fortran 77 compiler characteristics. #
# ---------------------------------------- #

#
# NOTE: LLNL_F77_LIBRARY_LDFLAGS, _LLNL_PROG_F77_V_OUTPUT, and _LLNL_PROG_F77_V
#       are identical to their AC_* cousins except that _LLNL_PROG_F77_V
#       calls AC_LINK_IFELSE instead of AC_COMPILE_IFELSE
#

# _LLNL_PROG_F77_V_OUTPUT([FLAG = $ac_cv_prog_f77_v])
# -------------------------------------------------
# Link a trivial Fortran program, compiling with a verbose output FLAG
# (which default value, $ac_cv_prog_f77_v, is computed by
# _AC_PROG_F77_V), and return the output in $ac_f77_v_output.  This
# output is processed in the way expected by AC_F77_LIBRARY_LDFLAGS,
# so that any link flags that are echoed by the compiler appear as
# space-separated items.
AC_DEFUN([_LLNL_PROG_F77_V_OUTPUT],
[AC_REQUIRE([AC_PROG_F77])dnl
AC_LANG_PUSH(Fortran 77)dnl

AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])

# Compile and link our simple test program by passing a flag (argument
# 1 to this macro) to the Fortran 77 compiler in order to get
# "verbose" output that we can then parse for the Fortran 77 linker
# flags.
ac_save_FFLAGS=$FFLAGS
FFLAGS="$FFLAGS m4_default([$1], [$ac_cv_prog_f77_v])"
(eval echo $as_me:__oline__: \"$ac_link\") >&AS_MESSAGE_LOG_FD
ac_f77_v_output=`eval $ac_link AS_MESSAGE_LOG_FD>&1 2>&1 | grep -v 'Driving:'`
echo "$ac_f77_v_output" >&AS_MESSAGE_LOG_FD
FFLAGS=$ac_save_FFLAGS

rm -f conftest*
AC_LANG_POP(Fortran 77)dnl

# If we are using xlf then replace all the commas with spaces.
if echo $ac_f77_v_output | grep xlfentry >/dev/null 2>&1; then
  ac_f77_v_output=`echo $ac_f77_v_output | sed 's/,/ /g'`
fi

# On HP/UX there is a line like: "LPATH is: /foo:/bar:/baz" where
# /foo, /bar, and /baz are search directories for the Fortran linker.
# Here, we change these into -L/foo -L/bar -L/baz (and put it first):
ac_f77_v_output="`echo $ac_f77_v_output |
	grep 'LPATH is:' |
	sed 's,.*LPATH is\(: *[[^ ]]*\).*,\1,;s,: */, -L/,g'` $ac_f77_v_output"

case $ac_f77_v_output in
  # If we are using xlf then replace all the commas with spaces.
  *xlfentry*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed 's/,/ /g'` ;;

  # With Intel ifc, ignore the quoted -mGLOB_options_string stuff (quoted
  # $LIBS confuse us, and the libraries appear later in the output anyway).
  *mGLOB_options_string*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed 's/\"-mGLOB[[^\"]]*\"/ /g'` ;;

  # Portland Group compiler has singly- or doubly-quoted -cmdline argument
  # Singly-quoted arguments were reported for versions 5.2-4 and 6.0-4.
  # Doubly-quoted arguments were reported for "PGF90/x86 Linux/x86 5.0-2".
  *-cmdline\ \'*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed "s/-cmdline  *'[[^']]*'/ /g"` ;;

  *-cmdline*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed 's/-cmdline  *"[[^"]]*"/ /g'` ;;

  # If we are using Cray Fortran then delete quotes.
  # Use "\"" instead of '"' for font-lock-mode.
  # FIXME: a more general fix for quoted arguments with spaces?
  *cft90*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed "s/\"//g"` ;;
esac
])# _LLNL_PROG_F77_V_OUTPUT


# _LLNL_PROG_F77_V
# --------------
#
# Determine the flag that causes the Fortran 77 compiler to print
# information of library and object files (normally -v)
# Needed for AC_F77_LIBRARY_FLAGS
# Some compilers don't accept -v (Lahey: -verbose, xlf: -V, Fujitsu: -###)
AC_DEFUN([_LLNL_PROG_F77_V],
[AC_CACHE_CHECK([how to get verbose linking output from $F77],
                [ac_cv_prog_f77_v],
[AC_LANG_ASSERT(Fortran 77)
AC_LINK_IFELSE([AC_LANG_PROGRAM()],
[ac_cv_prog_f77_v=
# Try some options frequently used verbose output
# It is better to try -V before -v for xlf
for ac_verb in -V -v -verbose --verbose -\#\#\#; do
  _LLNL_PROG_F77_V_OUTPUT($ac_verb)
  # look for -l* and *.a constructs in the output
  for ac_arg in $ac_f77_v_output; do
     case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]*)
          ac_cv_prog_f77_v=$ac_verb
          break 2 ;;
     esac
  done
done
if test -z "$ac_cv_prog_f77_v"; then
   AC_MSG_WARN([cannot determine how to obtain linking information from $F77])
fi],
                  [AC_MSG_WARN([compilation failed])])
])])# _LLNL_PROG_F77_V


# LLNL_F77_LIBRARY_LDFLAGS
# ----------------------
#
# Determine the linker flags (e.g. "-L" and "-l") for the Fortran 77
# intrinsic and run-time libraries that are required to successfully
# link a Fortran 77 program or shared library.  The output variable
# FLIBS is set to these flags.
#
# This macro is intended to be used in those situations when it is
# necessary to mix, e.g. C++ and Fortran 77, source code into a single
# program or shared library.
#
# For example, if object files from a C++ and Fortran 77 compiler must
# be linked together, then the C++ compiler/linker must be used for
# linking (since special C++-ish things need to happen at link time
# like calling global constructors, instantiating templates, enabling
# exception support, etc.).
#
# However, the Fortran 77 intrinsic and run-time libraries must be
# linked in as well, but the C++ compiler/linker doesn't know how to
# add these Fortran 77 libraries.  Hence, the macro
# "AC_F77_LIBRARY_LDFLAGS" was created to determine these Fortran 77
# libraries.
#
# This macro was packaged in its current form by Matthew D. Langston.
# However, nearly all of this macro came from the "OCTAVE_FLIBS" macro
# in "octave-2.0.13/aclocal.m4", and full credit should go to John
# W. Eaton for writing this extremely useful macro.  Thank you John.
AC_DEFUN([LLNL_F77_LIBRARY_LDFLAGS],
[AC_LANG_PUSH(Fortran 77)dnl
_LLNL_PROG_F77_V
AC_CACHE_CHECK([for Fortran 77 libraries], ac_cv_flibs,
[if test "x$FLIBS" != "x"; then
  ac_cv_flibs="$FLIBS" # Let the user override the test.
else

_LLNL_PROG_F77_V_OUTPUT

ac_cv_flibs=

# Save positional arguments (if any)
ac_save_positional="$[@]"

set X $ac_f77_v_output
while test $[@%:@] != 1; do
  shift
  ac_arg=$[1]
  case $ac_arg in
	*libgcc.a | *libgcc_s.a)
	  ;;
        [[\\/]]*.a | ?:[[\\/]]*.a)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_flibs, ,
              ac_cv_flibs="$ac_cv_flibs $ac_arg")
          ;;
        -bI:*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_flibs, ,
             [_AC_LINKER_OPTION([$ac_arg], ac_cv_flibs)])
          ;;
          # Ignore these flags.
        -lang* | -lcrt[[012]].o | -lcrtbegin.o | -lc | -lgcc* | -libmil | -LANG:=*)
          ;;
	-lfrtbegin )  ;; #(gkk) Ignore this one too
        -lkernel32)
          test x"$CYGWIN" != xyes && ac_cv_flibs="$ac_cv_flibs $ac_arg"
          ;;
        -[[LRuY]])
          # These flags, when seen by themselves, take an argument.
          # We remove the space between option and argument and re-iterate
          # unless we find an empty arg or a new option (starting with -)
	  case $[2] in
             "" | -*);;
             *)
		ac_arg="$ac_arg$[2]"
		shift; shift
		set X $ac_arg "$[@]"
		;;
	  esac
          ;;
        -YP,*)
          for ac_j in `echo $ac_arg | sed -e 's/-YP,/-L/;s/:/ -L/g'`; do
            _AC_LIST_MEMBER_IF($ac_j, $ac_cv_flibs, ,
                               [ac_arg="$ac_arg $ac_j"
                               ac_cv_flibs="$ac_cv_flibs $ac_j"])
          done
          ;;
        -[[lLR]]*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_flibs, ,
                             ac_cv_flibs="$ac_cv_flibs $ac_arg")
          ;;
          # Ignore everything else.
  esac
done
# restore positional arguments
set X $ac_save_positional; shift

# We only consider "LD_RUN_PATH" on Solaris systems.  If this is seen,
# then we insist that the "run path" must be an absolute path (i.e. it
# must begin with a "/").
case `(uname -sr) 2>/dev/null` in
   "SunOS 5"*)
      ac_ld_run_path=`echo $ac_f77_v_output |
                        sed -n 's,^.*LD_RUN_PATH *= *\(/[[^ ]]*\).*$,-R\1,p'`
      test "x$ac_ld_run_path" != x &&
        _AC_LINKER_OPTION([$ac_ld_run_path], ac_cv_flibs)
      ;;
   "Darwin 7"*)
      if test -n "$ac_cv_flibs"; then
	for ac_arg in $ac_cv_flibs; do
	  case $ac_arg in
	  -L*)
	    tmp_path="$tmp_path "`echo $ECHO_N $ac_arg | sed -e 's/^-L//'`
	    ;;
	  -lSystem) ;; # ignore this one
	  -lm)
	    modified_flibs="$modified_flibs $ac_arg"
	    ;;
	  -l*)
	    found="no"
	    if test -n "$tmp_path"; then
	      libname=`echo $ECHO_N $ac_arg | sed -e 's/^-l//'`
	      for tp in $tmp_path; do
		if test $found = "no"; then
		  if test -d $tp -a -r $tp; then
		    shortpath=`cd $tp 2>/dev/null && pwd`
		  else
		    shortpath=$tp
		  fi
		  if test -r "$shortpath/lib$libname.a" ; then
		    modified_flibs="$modified_flibs $shortpath/lib$libname.a"
		    found="yes"
		  elif test -r "$shortpath/lib$libname.so" ; then
		    modified_flibs="$modified_flibs $shortpath/lib$libname.so"
		    found="yes"
		  elif test -r "$shortpath/lib$libname.dylib" ; then
		    modified_flibs="$modified_flibs $shortpath/lib$libname.dylib"
		    found="yes"
		  fi
		fi
	      done
	    fi
	    if test $found = "no"; then
	      modified_flibs="$modified_flibs $ac_arg"
	    fi
	    ;;
	  esac
	done
	ac_cv_flibs="$modified_flibs"
      fi
      ;;
esac
fi # test "x$FLIBS" = "x"
])
FLIBS="$ac_cv_flibs"
AC_SUBST(FLIBS)
AC_LANG_POP(Fortran 77)dnl
])# LLNL_F77_LIBRARY_LDFLAGS


dnl #BHEADER**********************************************************************
dnl # Copyright (c) 2006   The Regents of the University of California.
dnl # Produced at the Lawrence Livermore National Laboratory.
dnl # Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
dnl # All rights reserved.
dnl #
dnl # This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
dnl # Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
dnl # disclaimer and the GNU Lesser General Public License.
dnl #
dnl # This program is free software; you can redistribute it and/or modify it
dnl # under the terms of the GNU General Public License (as published by the Free
dnl # Software Foundation) version 2.1 dated February 1999.
dnl #
dnl # This program is distributed in the hope that it will be useful, but WITHOUT
dnl # ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
dnl # FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
dnl # GNU General Public License for more details.
dnl #
dnl # You should have received a copy of the GNU Lesser General Public License
dnl # along with this program; if not, write to the Free Software Foundation,
dnl # Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
dnl #
dnl # $Revision$
dnl #EHEADER**********************************************************************

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
dnl **********************************************************************
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
dnl * HYPRE_SET_LINK_SUBDIRS
dnl *  sets appropriate sub-directory for linking based on using debug, 
dnl *  no-mpi or openmp when testing public alpha, beta or general releases
dnl **********************************************************************
AC_DEFUN([HYPRE_SET_LINK_SUBDIRS],
[
 if test "$casc_using_debug" = "yes" && "$casc_using_mpi" = "yes"
 then
     HYPRE_LINKDIR="${HYPRE_LINKDIR}/debug"
 fi

 if test "$casc_using_mpi" = "no"
 then
     HYPRE_LINKDIR="${HYPRE_LINKDIR}/serial"
     if test "$casc_using_debug" = "yes"
     then
        HYPRE_LINKDIR="${HYPRE_LINKDIR}/debug"
     fi
 fi

 if test "$casc_using_openmp" = "yes"
 then
     HYPRE_LINKDIR="${HYPRE_LINKDIR}/threads"
     if test "$casc_using_debug" = "yes"
     then
        HYPRE_LINKDIR="${HYPRE_LINKDIR}/debug"
     fi
 fi
])


dnl **********************************************************************
dnl * HYPRE_FIND_G2C
dnl *  try to find libg2c.a
dnl **********************************************************************
AC_DEFUN([HYPRE_FIND_G2C],
[
dnl AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])

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
dnl * Set compile FLAGS for optimization
dnl **********************************************************************
AC_DEFUN([ACX_OPTIMIZATION_FLAGS],
[AC_PREREQ(2.57)dnl

if test "x${casc_user_chose_cflags}" = "xno"
then
   case "${CC}" in
      gcc|mpicc)
        CFLAGS="-O2"
        ;;
      icc)
        CFLAGS="-O3 -tpp7"
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
      KCC|mpiKCC)
        CFLAGS="-fast +K3"
        ;;
      cc|mpcc|mpiicc|xlc)
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

if test "x${casc_user_chose_cxxflags}" = "xno"
then
   case "${CXX}" in
      gCC|mpiCC)
        CXXFLAGS="-O2"
        ;;
      icc)
        CXXFLAGS="-O3 -tpp7"
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
      KCC|mpiKCC)
        CXXFLAGS="-fast +K3"
        ;;
      CC|mpCC|mpiicc|xlC|cxx)
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

if test "x${casc_user_chose_fflags}" = "xno"
then
   case "${F77}" in
      g77)
        FFLAGS="-O"
        ;;
      ifort)
        FFLAGS="-O3 -tpp7"
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
      kf77|mpikf77)
        FFLAGS="-fast +K3"
        ;;
      f77|f90|mpxlf|mpif77|mpiifort|xlf)
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
fi])

dnl **********************************************************************
dnl * ACX_DEBUG_FLAGS
dnl *
dnl * Set compile FLAGS for debug
dnl **********************************************************************
AC_DEFUN([ACX_DEBUG_FLAGS],
[AC_PREREQ(2.57)dnl

if test "x${casc_user_chose_cflags}" = "xno"
then
   case "${CC}" in
      gcc|mpicc)
        CFLAGS="-g -Wall"
        ;;
      KCC|mpiKCC)
        CFLAGS="--c -g +K3"
        ;;
      icc)
        CFLAGS="-g -tpp7"
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
      cc|mpcc|mpiicc|xlc)
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

if test "x${casc_user_chose_cxxflags}" = "xno"
then
   case "${CXX}" in
      g++|mpig++)
        CXXFLAGS="-g -Wall"
        ;;
      KCC|mpiKCC)
        CXXFLAGS="-g +K3"
        ;;
      icc)
        CXXFLAGS="-g -tpp7"
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
      CC|mpCC|mpiicc|xlC|cxx)
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

if test "x${casc_user_chose_fflags}" = "xno"
then
   case "${F77}" in
      g77|mpig77)
        FFLAGS="-g -Wall"
        ;;
      kf77|mpikf77)
        FFLAGS="-g +K3"
        ;;
      ifort)
        FFLAGS="-g -tpp7"
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
      f77|f90|mpxlf|mpif77|mpiifort|xlf)
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

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



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

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/



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

