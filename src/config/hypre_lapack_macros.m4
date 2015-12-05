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
 * $Revision: 1.5 $
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
dnl @version $Id: hypre_lapack_macros.m4,v 1.5 2006/09/22 22:05:59 hill66 Exp $
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
