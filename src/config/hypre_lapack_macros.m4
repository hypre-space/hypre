dnl #BHEADER**********************************************************************
dnl # Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
dnl # Produced at the Lawrence Livermore National Laboratory.
dnl # This file is part of HYPRE.  See file COPYRIGHT for details.
dnl #
dnl # HYPRE is free software; you can redistribute it and/or modify it under the
dnl # terms of the GNU Lesser General Public License (as published by the Free
dnl # Software Foundation) version 2.1 dated February 1999.
dnl #
dnl # $Revision$
dnl #EHEADER**********************************************************************




dnl @synopsis AC_HYPRE_FIND_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the LAPACK
dnl linear-algebra interface (see http://www.netlib.org/lapack/).
dnl On success, it sets the LAPACKLIBS output variable to
dnl hold the requisite library linkages.
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl     $LAPACKLIBS $BLASLIBS $LIBS $FCLIBS
dnl
dnl in that order.  BLASLIBS is either the output variable of the HYPRE_FIND_BLAS
dnl macro (which is called by configure before this macro) or the user-defined 
dnl blas library.  FCLIBS is the output variable of the AC_FC_LIBRARY_LDFLAGS 
dnl macro, which is sometimes necessary in order to link with Fortran libraries. 
dnl
dnl The user may use --with-lapack-libs=<lib> and --with-lapack-lib-dirs=<dir>
dnl in order to use a specific LAPACK library <lib>.  In order to link successfully,
dnl however, be aware that you will probably need to use the same Fortran compiler
dnl (which can be set via the FC env. var.) as was used to compile the LAPACK and
dnl BLAS libraries.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a LAPACK
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands
dnl to run it if it is not found.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>

AC_DEFUN([AC_HYPRE_FIND_LAPACK], 
[
  AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])

#***************************************************************
#   Initialize return variables
#***************************************************************
  LAPACKLIBS="null"
  LAPACKLIBDIRS="null"

  AC_ARG_WITH(lapack,
        [AS_HELP_STRING([--with-lapack], [Find a system-provided LAPACK library])])

  case $with_lapack in
      yes) ;;
        *) LAPACKLIBS="internal" ;;
  esac

#***************************************************************
#   Save incoming LIBS and LDFLAGS values to be restored
#***************************************************************
  hypre_save_LIBS="$LIBS"
  hypre_save_LDFLGS="$LDFLAGS"
  LIBS="$LIBS $FCLIBS"

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
  if test "$LAPACKLIBS" = "null"; then
     AC_FC_FUNC(dsygv)
     for lib in $LAPACK_LIB_NAMES; do
        AC_CHECK_LIB($lib, $dsygv, [LAPACKLIBS=$lib], [], [-lblas])
     done
  fi

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

])dnl AC_HYPRE_FIND_LAPACK

dnl @synopsis AC_HYPRE_CHECK_USER_LAPACKLIBS
dnl
dnl This macro checks that the user-provided blas library is 
dnl linkable. Configure fails with an error message if this 
dnl check fails.
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl     $LAPACKLIBS $BLASLIBS $LIBS $FCLIBS
dnl
dnl in that order.  FCLIBS is the output variable of the
dnl AC_FC_LIBRARY_LDFLAGS macro, and is sometimes necessary in order to link
dnl with Fortran libraries.
dnl
dnl The user may specify a BLAS library by using the --with-lapack-lib=<lib>, or 
dnl --with-lapack-libs=<lib> and --with-lapack-lib-dirs=<dir> options.  In order to link successfully,
dnl however, be aware that you will probably need to use the same Fortran
dnl compiler (which can be set via the FC env. var.) as was used to compile
dnl the BLAS library.
dnl
dnl @author Daniel Osei-Kuffuor  <oseikuffuor1@llnl.gov>

AC_DEFUN([AC_HYPRE_CHECK_USER_LAPACKLIBS],
[
  AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])
dnl **************************************************************
dnl Define some variables
dnl **************************************************************
  hypre_lapack_link_ok="no"
dnl **************************************************************
dnl Get fortran linker name for test function (dgemm in this case)
dnl **************************************************************
  AC_FC_FUNC(dsygv)
dnl **************************************************************
dnl Get user provided path-to-lapack library
dnl **************************************************************
dnl  LDLAPACKLIBDIRS="$LAPACKLIBDIRS"
  USERLAPACKLIBS="$LAPACKLIBS"
  USERLAPACKLIBDIRS="$LAPACKLIBDIRS"  
  LAPACKLIBPATHS=""
  LAPACKLIBNAMES=""
  SUFFIXES=""
  for lapack_dir in $LAPACKLIBDIRS; do
    [lapack_dir=${lapack_dir##*-L}]
    LAPACKLIBPATHS="$LAPACKLIBPATHS $lapack_dir"
  done

dnl Case where explicit path could be given by the user
  for lapack_lib in $LAPACKLIBS; do
    [lapack_lib_name=${lapack_lib##*-l}]
    if test $lapack_lib = $lapack_lib_name;
    then
      [libsuffix=${lapack_lib##*.}]
      SUFFIXES="$SUFFIXES $libsuffix"
      [dir_path=${lapack_lib%/*}]
      LAPACKLIBPATHS="$dir_path $LAPACKLIBPATHS"
      [lapack_lib_name=${lapack_lib_name%%.*}]                  
      [lapack_lib_name=${lapack_lib_name##*/}]  
      [lapack_lib_name=${lapack_lib_name#*lib}]
      LAPACKLIBNAMES="$LAPACKLIBNAMES $lapack_lib_name"
    else
      LAPACKLIBNAMES="$LAPACKLIBNAMES $lapack_lib_name"
    fi
  done
  echo SUFFIXES=$SUFFIXES
dnl  echo LAPACKLIBS=$LAPACKLIBNAMES
dnl  echo LAPACKLIBPATHS=$LAPACKLIBPATHS
dnl **************************************************************
dnl Begin test:
dnl **************************************************************
  LAPACKLIBS="null"
  LAPACKLIBDIRS="null"
dnl **************************************************************
dnl Test for viable library path:
dnl **************************************************************  
  for dir in $LAPACKLIBPATHS; do
     if test $LAPACKLIBDIRS = "null"; then
        for lapack_lib in $LAPACKLIBNAMES; do
           if test "x$SUFFIXES" = "x"; then
              if test $LAPACKLIBS = "null" -a -f $dir/lib$lapack_lib.a; then
                 LAPACKLIBDIRS="-L$dir"
                 LAPACKLIBS="-l$lapack_lib"
              fi
              if test $LAPACKLIBS = "null" -a -f $dir/lib$lapack_lib.so; then
                 LAPACKLIBDIRS="-L$dir"
                 LAPACKLIBS="-l$lapack_lib"
              fi
           else
              for libsuffix in $SUFFIXES; do
                 if test $LAPACKLIBS = "null" -a -f $dir/lib$lapack_lib.$libsuffix; then
                    LAPACKLIBDIRS="-L$dir"
                    LAPACKLIBS="-l$lapack_lib"
                 fi
              done
           fi              
        done
     fi
  done
        
  if test $LAPACKLIBDIRS = "null" -o $LAPACKLIBS = "null"; then
     AC_MSG_ERROR([**************** Incorrect path to lapack library error: ***************************
     User-specified lapack library path is incorrect or cannot be used. Please specify full path to 
     lapack library or check that the provided library path exists. If a library file is not explicitly
     specified, configure checks for library files with extensions ".a" or ".so" in 
     the user-specified path. Otherwise use --with-lapack option to find the library on the system. 
     See "configure --help" for usage details.
     *****************************************************************************************],[9])           
  fi 
dnl **************************************************************
dnl Save current LIBS and LDFLAGS to be restored later 
dnl **************************************************************
    hypre_saved_LIBS="$LIBS"
    hypre_saved_LDFLAGS="$LDFLAGS"
    LIBS="$LIBS $FCLIBS"
    LDFLAGS="$LAPACKLIBDIRS $LDFLAGS"

dnl **************************************************************
dnl Check for dgemm in linkable list of libraries
dnl **************************************************************
dnl **************************************************************
dnl Get library base name
dnl **************************************************************
        [lapack_lib=${LAPACKLIBS##*-l}]
dnl **************************************************************
dnl Check if library works and print result
dnl **************************************************************      
        AC_CHECK_LIB($lapack_lib, $dsygv, [hypre_lapack_link_ok=yes],[],[-lblas])
dnl      fi
dnl    done

    if test $hypre_lapack_link_ok = "no"; then
      AC_MSG_ERROR([**************** Non-linkable lapack library error: ***************************
      User set LAPACK library path using either --with-lapack-lib=<lib>, or 
      --with-lapack-libs=<lapack_lib_base_name> and --with-lapack_dirs=<path-to-lapack-lib>, 
      but $USERLAPACKLIBDIRS $USERLAPACKLIBS provided cannot be used. See "configure --help" for usage details.
      *****************************************************************************************],[9])
    fi
dnl **************************************************************
dnl Restore LIBS and LDFLAGS
dnl **************************************************************
    LIBS="$hypre_saved_LIBS"
    LDFLAGS="$hypre_saved_LDFLAGS" 
dnl  fi
])
dnl Done with macro AC_HYPRE_CHECK_USER_LAPACKLIBS
