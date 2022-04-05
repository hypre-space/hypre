dnl Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
dnl HYPRE Project Developers. See the top-level COPYRIGHT file for details.
dnl
dnl SPDX-License-Identifier: (Apache-2.0 OR MIT)

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
  hypre_lapack_link_ok=""
dnl **************************************************************
dnl Get fortran linker name for test function (dsygv in this case)
dnl **************************************************************
dnl  AC_FC_FUNC(dsygv)
  
  if test $hypre_fmangle_lapack = 1
  then
     LAPACKFUNC="dsygv"
  elif test $hypre_fmangle_lapack = 2
  then
     LAPACKFUNC="dsygv_"
  elif test $hypre_fmangle_lapack = 3
  then
     LAPACKFUNC="dsygv__"
  elif test $hypre_fmangle_lapack = 4
  then
     LAPACKFUNC="DSYGV"          
  else
     LAPACKFUNC="dsygv dsygv_ dsygv__ DSYGV"
     hypre_fmangle_lapack=0
  fi
  
dnl **************************************************************
dnl Get user provided path-to-lapack library
dnl **************************************************************
dnl  LDLAPACKLIBDIRS="$LAPACKLIBDIRS"
  USERLAPACKLIBS="$LAPACKLIBS"
  USERLAPACKLIBDIRS="$LAPACKLIBDIRS"  
  LAPACKLIBPATHS=""
  LAPACKLIBNAMES=""
  SUFFIXES=""

dnl Case where explicit path could be given by the user
  for lapack_lib in $LAPACKLIBS; do
    [lapack_lib_name=${lapack_lib##*-l}]
    if test $lapack_lib = $lapack_lib_name;
    then
dnl      if test -f $lapack_lib; 
dnl      then
dnl         [libsuffix=${lapack_lib##*.}]
dnl         SUFFIXES="$SUFFIXES $libsuffix"
dnl         [dir_path=${lapack_lib%/*}]
dnl         LAPACKLIBPATHS="-L$dir_path $LAPACKLIBPATHS"
dnl         [lapack_lib_name=${lapack_lib_name%%.*}]                  
dnl         [lapack_lib_name=${lapack_lib_name##*/}]  
dnl         [lapack_lib_name=${lapack_lib_name#*lib}]
dnl         LAPACKLIBNAMES="$LAPACKLIBNAMES $lapack_lib_name"
dnl      else
dnl         AC_MSG_ERROR([**************** Invalid path to lapack library error: ***************************
dnl         User set LAPACK library path using either --with-lapack-lib=<lib>, or 
dnl         --with-lapack-libs=<lapack_lib_base_name> and --with-lapack_dirs=<path-to-lapack-lib>, 
dnl         but the path " $lapack_lib " 
dnl         in the user-provided path for --with-lapack-libs does not exist. Please
dnl         check that the provided path is correct.
dnl         *****************************************************************************************],[9])         
dnl      fi

         [libsuffix=${lapack_lib##*.}]
         SUFFIXES="$SUFFIXES $libsuffix"         
         if test "$libsuffix" = "a" -o "$libsuffix" = "so" ;
         then
dnl            if test -f $lapack_lib;
dnl            then
               [dir_path=${lapack_lib#*/}]
               [dir_path=${lapack_lib%/*}]
               LAPACKLIBPATHS="$LAPACKLIBPATHS -L/$dir_path"
               [lapack_lib_name=${lapack_lib_name%.*}]
               [lapack_lib_name=${lapack_lib_name##*/}]
               [lapack_lib_name=${lapack_lib_name#*lib}]
               LAPACKLIBNAMES="$LAPACKLIBNAMES $lapack_lib_name"
         else
            LAPACKLIBPATHS="$dir_path $LAPACKLIBPATHS"
         fi
    else
      LAPACKLIBNAMES="$LAPACKLIBNAMES $lapack_lib_name"
    fi
  done

dnl **************************************************************
dnl Save current LIBS and LDFLAGS to be restored later 
dnl **************************************************************
    hypre_saved_LIBS="$LIBS"
    hypre_saved_LDFLAGS="$LDFLAGS"
    LIBS="$LIBS $FCLIBS"
    LDFLAGS="$LAPACKLIBDIRS $LDFLAGS"

dnl **************************************************************
dnl Check for dsygv in linkable list of libraries
dnl **************************************************************
    if test "x$LAPACKLIBNAMES" != "x"; then
       hypre_lapack_link_ok=no
    fi
    for lapack_lib in $LAPACKLIBNAMES; do
dnl **************************************************************
dnl Check if library works and print result
dnl **************************************************************      
        for func in $LAPACKFUNC; do                
           AC_CHECK_LIB($lapack_lib, $func, [hypre_lapack_link_ok=yes],[],[-lblas])
           if test "$hypre_lapack_link_ok" = "yes"; then
              break 2
           fi
        done
    done

    if test "$hypre_lapack_link_ok" = "no"; then
      AC_MSG_ERROR([**************** Non-linkable lapack library error: ***************************
      User set LAPACK library path using either --with-lapack-lib=<lib>, or 
      --with-lapack-libs=<lapack_lib_base_name> and --with-lapack_dirs=<path-to-lapack-lib>, 
      but $USERLAPACKLIBDIRS $USERLAPACKLIBS provided cannot be used. See "configure --help" for usage details.
      *****************************************************************************************],[9])
    fi

dnl **************************************************************
dnl set HYPRE_FMANGLE_LAPACK flag if not set
dnl **************************************************************
    if test "$hypre_lapack_link_ok" = "yes" -a "$hypre_fmangle_lapack" = "0"
    then
       if test "$func" = "dsygv"
       then
          hypre_fmangle_lapack=1
       elif test "$func" = "dsygv_"
       then
          hypre_fmangle_lapack=2
       elif test "$func" = "dsygv__"
       then
          hypre_fmangle_lapack=3
       else
          hypre_fmangle_lapack=4
       fi
       AC_DEFINE_UNQUOTED(HYPRE_FMANGLE_LAPACK, [$hypre_fmangle_lapack], [Define as in HYPRE_FMANGLE to set the LAPACK name mangling scheme])
    fi                    

dnl **************************************************************
dnl Restore LIBS and LDFLAGS
dnl **************************************************************
    LIBS="$hypre_saved_LIBS"
    LDFLAGS="$hypre_saved_LDFLAGS" 
dnl  fi
])
dnl Done with macro AC_HYPRE_CHECK_USER_LAPACKLIBS
