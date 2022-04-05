dnl Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
dnl HYPRE Project Developers. See the top-level COPYRIGHT file for details.
dnl
dnl SPDX-License-Identifier: (Apache-2.0 OR MIT)

dnl @synopsis AC_HYPRE_FIND_BLAS([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the BLAS
dnl linear-algebra interface (see http://www.netlib.org/blas/).
dnl On success, it sets the BLASLIBS output variable to
dnl hold the requisite library linkages.
dnl
dnl To link with BLAS, you should link with:
dnl
dnl 	$BLASLIBS $LIBS $FCLIBS
dnl
dnl in that order.  FCLIBS is the output variable of the
dnl AC_FC_LIBRARY_LDFLAGS macro, and is sometimes necessary in order to link
dnl with fortran libraries.
dnl
dnl Many libraries are searched for, from ATLAS to CXML to ESSL.
dnl The user may specify a BLAS library by using the --with-blas-libs=<lib>
dnl and --with-blas-lib-dirs=<dir> options.  In order to link successfully,
dnl however, be aware that you will probably need to use the same Fortran
dnl compiler (which can be set via the FC env. var.) as was used to compile
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

AC_DEFUN([AC_HYPRE_FIND_BLAS],
[
  AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])

#***************************************************************
#   Initialize return variables
#***************************************************************
  BLASLIBS="null"
  BLASLIBDIRS="null"

  AC_ARG_WITH(blas,
	[AS_HELP_STRING([--with-blas], [Find a system-provided BLAS library])])

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
  if test "$BLASLIBS" = "null"; then
     AC_FC_FUNC(dgemm)
     for lib in $BLAS_LIB_NAMES; do
        AC_CHECK_LIB($lib, $dgemm, [BLASLIBS=$lib])
    done
  fi

#***************************************************************
#   Set path to selected BLAS library 
#***************************************************************
  BLAS_SEARCH_DIRS="/usr/lib /usr/local/lib /lib"

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
#   Set variables if ATLAS or DXML libraries are used 
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

])dnl AC_HYPRE_FIND_BLAS

dnl @synopsis AC_HYPRE_CHECK_USER_BLASLIBS
dnl
dnl This macro checks that the user-provided blas library is 
dnl linkable. Configure fails with an error message if this 
dnl check fails.
dnl
dnl To link with BLAS, you should link with:
dnl
dnl 	$BLASLIBS $LIBS $FCLIBS
dnl
dnl in that order.  FCLIBS is the output variable of the
dnl AC_FC_LIBRARY_LDFLAGS macro, and is sometimes necessary in order to link
dnl with Fortran libraries.
dnl
dnl The user may specify a BLAS library by using the --with-blas-lib=<lib>, or 
dnl --with-blas-libs=<lib> and --with-blas-lib-dirs=<dir> options.  In order to link successfully,
dnl however, be aware that you will probably need to use the same Fortran
dnl compiler (which can be set via the FC env. var.) as was used to compile
dnl the BLAS library.
dnl
dnl @author Daniel Osei-Kuffuor  <oseikuffuor1@llnl.gov>

AC_DEFUN([AC_HYPRE_CHECK_USER_BLASLIBS],
[
  AC_REQUIRE([AC_FC_LIBRARY_LDFLAGS])
dnl **************************************************************
dnl Define some variables
dnl **************************************************************
  hypre_blas_link_ok=""
dnl **************************************************************
dnl Get fortran linker name for test function (dgemm in this case)
dnl **************************************************************
dnl  AC_FC_FUNC(dgemm)

  if test $hypre_fmangle_blas = 1
  then
     BLASFUNC="dgemm"
  elif test $hypre_fmangle_blas = 2
  then
     BLASFUNC="dgemm_"
  elif test $hypre_fmangle_blas = 3
  then
     BLASFUNC="dgemm__"
  elif test $hypre_fmangle_blas = 4
  then
     BLASFUNC="DGEMM"          
  else
     BLASFUNC="dgemm dgemm_ dgemm__ DGEMM"
     hypre_fmangle_blas=0
  fi
  
dnl **************************************************************
dnl Get user provided path-to-blas library
dnl **************************************************************
dnl  LDBLASLIBDIRS="$BLASLIBDIRS"
  USERBLASLIBS="$BLASLIBS"
  USERBLASLIBDIRS="$BLASLIBDIRS"  
  BLASLIBPATHS="$BLASLIBDIRS"
  BLASLIBNAMES=""
  SUFFIXES=""

dnl Case where explicit path could be given by the user
  for blas_lib in $BLASLIBS; do
    [blas_lib_name=${blas_lib##*-l}]
    if test $blas_lib = $blas_lib_name;
    then
dnl      if test -f $blas_lib; 
dnl      then
dnl         [libsuffix=${blas_lib##*.}]
dnl         SUFFIXES="$SUFFIXES $libsuffix"
dnl         [dir_path=${blas_lib%/*}]
dnl         BLASLIBPATHS="-L$dir_path $BLASLIBPATHS"
dnl         [blas_lib_name=${blas_lib_name%%.*}]                  
dnl         [blas_lib_name=${blas_lib_name##*/}]  
dnl         [blas_lib_name=${blas_lib_name#*lib}]
dnl         BLASLIBNAMES="$BLASLIBNAMES $blas_lib_name"
dnl      else
dnl         AC_MSG_ERROR([**************** Invalid path to blas library error: ***************************
dnl         User set BLAS library path using either --with-blas-lib=<lib>, or 
dnl        --with-blas-libs=<blas_lib_base_name> and --with-blas_dirs=<path-to-blas-lib>, 
dnl         but the path " $blas_lib " 
dnl         in the user-provided path for --with-blas-libs does not exist. Please
dnl         check that the provided path is correct.
dnl         *****************************************************************************************],[9])         
dnl      fi

         [libsuffix=${blas_lib##*.}]
         SUFFIXES="$SUFFIXES $libsuffix"         
         if test "$libsuffix" = "a" -o "$libsuffix" = "so" ;
         then
dnl            if test -f $blas_lib;
dnl            then
               [dir_path=${blas_lib#*/}]
               [dir_path=${blas_lib%/*}]
               BLASLIBPATHS="$BLASLIBPATHS -L/$dir_path"
               [blas_lib_name=${blas_lib_name%.*}]
               [blas_lib_name=${blas_lib_name##*/}]
               [blas_lib_name=${blas_lib_name#*lib}]
               BLASLIBNAMES="$BLASLIBNAMES $blas_lib_name"
dnl            else
dnl               AC_MSG_ERROR([**************** Invalid path to blas library error: ***************************
dnl               User set BLAS library path using either --with-blas-lib=<lib>, or 
dnl              --with-blas-libs=<blas_lib_base_name> and --with-blas_dirs=<path-to-blas-lib>, 
dnl               but the path " $blas_lib " 
dnl               in the user-provided path for --with-blas-libs does not exist. Please
dnl               check that the provided path is correct.
dnl               *****************************************************************************************],[9])              
dnl            fi
         else
            BLASLIBPATHS="$dir_path $BLASLIBPATHS"
         fi
    else
      BLASLIBNAMES="$BLASLIBNAMES $blas_lib_name"
    fi
  done
    
dnl **************************************************************
dnl Save current LIBS and LDFLAGS to be restored later 
dnl **************************************************************
    hypre_saved_LIBS="$LIBS"
    hypre_saved_LDFLAGS="$LDFLAGS"
    LIBS="$LIBS $FCLIBS"
    LDFLAGS="$BLASLIBPATHS $LDFLAGS"

dnl    echo LDFLAGS=$LDFLAGS
dnl    echo LIBS=$LIBS
dnl    echo BLASLIBPATHS=$BLASLIBPATHS
dnl **************************************************************
dnl Check for dgemm in linkable list of libraries
dnl **************************************************************
    if test "x$BLASLIBNAMES" != "x"; then
       hypre_blas_link_ok=no
    fi
    for blas_lib in $BLASLIBNAMES; do
dnl **************************************************************
dnl Check if library works and print result
dnl **************************************************************   
        for func in $BLASFUNC; do                
           AC_CHECK_LIB($blas_lib, $func, [hypre_blas_link_ok=yes])
           if test "$hypre_blas_link_ok" = "yes"; then
              break 2
           fi
       done
    done

    if test "$hypre_blas_link_ok" = "no"; then
      AC_MSG_ERROR([**************** Non-linkable blas library error: ***************************
      User set BLAS library path using either --with-blas-lib=<lib>, or 
      --with-blas-libs=<blas_lib_base_name> and --with-blas_dirs=<path-to-blas-lib>, 
      but $USERBLASLIBDIRS $USERBLASLIBS provided cannot be used. See "configure --help" for usage details.
      *****************************************************************************************],[9])
    fi

dnl **************************************************************
dnl set HYPRE_FMANGLE_BLAS flag if not set
dnl **************************************************************
    if test "$hypre_blas_link_ok" = "yes" -a "$hypre_fmangle_blas" = "0"
    then
       if test "$func" = "dgemm"
       then
          hypre_fmangle_blas=1
       elif test "$func" = "dgemm_"
       then
          hypre_fmangle_blas=2
       elif test "$func" = "dgemm__"
       then
          hypre_fmangle_blas=3
       else
          hypre_fmangle_blas=4
       fi
       AC_DEFINE_UNQUOTED(HYPRE_FMANGLE_BLAS, [$hypre_fmangle_blas], [Define as in HYPRE_FMANGLE to set the BLAS name mangling scheme])
    fi 
dnl **************************************************************
dnl Restore LIBS and LDFLAGS
dnl **************************************************************
    LIBS="$hypre_saved_LIBS"
    LDFLAGS="$hypre_saved_LDFLAGS" 
dnl  fi
])
dnl Done with macro AC_HYPRE_CHECK_USER_BLASLIBS
  
