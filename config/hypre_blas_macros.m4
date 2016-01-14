dnl #*BHEADER********************************************************************
dnl # Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
dnl # Produced at the Lawrence Livermore National Laboratory.
dnl # This file is part of HYPRE.  See file COPYRIGHT for details.
dnl #
dnl # HYPRE is free software; you can redistribute it and/or modify it under the
dnl # terms of the GNU Lesser General Public License (as published by the Free
dnl # Software Foundation) version 2.1 dated February 1999.
dnl #
dnl # $Revision$
dnl #EHEADER*********************************************************************



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
  LIBS="$LIBS $FCLIBS"

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
  hypre_blas_link_ok=no
dnl **************************************************************
dnl Get fortran linker name for test function (dgemm in this case)
dnl **************************************************************
  AC_FC_FUNC(dgemm)
dnl **************************************************************
dnl Get user provided path-to-blas library
dnl **************************************************************
dnl  LDBLASLIBDIRS="$BLASLIBDIRS"
  USERBLASLIBS="$BLASLIBS"
  USERBLASLIBDIRS="$BLASLIBDIRS"  
  BLASLIBPATHS=""
  BLASLIBNAMES=""
  SUFFIXES=""
  for blas_dir in $BLASLIBDIRS; do
    [blas_dir=${blas_dir##*-L}]
    BLASLIBPATHS="$BLASLIBPATHS $blas_dir"
  done

dnl Case where explicit path could be given by the user
  for blas_lib in $BLASLIBS; do
    [blas_lib_name=${blas_lib##*-l}]
    if test $blas_lib = $blas_lib_name;
    then
      [libsuffix=${blas_lib##*.}]
      SUFFIXES="$SUFFIXES $libsuffix"
      [dir_path=${blas_lib%/*}]
      BLASLIBPATHS="$dir_path $BLASLIBPATHS"
      [blas_lib_name=${blas_lib_name%%.*}]                  
      [blas_lib_name=${blas_lib_name##*/}]  
      [blas_lib_name=${blas_lib_name#*lib}]
      BLASLIBNAMES="$BLASLIBNAMES $blas_lib_name"
    else
      BLASLIBNAMES="$BLASLIBNAMES $blas_lib_name"
    fi
  done
dnl  echo SUFFIXES=$SUFFIXES
dnl  echo BLASLIBS=$BLASLIBNAMES
dnl  echo BLASLIBPATHS=$BLASLIBPATHS
dnl **************************************************************
dnl Begin test:
dnl **************************************************************
  BLASLIBS="null"
  BLASLIBDIRS="null"
dnl  if test "x$BLASLIBNAMES" != "x";
dnl  then

dnl **************************************************************
dnl Test for viable library path:
dnl **************************************************************  
  for dir in $BLASLIBPATHS; do
     if test $BLASLIBDIRS = "null"; then
        for blas_lib in $BLASLIBNAMES; do
           if test "x$SUFFIXES" = "x"; then
              if test $BLASLIBS = "null" -a -f $dir/lib$blas_lib.a; then
                 BLASLIBDIRS="-L$dir"
                 BLASLIBS="-l$blas_lib"
              fi
              if test $BLASLIBS = "null" -a -f $dir/lib$blas_lib.so; then
                 BLASLIBDIRS="-L$dir"
                 BLASLIBS="-l$blas_lib"
              fi
           else
              for libsuffix in $SUFFIXES; do
                 if test $BLASLIBS = "null" -a -f $dir/lib$blas_lib.$libsuffix; then
                    BLASLIBDIRS="-L$dir"
                    BLASLIBS="-l$blas_lib"
                 fi
              done
           fi              
        done
     fi
  done
        
  if test $BLASLIBDIRS = "null" -o $BLASLIBS = "null"; then
     AC_MSG_ERROR([**************** Incorrect path to blas library error: ***************************
     User-specified blas library path is incorrect or cannot be used. Please specify full path to 
     blas library or check that the provided library path exists. If a library file is not explicitly
     specified, configure checks for library files with extensions ".a" or ".so" in 
     the user-specified path. Otherwise use --with-blas option to find the library on the system. 
     See "configure --help" for usage details.
     *****************************************************************************************],[9])           
dnl  else
dnl     echo BLASLIBDIRS=$BLASLIBDIRS
dnl     BLASLIBS="-l$BLASLIBS"
dnl     BLASLIBDIRS="-L$BLASLIBDIRS"
  fi     
    
dnl **************************************************************
dnl Save current LIBS and LDFLAGS to be restored later 
dnl **************************************************************
    hypre_saved_LIBS="$LIBS"
    hypre_saved_LDFLAGS="$LDFLAGS"
    LIBS="$LIBS $FCLIBS"
    LDFLAGS="$BLASLIBDIRS $LDFLAGS"

dnl    echo LDFLAGS=$LDFLAGS
dnl    echo LIBS=$LIBS
dnl    echo BLASLIBPATHS=$BLASLIBPATHS
dnl **************************************************************
dnl Check for dgemm in linkable list of libraries
dnl **************************************************************
dnl    for blas_lib in $BLASLIBS; do
dnl      if test $BLASLIBS = "null"; then
dnl **************************************************************
dnl Get library base name
dnl **************************************************************
        [blas_lib=${BLASLIBS##*-l}]
dnl        [blas_lib=${blas_lib##*/}]        
dnl        [blas_lib=${blas_lib##*lib}]
dnl        [blas_lib=${blas_lib%%.*}]
dnl **************************************************************
dnl Check if library works and print result
dnl **************************************************************                   
dnl        AC_CHECK_LIB($blas_lib, $dgemm, [BLASLIBS=$blas_lib])
        AC_CHECK_LIB($blas_lib, $dgemm, [hypre_blas_link_ok=yes])

dnl      fi
dnl    done

    if test $hypre_blas_link_ok = "no"; then
      AC_MSG_ERROR([**************** Non-linkable blas library error: ***************************
      User set BLAS library path using either --with-blas-lib=<lib>, or 
      --with-blas-libs=<blas_lib_base_name> and --with-blas_dirs=<path-to-blas-lib>, 
      but $USERBLASLIBDIRS $USERBLASLIBS provided cannot be used. See "configure --help" for usage details.
      *****************************************************************************************],[9])
    fi
dnl    else
dnl **************************************************************
dnl Check if working library was provided by user
dnl **************************************************************   
dnl        for dir in $BLASLIBPATHS; do
dnl          if test $BLASLIBDIRS = "null" -a -f $dir/lib$BLASLIBS.a; then
dnl             BLASLIBDIRS="$dir"
dnl          fi
dnl          if test $BLASLIBDIRS = "null" -a -f $dir/lib$BLASLIBS.so; then
dnl             BLASLIBDIRS="$dir"
dnl          fi
dnl        done
        
dnl        if test $BLASLIBDIRS = "null"; then
dnl           AC_MSG_ERROR([**************** Incorrect path to blas library error: ***************************
dnl           Configure found a linkable blas library, but the user-specified path is incorrect.
dnl           User set BLAS library path using either --with-blas-lib=<lib>, or 
dnl           --with-blas-libs=<blas_lib_base_name> and --with-blas_dirs=<path-to-blas-lib>, 
dnl           but the library $USERBLASLIBDIRS $USERBLASLIBS cannot be used. Please check that the provided 
dnl           library path is correct. Otherwise use --with-blas option to find the library on the system. 
dnl           See "configure --help" for usage details.
dnl           *****************************************************************************************],[9])           
dnl        else
dnl           echo BLASLIBDIRS=$BLASLIBDIRS
dnl           BLASLIBS="-l$BLASLIBS"
dnl           BLASLIBDIRS="-L$BLASLIBDIRS"
dnl        fi  
dnl    fi 
dnl **************************************************************
dnl Restore LIBS and LDFLAGS
dnl **************************************************************
    LIBS="$hypre_saved_LIBS"
    LDFLAGS="$hypre_saved_LDFLAGS" 
dnl  fi
])
dnl Done with macro AC_HYPRE_CHECK_USER_BLASLIBS
