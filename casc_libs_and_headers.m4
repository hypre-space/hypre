

dnl *********************************************************************
dnl * CASC_ADD_LIB(LIBRARY, FUNCTION, DIRECTORY-LIST[, PREFIX[, 
dnl *              ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]]])
dnl * checks first if LIBRARY is available on the linking search path and
dnl * if FUNCTION can be linked with LIBRARY.  If so, -lLIBRARY is added
dnl * to the variable [PREFIX]LIBS. (i.e., if prefix is LD, -llibrary is
dnl * added to LDLIBS.)  If not, checks whitespace-separated
dnl * DIRECTORY-LIST to see if LIBRARY exists in a specified directory and
dnl * can be linked with FUNCTION.  If so, the first directory where
dnl * linking is successful is added to the front of [PREFIX]LIBDIRS, and
dnl * -lLIBRARY is added to the end of [PREFIX]LIBS.  If no prefix is
dnl * specified, the directories and libraries are added to LIBS and
dnl * LIBDIRS, respectively.  If the order of -l flags on the linking
dnl * lines is important, CASC_ADD_LIB should be called for each library
dnl * in the order they should appear on linking lines.  Mere existence of
dnl * LIBRARY in the search path or in a specified directory can usually
dnl * be determined by entering 'main' for FUNCTION.  Optional argument
dnl * ACTION-IF-FOUND contains additional instructions to execute as soon
dnl * as LIBRARY is found in any directory.  Optional argument
dnl * ACTION-IF-NOT-FOUND contains instructions to execute if LIBRARY is
dnl * not found anywhere.
dnl **********************************************************************

AC_DEFUN(CASC_ADD_LIB,
[
   # define some macros to hopefully improve readability
   define([m_THESE_LIBS],[$4LIBS])
   define([m_THESE_LIBDIRS],[$4LIBDIRS])

   # check for the library from first argument.  If linking is successful
   # the first time, the job is done, otherwise loop through DIRECTORY-LIST
   CASC_CHECK_LIB($1, $2, m_THESE_LIBS="$m_THESE_LIBS -l$1"
                          casc_lib_found=yes 
                          ifelse([$5], , , [$5]),

      dnl * If library not found
      for casc_lib_dir in $3; do

         CASC_CHECK_LIB($1, $2, 
            m_THESE_LIBDIRS="-L$casc_lib_dir $m_THESE_LIBDIRS"
            m_THESE_LIBS="$m_THESE_LIBS -l$1"
            casc_lib_found=yes
            ifelse([$5], , , [$5])
            break
            , ,
            -L$casc_lib_dir $m_THESE_LIBDIRS $m_THESE_LIBS -l$1, no)
      done
      , $m_THESE_LIBDIRS $m_THESE_LIBS, no)  dnl * last two arguments for
                                             dnl * first check

   # ACTION-IF-NOT_FOUND for when the library is found nowhere
   ifelse([$6], , ,
      if test "$casc_lib_found" != "yes"; then
         [$6]
      fi
   )

   unset casc_lib_found

   undefine([m_THESE_LIBS])
   undefine([m_THESE_LIBDIRS])

])dnl


dnl ***********************************************************************
dnl CASC_CHECK_LIB(LIBRARY, FUNCTION [, ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND
dnl              [, OTHER-LIBRARIES [, CACHE-CHOICE]]]])
dnl * This is the same as AC_CHECK_LIB, except when it tests for LIBRARY
dnl * it puts the flag -lLIBRARY after $LIBS and OTHER-LIBRARIES.  The Sun
dnl * cc compiler does not search for LIBRARY in any directories specified
dnl * by -L in OTHER-LIBRARIES when -lLIBRARY is listed first.  The
dnl * functionality of this macro is the same as that of AC_CHECK_LIB in
dnl * the Autoconf documentation.  
dnl * CACHE-CHOICE [$6]added by N. Elliott, 6-24-98.  If CACHE-CHOICE is 'no',
dnl * the results of this test are not cached.  CACHE-CHOICE should be
dnl * used only when this test is called recursively.
dnl **********************************************************************

AC_DEFUN(CASC_CHECK_LIB,
[AC_MSG_CHECKING([for -l$1])
dnl Use a cache variable name containing both the library and function name,
dnl because the test really is for library $1 defining function $2, not
dnl just for library $1.  Separate tests with the same $1 and different $2s
dnl may have different results.
ac_lib_var=`echo $1['_']$2 | tr './+\055' '__p_'`
AC_CACHE_VAL(ac_cv_lib_$ac_lib_var,
[ac_save_LIBS="$LIBS"
LIBS="$5 $LIBS -l$1"
AC_TRY_LINK(dnl
ifelse([$2], [main], , dnl Avoid conflicting decl of main.
[/* Override any gcc2 internal prototype to avoid an error.  */
]ifelse(AC_LANG, CPLUSPLUS, [#ifdef __cplusplus 
extern "C"
#endif
])dnl
[/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char $2();
]),
            [$2()],
            eval "ac_cv_lib_$ac_lib_var=yes",
            eval "ac_cv_lib_$ac_lib_var=no")dnl
LIBS="$ac_save_LIBS"
])dnl
if eval "test \"`echo '$ac_cv_lib_'$ac_lib_var`\" = yes"; then
  AC_MSG_RESULT(yes)  
  ifelse([$3], ,
[changequote(, )dnl
  ac_tr_lib=HAVE_LIB`echo $1 | tr 'abcdefghijklmnopqrstuvwxyz' 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'`
changequote([, ])dnl
  AC_DEFINE_UNQUOTED($ac_tr_lib)
  LIBS="-l$1 $LIBS"
], [
ifelse([$6], no, unset ac_cv_lib_$ac_lib_var)
$3])
else
  AC_MSG_RESULT(no) 
ifelse([$4], , , [
ifelse([$6], no, unset ac_cv_lib_$ac_lib_var)
$4
])dnl
fi
ifelse([$6], no, unset ac_cv_lib_$ac_lib_var)
])



dnl *********************************************************************
dnl * CASC_CHECK_HEADER(HEADER-FILE, DIRECTORY-LIST[, ACTION-IF-FOUND[,
dnl *                   ACTION-IF-NOT-FOUND]])
dnl * This macro is an alternative to AC_CHECK_HEADER.  It does
dnl * essentially the same thing, but it allows the user to specify
dnl * a directory list if HEADER-FILE can not be found in the current path
dnl * for #includes, and it adds to the variable INCLUDES the first
dnl * directory in DIRECTORY-LIST from where HEADER-FILE can be included.
dnl *********************************************************************

AC_DEFUN(CASC_CHECK_HEADER,
[
   dnl * loop through the directory list.  The first iteration leaves the
   dnl * casc_dir variable empty to check if the header can be #included
   dnl * without specifying a directory.
   for casc_dir in '' $2 ; do
      if test -n "$casc_dir"; then
         casc_header=$casc_dir/$1
      else
         casc_header=$1
      fi

      dnl * Check for the header.  Add the necessary -I flag to INCLUDES
      AC_CHECK_HEADER( $casc_header,
         if test -n "$casc_dir"; then
            INCLUDES="$INCLUDES -I$casc_dir"
         fi
         casc_header_found=yes
         ifelse([$3], , , [$3])
         break )

   done

   dnl * This takes care of the action if not found
   ifelse([$4], , ,
      if test "$casc_header_found" != "yes"; then
         [$4]
      fi
   )

   unset casc_header_found
])dnl


dnl **********************************************************************
dnl * CASC_CREATE_PACKAGE_OPTION(PACKAGE-NAME[, DIR-LIST[, FILE]])
dnl * This is a general macro that creates a configure command-line option
dnl * called `--with-PACKAGE-NAME-dir' which will allow the user to
dnl * specify the location of the installation of an outside software
dnl * package, such as PETSc or ISIS++.  After a check to make sure the
dnl * given directory is valid (see below for discussion of validity), the
dnl * directory's path is stored in the shell variable PACKAGE-NAME_DIR.
dnl * For example, to allow the user to specify the location of PETSc,
dnl * place `CASC_CREATE_PACKAGE_OPTION(PETSC)' in configure.in.  Then the
dnl * user, if configuring on the CASC Sun cluster, would type `configure
dnl * --with-PETSC-dir=/home/casc/petsc', and the directory's path would
dnl * be stored in PETSC_DIR.  With this macro, the user is also permitted
dnl * to set the variable PACKAGE-NAME_DIR in the environment before
dnl * running configure, but any choice made on the command line would
dnl * override any preset values.  
dnl *
dnl * This macro takes an optional second argument, DIR-LIST, which is a
dnl * whitespace-separated list of directories where the developer thinks
dnl * PACKAGE-NAME might be installed.  If DIR-LIST is given, and the user
dnl * does not use the `--with' option to give the location of
dnl * PACKAGE-NAME (or if the directory given by the user does not exist),
dnl * then configure will assign to PACKAGE-NAME_DIR the path of the first
dnl * directory in DIR-LIST that is valid.
dnl *
dnl * Validity:  The optional third argument to this macro is FILE, which
dnl * should be either the name of a file in the top directory of the
dnl * package in question or the relative path of a file in a subdirectory
dnl * of the package.  If the argument FILE is given, then configure will
dnl * consider a user specified directory or a directory from DIR-LIST 
dnl * valid only if FILE exists in the directory.  If this argument is not
dnl * given, then configure will consider a directory valid simply if it
dnl * is indeed a directory.  FILE should be a file with a unique name
dnl * that can be expected to exist in the same location in any 
dnl * installation of the package in question.  If you know of no such
dnl * file, do not include a third argument when invoking this macro.
dnl * 
dnl * This macro also gives the user the command-line option
dnl * `--without-PACKAGE-NAME-dir', which, when invoked, will leave the
dnl * variable PACKAGE-NAME_DIR empty.  This option should be invoked when
dnl * the user wants to exclude a package from the configuration.
dnl * 
dnl * NOTE:  Since PACKAGE-NAME is used as part of both a command-line
dnl * option and a variable name, it MUST consist of only alphanumeric
dnl * characters.  PACKAGE-NAME is only a label, so it need not conform to
dnl * any existing directory or file name.  I would recommend that it be
dnl * all caps, as it becomes part of the name of a variable that is
dnl * substituted into the Makefile.
dnl **********************************************************************

AC_DEFUN(CASC_CREATE_PACKAGE_OPTION,
[
   AC_MSG_CHECKING([for $1 directory])

   dnl * $1 stands for the PACKAGE-NAME.  If [$1]_DIR has been set in the
   dnl * environment, give its value to casc_env_[$1]_dir, and clear
   dnl * [$1]_DIR.  The environmental value will ultimately be reassigned
   dnl * to [$1]_DIR if it is valid and no command-line options are able
   dnl * to change [$1]_DIR to a valid directory.  The environmental value
   dnl * will also be used even if it is invalid, if the command-line
   dnl * options and the DIRECTORY-LIST are both unable to generate a
   dnl * valid value.
   casc_result=
   casc_env_[$1]_dir=$[$1]_DIR
   [$1]_DIR=

   AC_ARG_WITH($1-dir, 
[  --with-$1-dir=DIR    $1 is installed in directory DIR
  --without-$1-dir     do not look for $1],

               if test "$withval" = "no"; then
                  casc_result="configuring without [$1]"
                  [$1]_DIR=
               fi
               , )

   dnl * If "--without-$1-dir" was given, then [$1]_DIR is left blank.
   dnl * Otherwise there is the following procedure to try to give
   dnl * [$1]_DIR a valid value:
   dnl *
   dnl * if "--with-$1-dir" was given
   dnl *    if the argument to "--with-$1-dir" is valid
   dnl *       assign the argument to [$1]_DIR
   dnl *    endif
   dnl * endif
   dnl *
   dnl * if a value for [$1]_DIR has not yet been found
   dnl *    if [$1]_DIR from the environment exists and is valid
   dnl *       assign the environmental value to [$1]_DIR
   dnl *    endif
   dnl * endif
   dnl *
   dnl * if [$1]_DIR still has no value
   dnl *    if the macro was given a DIRECTORY-LIST argument
   dnl *       for each directory in the list
   dnl *          if the directory is valid
   dnl *             assign the directory to [$1]_DIR
   dnl *             break loop
   dnl *          else
   dnl *             continue loop
   dnl *          endif
   dnl *       end loop
   dnl *       if [$1]_DIR still doesn't have a value
   dnl *          casc_result="none"
   dnl *       else
   dnl *          casc_result=$[$1]_DIR
   dnl *       endif
   dnl *    else
   dnl *       casc_result="none"
   dnl *    endif
   dnl * endif

   if test "$with_[$1]_dir" != "no"; then

      if test -d "$with_[$1]_dir"; then

         ifelse([$3], , ,
            if test -f $with_[$1]_dir/[$3]; then)

               casc_result="$with_[$1]_dir"
               [$1]_DIR="$casc_result"

         ifelse([$3], , ,
            fi)
      fi

      if test -z "$casc_result"; then

         if test -d "$casc_env_[$1]_dir"; then

            ifelse([$3], , ,
               if test -f $casc_env_[$1]_dir/[$3]; then)

                  casc_result="$casc_env_[$1]_dir"
                  [$1]_DIR="$casc_result"

            ifelse([$3], , ,
               fi)
         fi
      fi



      if test -z "$casc_result"; then
         [$1]_DIR=
   
         ifelse([$2], ,
            casc_result="none" ,

            for casc_dir in $2; do

               if test -d "$casc_dir"; then

                  ifelse([$3], , ,
                     if test -f $casc_dir/[$3]; then)

                        $1_DIR=$casc_dir

                  ifelse([$3], , ,
                     fi)

                  break
               fi
            done

            if test -z "$[$1]_DIR"; then
               casc_result="none"

            else
               casc_result="$[$1]_DIR"
            fi
         )
      fi
   fi

   dnl * $casc_result either is a valid value for [$1]_DIR or "none".
   dnl * if none, then assign the original environmental value of
   dnl * [$1]_DIR, whatever it may be, to casc_result and [$1]_DIR.  If
   dnl * there was no environmental value, then $casc_result remains
   dnl * "none" and [$1]_DIR is left empty.

   if test "$casc_result" = "none"; then

      if test -n "$casc_env_[$1]_dir"; then

         casc_result="$casc_env_[$1]_dir"
         [$1]_DIR="$casc_result"
      fi
   fi

   AC_MSG_RESULT($casc_result)
   AC_SUBST([$1]_DIR)
])


dnl smr_ARG_WITHLIB from FVWM by S. Robbins 
dnl Allow argument for optional libraries; wraps AC_ARG_WITH, to
dnl provide a "--with-foo-lib" option in the configure script, where foo
dnl is presumed to be a library name.  The argument given by the user
dnl (i.e. "bar" in ./configure --with-foo-lib=bar) may be one of four 
dnl things:
dnl     * boolean (no, yes or blank): whether to use library or not
dnl     * file: assumed to be the name of the library
dnl     * directory: assumed to *contain* the library
dnl     * a quoted, space-separated list of linker flags needed to link
dnl       with this library.  (To be used if this library requires
dnl       linker flags other than the normal `-L' and `-l' flags.)
dnl 
dnl The argument is sanity-checked.  If all is well, two variables are
dnl set: "with_foo" (value is yes, no, or maybe), and "foo_LIBFLAGS" (value
dnl is either blank, a file, -lfoo, '-L/some/dir -lfoo', or whatever 
dnl linker flags the user gives). The idea is: the first tells you whether
dnl the library is to be used or not (or the user didn't specify one way
dnl or the other) and the second to put on the command line for linking
dnl with the library.
dnl
dnl Usage:
dnl smr_ARG_WITHLIB(name, libname, description)
dnl 
dnl name                name for --with argument ("foo" for libfoo)
dnl libname             (optional) actual name of library,
dnl                     if different from name
dnl description         (optional) used to construct help string
dnl 
dnl Changes:  Changed some identifier names.
dnl           --with-foo-library is now --with-foo-lib
dnl           foo_LIBS is now foo_LIBFLAGS
dnl           Fourth posibility for argument to --with-foo-lib added
dnl           Documentation above changed to reflect these changes
dnl           Noah Elliott, October 1998


AC_DEFUN(CASC_SMR_ARG_WITHLIB,
[
   smr_ARG_WITHLIB([$1],[$2],[$3])
])dnl

AC_DEFUN(smr_ARG_WITHLIB, [

ifelse($2, , smr_lib=[$1], smr_lib=[$2]) 
    
AC_ARG_WITH([$1]-lib,
ifelse($3, ,
[  --with-$1-lib[=PATH]       use $1 library], 
[  --with-$1-lib[=PATH]       use $1 library ($3)]),
[
    if test "$withval" = yes; then
        with_[$1]=yes
        [$1]_LIBFLAGS="-l${smr_lib}"
    elif test "$withval" = no; then
        with_[$1]=no
        [$1]_LIBFLAGS=
    else
        with_[$1]=yes
        if test -f "$withval"; then
            [$1]_LIBFLAGS=$withval
        elif test -d "$withval"; then
            [$1]_LIBFLAGS="-L$withval -l${smr_lib}"
        else
            case $withval in
            -*)
               [$1]_LIBFLAGS="$withval"
            ;;
            *)
               AC_MSG_ERROR(
                  [argument must be boolean, file, directory, or compiler flags]
                           )
            ;;
            esac
        fi
    fi
], [
    with_[$1]=maybe
    [$1]_LIBFLAGS="-l${smr_lib}"
])])

    
dnl smr_ARG_WITHINCLUDES from FVWM by S. Robbins
dnl Check if the include files for a library are accessible, and
dnl define the variable "name_INCLUDE" with the proper "-I" flag for
dnl the compiler.  The user has a chance to specify the includes
dnl location, using "--with-foo-include".
dnl 
dnl This should be used *after* smr_ARG_WITHLIB *and* AC_CHECK_LIB are
dnl successful.
dnl 
dnl Usage:
dnl smr_ARG_WITHINCLUDES(name, header, extra-flags)
dnl 
dnl name                library name, MUST same as used with smr_ARG_WITHLIB
dnl header              a header file required for using the lib
dnl extra-flags         (optional) flags required when compiling the
dnl                     header, typically more includes; for ex. X_CFLAGS
dnl
dnl Changes:  Changed some identifier names.
dnl           --with-foo-includes is now --with-foo-include
dnl           name_CFLAGS is now name_INCLUDE
dnl           Documentation above changed to reflect these changes
dnl           Noah Elliott, October 1998

AC_DEFUN(CASC_SMR_ARG_WITHINCLUDES,
[
   smr_ARG_WITHINCLUDES([$1], [$2], [$3])
])dnl

AC_DEFUN(smr_ARG_WITHINCLUDES, [

AC_ARG_WITH([$1]-include,
[  --with-$1-include=DIR  set directory for $1 headers],
[
    if test -d "$withval"; then
        [$1]_INCLUDE="-I${withval}"
    else
        AC_MSG_ERROR(argument must be a directory)
    fi])

dnl This bit of logic comes from autoconf's AC_PROG_CC macro.  We need
dnl to put the given include directory into CPPFLAGS temporarily, but
dnl then restore CPPFLAGS to its old value.
dnl 
smr_test_CPPFLAGS="${CPPFLAGS+set}"
smr_save_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$CPPFLAGS ${[$1]_CFLAGS}"

    ifelse($3, , , CPPFLAGS="$CPPFLAGS [$3]")
    AC_CHECK_HEADERS($2)
   
if test "$smr_test_CPPFLAGS" = set; then
    CPPFLAGS=$smr_save_CPPFLAGS
else
    unset CPPFLAGS
fi
])
    
        
dnl smr_CHECK_LIB from FVWM by S. Robbins
dnl Probe for an optional library.  This macro creates both
dnl --with-foo-lib and --with-foo-include options for the configure
dnl script.  If --with-foo-lib is *not* specified, the default is to
dnl probe for the library, and use it if found.
dnl
dnl Usage:
dnl smr_CHECK_LIB(name, libname, desc, func, header, x-libs, x-flags)
dnl 
dnl name        name for --with options
dnl libname     (optional) real name of library, if different from
dnl             above
dnl desc        (optional) short descr. of library, for help string
dnl func        function of library, to probe for
dnl header      (optional) header required for using library
dnl x-libs      (optional) extra libraries, if needed to link with lib
dnl x-flags     (optional) extra flags, if needed to include header files
dnl
dnl Changes:  identifier names and documentation modified to reflect
dnl           changes to smr_ARG_WITHLIB and smr_ARG_WITHINCLUDES
dnl           Noah Elliott, October 1998

AC_DEFUN(CASC_SMR_CHECK_LIB,
[
   smr_CHECK_LIB([$1], [$2], [$3], [$4], [$5], [$6], [$7])
])dnl

AC_DEFUN(smr_CHECK_LIB,
[   
ifelse($2, , smr_lib=[$1], smr_lib=[$2])
ifelse($5, , , smr_header=[$5])
smr_ARG_WITHLIB($1,$2,$3)
if test "$with_$1" != no; then
    AC_CHECK_LIB($smr_lib, $4,
        smr_havelib=yes, smr_havelib=no,
        ifelse($6, , ${$1_LIBFLAGS}, [${$1_LIBFLAGS} $6]))
    if test "$smr_havelib" = yes -a "$smr_header" != ""; then
        smr_ARG_WITHINCLUDES($1, $smr_header, $7)
        smr_safe=`echo "$smr_header" | sed 'y%./+-%__p_%'`
        if eval "test \"`echo '$ac_cv_header_'$smr_safe`\" != yes"; then
            smr_havelib=no
        fi
    fi
    if test "$smr_havelib" = yes; then
        AC_MSG_RESULT(Using $1 library)
    else
        $1_LIBFLAGS=
        $1_INCLUDE=
        if test "$with_$1" = maybe; then
            AC_MSG_RESULT(Not using $1 library)
        else
            AC_MSG_WARN(Requested $1 library not found!)
        fi
    fi
fi])
