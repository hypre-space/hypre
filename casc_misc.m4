
dnl **********************************************************************
dnl * CASC_CONFIG_OUTPUT_LIST(DIR-LIST[, OUTPUT-FILE])
dnl *
dnl * The intent of this macro is to make configure handle the possibility
dnl * that a portion of the directory tree of a project may not be
dnl * present.  This will modify the argument list of AC_OUTPUT to contain
dnl * only output file names for which corresponding input files exist.
dnl * If you are not concerned about the possible absence of the necessary
dnl * input (.in) files, it is better to not use this macro and to
dnl * explicitly list all of the output files in a call to AC_OUTPUT.
dnl * Also, If you wish to create a file Foo from a file with a name
dnl * other than Foo.in, this macro will not work, and you must use
dnl * AC_OUTPUT.
dnl *
dnl * This macro checks for the existence of the file OUTPUT-FILE.in in
dnl * each directory specified in the whitespace-separated DIR-LIST.  
dnl * (Directories should be specified by relative path from the directory 
dnl * containing configure.in.) If OUTPUT-FILE is not specified, the
dnl * default is 'Makefile'.  For each directory that contains 
dnl * OUTPUT-FILE.in, the relative path of OUTPUT-FILE is added to the 
dnl * shell variable OUTPUT-FILE_list.  When AC_OUTPUT is called,
dnl * '$OUTPUT-FILE_list' should be included in the argument list.  So if
dnl * you have a directory tree and each subdirectory contains a 
dnl * Makefile.in, DIR-LIST should be a list of every subdirectory and
dnl * OUTPUT-FILE can be omitted, because 'Makefile' is the default.  When
dnl * configure runs, it will check for the existence of a Makefile.in in
dnl * each directory in DIR-LIST, and if so, the relative path of each
dnl * intended Makefile will be added to the variable Makefile_list.
dnl *
dnl * This macro can be called multiple times, if there are files other
dnl * than Makefile.in with a .in suffix other that are intended to be 
dnl * processed by configure. 
dnl *
dnl * Example
dnl *     If directories dir1 and dir2 both contain a file named Foo.in, 
dnl *     and you wish to use configure to create a file named Foo in each
dnl *     directory, then call 
dnl *     CASC_CONFIG_OUTPUT_LIST(dir1 dir2, Foo)
dnl *     If you also called this macro for Makefile as described above,
dnl *     you should call
dnl *     AC_OUTPUT($Makefile_list $Foo_list)
dnl *     at the end of configure.in .
dnl *********************************************************************


AC_DEFUN(CASC_CONFIG_OUTPUT_LIST,
[
   dnl * m_OUTPUT_LIST is a macro to store the name of the variable
   dnl * which will contain the list of output files
   define([m_OUTPUT_LIST], ifelse([$2], , Makefile_list, [$2_list]))

   if test -z "$srcdir"; then
      srcdir=.
   fi

   dnl * use "Makefile" if second argument not given
   if test -n "$2"; then
      casc_output_file=$2
   else   
      casc_output_file=Makefile
   fi   
      
   dnl * Add a file to the output list if its ".in" file exists.
   for casc_dir in $1; do
      if test -f $srcdir/$casc_dir/$casc_output_file.in; then
         m_OUTPUT_LIST="$m_OUTPUT_LIST $casc_dir/$casc_output_file"
      fi
   done
])dnl


dnl **********************************************************************
dnl * CASC_GUESS_ARCH
dnl * Guesses a one-word name for the current architecture, unless ARCH
dnl * has been preset.  This is an alternative to the built-in macro
dnl * AC_CANONICAL_HOST, which gives a three-word name.  Uses the utility
dnl * 'tarch', which is a Bourne shell script that should be in the same  
dnl * directory as the configure script.  If tarch is not present or if it
dnl * fails, ARCH is set to the value, if any, of shell variable HOSTTYPE,
dnl * otherwise ARCH is set to "unknown".
dnl **********************************************************************

AC_DEFUN(CASC_GUESS_ARCH,
[
   AC_MSG_CHECKING(the architecture)

   dnl * $ARCH could already be set in the environment or earlier in configure
   dnl * Use the preset value if it exists, otherwise go throug the procedure
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
         ARCH=$HOSTTYPE
      else
         ARCH="`$casc_tarch`"

         if test -z "$ARCH" || test "$ARCH" = "unknown"; then
            ARCH=$HOSTTYPE
         fi
      fi

      dnl * if $ARCH is still empty, give it the value "unknown".
      if test -z "$ARCH"; then
         ARCH=unknown
         AC_MSG_WARN(architecture is unknown)
      else
         AC_MSG_RESULT($ARCH)
      fi    
   else
      AC_MSG_RESULT($ARCH)
   fi

   AC_SUBST(ARCH)

])dnl


dnl **********************************************************************
dnl * CASC_SET_SUFFIX_RULES is not like the other macros in aclocal.m4
dnl * because it does not run any kind of test on the system on which it
dnl * is running.  All it does is create several variables which contain
dnl * the text of some simple implicit suffix rules that can be
dnl * substituted into Makefile.in.  The suffix rules that come from the
dnl * macro all deal with compiling a source file into an object file.  If
dnl * this macro is called in configure.in, then if `@CRULE@' is placed in
dnl * Makefile.in, the following will appear in the generated Makefile:
dnl *
dnl * .c.o:
dnl *         @echo "Making (c) " $@ 
dnl *         @${CC} -o $@ -c ${CFLAGS} $<	
dnl *
dnl * The following is a list of the variables created by this macro and
dnl * the corresponding suffixes of the files that each implicit rule 
dnl * deals with.
dnl *
dnl * CRULE       --   .c
dnl * CXXRULE     --   .cxx
dnl * CPPRULE     --   .cpp
dnl * CCRULE      --   .cc
dnl * CAPCRULE    --   .C
dnl * F77RULE     --   .f
dnl *
dnl * There are four suffix rules for C++ files because of the different
dnl * suffixes that can be used for C++.  Only use the one which
dnl * corresponds to the suffix you use for your C++ files.
dnl *
dnl * The rules created by this macro require you to use the following
dnl * conventions for Makefile variables:
dnl *
dnl * CC        = C compiler
dnl * CXX       = C++ compiler
dnl * F77       = Fortran 77 compiler
dnl * CFLAGS    = C compiler flags
dnl * CXXFLAGS  = C++ compiler flags
dnl * FFLAGS    = Fortran 77 compiler flags
dnl **********************************************************************

AC_DEFUN(CASC_SET_SUFFIX_RULES,
[
   dnl * Things weren't working whenever "$@" showed up in the script, so
   dnl * I made the symbol $at_sign to signify '@'
   at_sign=@

   dnl * All of the backslashes are used to handle the $'s and the
   dnl * newlines which get passed through echo and sed.

   CRULE=`echo ".c.o:\\\\
\t@echo \"Making (c) \" \\$$at_sign \\\\
\t@\\${CC} -o \\$$at_sign -c \\${CFLAGS} \$<"`

   AC_SUBST(CRULE)

   CXXRULE=`echo ".cxx.o:\\\\
\t@echo \"Making (c++) \" \\$$at_sign \\\\
\t@\\${CXX} -o \\$$at_sign -c \\${CXXFLAGS} \$<"`

   AC_SUBST(CXXRULE)

   CPPRULE=`echo ".cpp.o:\\\\
\t@echo \"Making (c++) \" \\$$at_sign \\\\
\t@\\${CXX} -o \\$$at_sign -c \\${CXXFLAGS} \$<"`

   AC_SUBST(CPPRULE)

   CCRULE=`echo ".cc.o:\\\\
\t@echo \"Making (c++) \" \\$$at_sign \\\\
\t@\\${CXX} -o \\$$at_sign -c \\${CXXFLAGS} \$<"`

   AC_SUBST(CCRULE)

   CAPCRULE=`echo ".C.o:\\\\
\t@echo \"Making (c++) \" \\$$at_sign \\\\
\t@\\${CXX} -o \\$$at_sign -c \\${CXXFLAGS} \$<"`

   AC_SUBST(CAPCRULE)

   F77RULE=`echo ".f.o:\\\\
\t@echo \"Making (f) \" \\$$at_sign \\\\
\t@\\${F77} -o \\$$at_sign -c \\${FFLAGS} \$<"`

   AC_SUBST(F77RULE)

])



