
dnl ******************************************************************
dnl * CASC_PROG_F77 searches the PATH for an available Fortran 77
dnl * compiler. It assigns the name of the compiler to F77.
dnl ******************************************************************

AC_DEFUN(CASC_PROG_F77,
[
dnl   AC_BEFORE([$0], [AC_PROG_CPP])dnl
   AC_CHECK_PROGS(F77, f77 g77 xlf cf77 if77 nf77)
   test -z "$F77" && AC_MSG_ERROR([no acceptable f77 found in \$PATH])
   FFLAGS="-g -O"
   AC_SUBST(FFLAGS)
])dnl


dnl **********************************************************************
dnl * CASC_PROG_FPP searches the PATH for a preprocessor for Fortran files
dnl * with preprocessor directives
dnl **********************************************************************

AC_DEFUN(CASC_PROG_FPP,
[
   AC_CHECK_PROGS(FPP, fpp cpp "$CC -E" "cc -E" "gcc -E")
   test -z "$FPP" && AC_MSG_ERROR([no acceptable fpp found in \$PATH])
])dnl


dnl **********************************************************************
dnl * CASC_CHECK_F77_PP checks whether the preprocessor needs to
dnl * be called before calling the compiler for Fortran files with
dnl * preprocessor directives.  If the preprocessor is necessary,
dnl * F77NEEDSPP is set to "yes", otherwise it is set to "no"
dnl **********************************************************************

AC_DEFUN(CASC_CHECK_F77_PP,
[
   AC_REQUIRE([CASC_PROG_F77])

   rm -f testpp.o

   AC_MSG_CHECKING(whether $FPP needs to be called before $F77)

   # This is a dumb little fortran program with C preprocessor calls
   # It will compile only if $F77 has a built-in preprocessor

   cat > testpp.F << EOF
#define FOO 3
	program testpp
	integer num
        integer sum
        num = FOO
#ifdef FOO
        sum = num + num
#else
        sum = num + num + num
#endif
        end 
EOF

   # Compile the program and set $F77NEEDSPP appropriately
   $F77 -DBAR -c testpp.F 
   if test -f testpp.o; then 
      F77NEEDSPP=no 
   else 
      F77NEEDSPP=yes 
   fi

   AC_MSG_RESULT($F77NEEDSPP)
   rm -f testpp.o testpp.F

   AC_SUBST(F77NEEDSPP)
])dnl



dnl **********************************************************************
dnl * CASC_CHECK_LIB_FORTRAN(LIBRARY, [, ACTION-IF-FOUND [,
dnl *                        ACTION-IF-NOT-FOUND [, OTHER-LIBRARIEs]]])
dnl * 
dnl * Checks whether LIBRARY can be used to link a sample C function
dnl * that contains a call to a sample Fortran 77 function. If linking
dnl * is successful, ACTION-IF-FOUND is executed, otherwise
dnl * ACTION-IF-NOT-FOUND is executed.  The default for ACTION-IF-FOUND is
dnl * to add -lLIBRARY to $LIBS.  The default for ACTION-IF-NOT-FOUND is
dnl * nothing.  OTHER-LIBRARIES can include the full names of libraries in
dnl * the current directory, -l flags specifying other libraries, -L tags
dnl * specifying the location of libraries (This macro may not check the
dnl * same lib directories as would be checked by the linker by default on
dnl * the command line.), and/or the names of object files, all separated
dnl * by a space, whatever might be necessary for successful linkage.
dnl **********************************************************************

AC_DEFUN(CASC_CHECK_LIB_FORTRAN,
[
   # This macro needs a f77 compiler and knowledge of the name-mangling scheme
   AC_REQUIRE([CASC_PROG_F77])
   AC_REQUIRE([PAC_GET_FORTNAMES])

   if test -z "$FORTRANNAMES" ; then
      NAMESTYLE="FORTRANNOUNDERSCORE"
   else
      NAMESTYLE=$FORTRANNAMES
   fi

   # This is a little subroutine to be called later by a C main function
   cat > testflib_.f << EOF
        subroutine testflib(i)
        integer i
        print *, "This tests which libraries work"
        return
        end
EOF

   $F77 -c testflib_.f

   # Mangle testflib's name appropriatiately
   case $NAMESTYLE in
      FORTRANDOUBLEUNDERSCORE)
         THIS_FUNCTION=testflib_;;

      FORTRANUNDERSCORE)
         THIS_FUNCTION=testflib_;;

      FORTRANCAPS)
         THIS_FUNCTION=TESTFLIB;;

      FORTRANNOUNDERSCORE)
         THIS_FUNCTION=testflib;;

   esac

   # Checks if the LIBRARY from the argument list can be used to link
   # a C test program with testflib
   CASC_CHECK_LIB($1, $THIS_FUNCTION, $2, $3, testflib_.o $4)

   rm -f testflib_.o testflib_.f

])dnl


dnl ********************************************************************* 
dnl * CASC_SET_F77LIBS sets the necessary library flags for linking C and
dnl * Fortran 77 codes with the C linker.  The necessary -l flags are put 
dnl * into the variable F77LIBS, and the necessary -L flags are put into  
dnl * the variable F77LIBDIRS.  This macro first checks to see if the
dnl * shell variable $F77LIBS is already set.  If so, the preset value is
dnl * used.  Otherwise this macro only works for known architectures.
dnl *********************************************************************

AC_DEFUN(CASC_SET_F77LIBS,
[
   AC_REQUIRE([CASC_GUESS_ARCH])
   AC_REQUIRE([CASC_PROG_F77])

   if test -z "$casc_f77_libs"; then
      case $ARCH in 
         sun4 | solaris)
            case $F77 in
               *g77)
                   CASC_CHECK_LIB_FORTRAN(f2c,
                                          F77LIBDIRS="-L/home/casc/g77/lib"
                                          F77LIBS="-lf2c"
                                          , ,
                                          -L/home/casc/g77/lib -lf2c);;
               *)
                  CASC_CHECK_LIB_FORTRAN(sunmath,
                                   F77LIBDIRS="-L/opt/SUNWspro/SC4.2/lib"
                                   F77LIBS="-lF77 -lsunmath"
                                   , ,
                                   -L/opt/SUNWspro/SC4.2/lib -lF77 -lsunmath);;
               esac
         ;;
         alpha)
            CASC_CHECK_LIB_FORTRAN(for, F77LIBS="-lfor", , );;

         rs6000)
            CASC_CHECK_LIB_FORTRAN(xlf90, F77LIBS="-lxlf90", , );;

         IRIX64 | iris4d)
            CASC_CHECK_LIB_FORTRAN(I77, 
                                  F77LIBS="-lF77 -lU77 -lI77 -lisam", ,
                                  -lF77 -lU77 -lI77 -lisam);;

         *)
            AC_MSG_WARN(
[unable to set F77LIBFLAGS.  They must be set as a shell variable or
 with a command-line option])
         ;;             

      esac

   else
      if test -n "$casc_f77_lib_dirs"; then
         for casc_lib_dir in $casc_f77_lib_dirs; do
            F77LIBDIRS="-L$casc_lib_dir $F77LIBDIRS"       
         done
      fi

      for casc_lib in $casc_f77_libs; do
         F77LIBS="$F77LIBS -l$casc_lib"
      done
   fi

   F77LIBFLAGS="$F77LIBDIRS $F77LIBS"
])dnl


dnl *********************************************************************
dnl * CASC_FIND_F77LIBS may be a replacement for CASC_SET_F77LIBS.  This
dnl * macro automatically finds the flags necessary to located the 
dnl * libraries needed for a Fortran/C interface.  It is more robust than
dnl * CASC_SET_F77LIBS, because it is not based on the architecture name.
dnl * The test is performed directly on the Fortran compiler using the
dnl * macro LF_FLIBS. When CASC_FIND_F77LIBS is included in configure.in,
dnl * it will set the variable F77LIBFLAGS to be a list of flags, which 
dnl * will probably be a set of -L, -R, -l, and -u flags, as well as
dnl * perhaps the absolute paths of some libraries.  The drawback to this
dnl * macro is that it will usually insert some flags (mostly -L flags) 
dnl * that aren't needed, but hopefully they will be harmless.  I haven't
dnl * seen the extra flags that are included by this macro break anything
dnl * yet.  Hopefully more testing on more machines will give confidence
dnl * that this really works and will be able to set up the Fortran links
dnl * on an unknown system.  If this macro sets up nothing, then
dnl * CASC_SET_F77LIBS is called as a backup
dnl *********************************************************************

AC_DEFUN(CASC_FIND_F77LIBS,
[

   if test -z "$F77LIBFLAGS"; then

      dnl * LF_FLIBS creates variable $flibs_result containing a list of 
      dnl * flags related to the Fortran compiler
      LF_FLIBS

      for casc_flag in $flibs_result; do

         dnl * Here we sort the flags in $flibs_result
         case $casc_flag in
         -l* | /*)
            casc_f77_libs="$casc_f77_libs $casc_flag"
         ;;
         -L*)
            casc_f77_dirs="$casc_flag $casc_f77_dirs"
         ;;
         *)
            casc_other_flags="$casc_other_flags $casc_flag"
         ;;
         esac

      done

      F77LIBFLAGS="$casc_other_flags $casc_f77_dirs"

      if test -n "`echo $F77LIBFLAGS | grep '\-R/'`"; then
         F77LIBFLAGS=`echo $F77LIBFLAGS | sed 's/-R\//-R \//'`
      fi

      dnl * each -l flag is checked using CASC_CHECK_LIB_FORTRAN, until
      dnl * successful linking of a test program is accomplished, at which
      dnl * time the loop is broken.  If successful linking does not occur,
      dnl * CASC_CHECK_LIB will check for the library's existence and add
      dnl * to F77LIBFLAGS if it exists.  All libraries listed by explicit
      dnl * path are added to F77LIBFLAGS
      for casc_flag in $casc_f77_libs; do
         case $casc_flag in
         /*)
            if test -f "$casc_flag"; then
               F77LIBFLAGS="$F77LIBFLAGS $casc_flag"
            fi
         ;;
         -l*)
            casc_lib_name=`echo "$casc_flag" | sed 's/-l//g'`
            CASC_CHECK_LIB_FORTRAN($casc_lib_name,
               F77LIBFLAGS="$F77LIBFLAGS $casc_flag"
               casc_result=yes,
               F77LIBFLAGS="$F77LIBFLAGS $casc_flag",
               $F77LIBFLAGS)

            if test "$casc_result" = yes; then 
               casc_result=
               break
            fi
         ;;
         esac
      done

      # if this macro didn't work call CASC_SET_F77LIBS
      if test -z "$F77LIBFLAGS"; then
         CASC_SET_F77LIBS
      fi        

      dnl * IBM MPI uses /usr/lpp/ppe.poe/libc.a instead of /lib/libc.a
      dnl * so we need to make sure that -L/lib is not part of the 
      dnl * linking line when we use IBM MPI.  This only appears in
      dnl * configure when CASC_FIND_MPI is called first.
      ifdef([AC_PROVIDE_CASC_FIND_MPI], 
         if test -n "`echo $F77LIBFLAGS | grep '\-L/lib '`"; then
            if test -n "`echo $F77LIBFLAGS | grep xlf`"; then
               F77LIBFLAGS=`echo $F77LIBFLAGS | sed 's/-L\/lib //g'`
            fi
         fi
      )

   fi

   AC_SUBST(F77LIBFLAGS)

])dnl


dnl * The following are macros copied from outside sources


dnl ********************************************************************
dnl * CASC_GET_FORTNAMES is a wrapper for the macro PAC_GET_FORTNAMES.
dnl * The two can be used interchangeably.
dnl *
dnl * PAC_GET_FORTNAMES is distributed with mpich.  It checks what format
dnl * is used to call Fortran subroutines in C functions.  This macro
dnl * defines the shell variable $FORTRANNAMES and creates -D  
dnl * preprocessor flags that tell what the Fortran name-mangling is.  The
dnl * preprocessor macros defined are FORTRAN_DOUBLE_UNDERSCORE,
dnl * FORTRAN_UNDERSCORE, FORTRAN_CAPS, and FORTRAN_NO_UNDERSCORE.  The
dnl * possible values for FORTRANNAMES are the same words without
dnl * underscores.
dnl * 
dnl * Changes:
dnl *    AC_DEFINE lines to define preprocessor macros that are assigned
dnl *    to DEFS added by Noah Elliott May 18, 1998
dnl ********************************************************************

AC_DEFUN(CASC_GET_FORTNAMES,
[
   PAC_GET_FORTNAMES
])dnl

AC_DEFUN(PAC_GET_FORTNAMES,[
   # Check for strange behavior of Fortran.  For example, some FreeBSD
   # systems use f2c to implement f77, and the version of f2c that they
   # use generates TWO (!!!) trailing underscores 
   # Currently, WDEF is not used but could be...
   #
   # Eventually, we want to be able to override the choices here and
   # force a particular form.  This is particularly useful in systems
   # where a Fortran compiler option is used to force a particular
   # external name format (rs6000 xlf, for example).
   cat > confftest.f <<EOF
       subroutine mpir_init_fop( a )
       integer a
       a = 1
       return
       end
EOF
   $F77 $FFLAGS -c confftest.f > /dev/null 2>&1
   if test ! -s confftest.o ; then
        AC_MSG_ERROR([Unable to compile a Fortran test program])
        NOF77=1
        HAS_FORTRAN=0
   elif test -z "$FORTRANNAMES" ; then
    # We have to be careful here, since the name may occur in several  
    # forms.  We try to handle this by testing for several forms
    # directly.
    if test $arch_CRAY ; then
     # Cray doesn't accept -a ...
     nameform1=`strings confftest.o | grep mpir_init_fop_  | head -1`
     nameform2=`strings confftest.o | grep MPIR_INIT_FOP   | head -1`
     nameform3=`strings confftest.o | grep mpir_init_fop   | head -1`
     nameform4=`strings confftest.o | grep mpir_init_fop__ | head -1`
    else
     nameform1=`strings -a confftest.o | grep mpir_init_fop_  | head -1`
     nameform2=`strings -a confftest.o | grep MPIR_INIT_FOP   | head -1`
     nameform3=`strings -a confftest.o | grep mpir_init_fop   | head -1`
     nameform4=`strings -a confftest.o | grep mpir_init_fop__ | head -1`
    fi
    /bin/rm -f confftest.f confftest.o
    if test -n "$nameform4" ; then
        AC_DEFINE(FORTRAN_DOUBLE_UNDERSCORE)
        echo "Fortran externals are lower case and have 1 or 2 trailing underscores"
        FORTRANNAMES="FORTRANDOUBLEUNDERSCORE"
    elif test -n "$nameform1" ; then
        # We don't set this in CFLAGS; it is a default case
        AC_DEFINE(FORTRAN_UNDERSCORE)
        echo "Fortran externals have a trailing underscore and are lowercase"
        FORTRANNAMES="FORTRANUNDERSCORE"
    elif test -n "$nameform2" ; then
        AC_DEFINE(FORTRAN_CAPS)
        echo "Fortran externals are uppercase"
        FORTRANNAMES="FORTRANCAPS"
    elif test -n "$nameform3" ; then
        AC_DEFINE(FORTRAN_NO_UNDERSCORE)
        echo "Fortran externals are lower case"
        FORTRANNAMES="FORTRANNOUNDERSCORE"
    else
        AC_MSG_ERROR([Unable to determine the form of Fortran external names])
#       print_error "If you have problems linking, try using the -nof77 option"
#        print_error "to configure and rebuild MPICH."
        NOF77=1
        HAS_FORTRAN=0
    fi
    fi
    rm -f confftest.f confftest.o
    if test -n "$FORTRANNAMES" ; then
        WDEF="-D$FORTRANNAMES"
    fi  
    ])dnl


dnl ****************************************************************
dnl * LF_FLIBS was copied from E. Gkioulekas' autotools package for use in
dnl * CASC_FIND_F77LIBS.  It's probably not good to be used all by itself.
dnl * It uses the output the Fortran compiler gives when given a -v flag
dnl * to produce a list of flags that the Fortran compiler uses.  From
dnl * this list CASC_FIND_F77LIBS sets up the Fortran/C interface flags.
dnl *
dnl * Changes:
dnl * AC_SUBST(FLIBS) suppressed by N. Elliott 7-10-98
dnl *****************************************************************
   
AC_DEFUN(LF_FLIBS,[
  AC_MSG_CHECKING(for Fortran libraries)
  dnl
  dnl Write a minimal program and compile it with -v. I don't know
  dnl what to do if your compiler doesn't have -v
  dnl
  changequote(, )dnl
  echo "      END" > conftest.f
  foutput=`${F77-f77} -v -o conftest conftest.f 2>&1`
  dnl
  dnl The easiest thing to do for xlf output is to replace all the commas
  dnl with spaces.  Try to only do that if the output is really from xlf,
  dnl since doing that causes problems on other systems.
  dnl
  xlf_p=`echo $foutput | grep xlfentry`
  if test -n "$xlf_p"; then
    foutput=`echo $foutput | sed 's/,/ /g'`
  fi
  dnl
  ld_run_path=`echo $foutput | \
    sed -n -e 's/^.*LD_RUN_PATH *= *\([^ ]*\).*/\1/p'`
  dnl
  dnl We are only supposed to find this on Solaris systems...
  dnl Uh, the run path should be absolute, shouldn't it?
  dnl
  case "$ld_run_path" in
    /*)
      if test "$ac_cv_prog_gcc" = yes; then
        ld_run_path="-Xlinker -R -Xlinker $ld_run_path"
      else
        ld_run_path="-R $ld_run_path"
      fi
    ;;
    *)
      ld_run_path=
    ;;
  esac
  dnl
  flibs=
  lflags=
  dnl
  dnl If want_arg is set, we know we want the arg to be added to the list,
  dnl so we don't have to examine it.
  dnl
  want_arg=
  dnl
  for arg in $foutput; do
    old_want_arg=$want_arg
    want_arg=
  dnl
  dnl None of the options that take arguments expect the argument to
  dnl start with a -, so pretend we didn't see anything special.
  dnl
    if test -n "$old_want_arg"; then
      case "$arg" in
        -*)
        old_want_arg=
        ;;
      esac
    fi
    case "$old_want_arg" in
      '')
        case $arg in
        /*.a)
          exists=false
          for f in $lflags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            lflags="$lflags $arg"
          fi
        ;;
        -bI:*)
          exists=false
          for f in $lflags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            if test "$ac_cv_prog_gcc" = yes; then
              lflags="$lflags -Xlinker $arg"
            else
              lflags="$lflags $arg"
            fi
          fi
        ;;
        -lang* | -lcrt0.o | -lc )
          arg=
        ;;
        -[lLR])
          want_arg=$arg
          arg=
        ;;
        -[lLR]*)
          exists=false
          for f in $lflags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            case "$arg" in
              -lkernel32)
                case "$canonical_host_type" in
                  *-*-cygwin32)
                  ;;
                  *)
                    lflags="$lflags $arg"
                  ;;
                esac
              ;;
              -lm)
              ;;
              *)
                lflags="$lflags $arg"
              ;;
            esac
          fi
        ;;
        -u)
          want_arg=$arg
          arg=
        ;;
        -Y)
          want_arg=$arg
          arg=
        ;;
        *)
          arg=
        ;;
        esac
      ;;
      -[lLR])
        arg="$old_want_arg $arg"
      ;;
      -u)
        arg="-u $arg"
      ;;
      -Y)
  dnl
  dnl Should probably try to ensure unique directory options here too.
  dnl This probably only applies to Solaris systems, and then will only
  dnl work with gcc...
  dnl
        arg=`echo $arg | sed -e 's%^P,%%'`
        SAVE_IFS=$IFS
        IFS=:
        list=
        for elt in $arg; do
        list="$list -L$elt"
        done
        IFS=$SAVE_IFS
        arg="$list"
      ;;
    esac
  dnl
    if test -n "$arg"; then
      flibs="$flibs $arg"
    fi
  done
  if test -n "$ld_run_path"; then
    flibs_result="$ld_run_path $flibs"
  else
    flibs_result="$flibs"
  fi
  changequote([, ])dnl
  rm -f conftest.f conftest.o conftest
  dnl
  dnl Phew! Done! Now, output the result
  dnl
  FLIBS="$flibs_result"
  AC_MSG_RESULT([$FLIBS])
dnl  AC_SUBST(FLIBS)
])dnl

