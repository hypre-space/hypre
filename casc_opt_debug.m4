
dnl *********************************************************************
dnl * CASC_SET_COPT(OPTIMIZATION-FLAGS)
dnl * Call this macro to set C compiler optimization flags to
dnl * OPTIMIZATION-FLAGS.  They will be stored in COPT.  Existing values
dnl * of COPT in the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_COPT,
[
   if test -z "$COPT"; then
      COPT="$1"
   fi
   AC_SUBST(COPT)
])dnl


dnl *********************************************************************
dnl * CASC_SET_CXXOPT(OPTIMIZATION-FLAGS)
dnl * Call this macro to set C++ compiler optimization flags to
dnl * OPTIMIZATION-FLAGS.  They will be stored in CXXOPT.  Existing values
dnl * of CXXOPT in the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_CXXOPT,
[
   if test -z "$CXXOPT"; then
      CXXOPT="$1"
   fi
   AC_SUBST(CXXOPT)
])dnl


dnl *********************************************************************
dnl * CASC_SET_FOPT(OPTIMIZATION-FLAGS)
dnl * Call this macro to set F77 compiler optimization flags to
dnl * OPTIMIZATION-FLAGS.  They will be stored in FOPT.  Existing values
dnl * of FOPT in the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_FOPT,
[
   if test -z "$FOPT"; then
      FOPT="$1"
   fi
   AC_SUBST(FOPT)
])dnl


dnl *********************************************************************
dnl * CASC_SET_CDEBUG(DEBUG-FLAGS)
dnl * Call this macro to set C compiler debugging flags to DEBUG-FLAGS.
dnl * They will be stored in CDEBUG.  Existing values of CDEBUG in
dnl * the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_CDEBUG,
[
   if test -z "$CDEBUG"; then
      CDEBUG="$1"
   fi
   AC_SUBST(CDEBUG)
])dnl


dnl *********************************************************************
dnl * CASC_SET_CXXDEBUG(DEBUG-FLAGS)
dnl * Call this macro to set C++ compiler debugging flags to DEBUG-FLAGS.
dnl * They will be stored in CXXDEBUG.  Existing values of CXXDEBUG in
dnl * the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_CXXDEBUG,
[
   if test -z "$CXXDEBUG"; then
      CXXDEBUG="$1"
   fi
   AC_SUBST(CXXDEBUG)
])dnl


dnl *********************************************************************
dnl * CASC_SET_FDEBUG(DEBUG-FLAGS)
dnl * Call this macro to set F77 compiler debugging flags to DEBUG-FLAGS.
dnl * They will be stored in FDEBUG.  Existing values of FDEBUG in
dnl * the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_FDEBUG,
[
   if test -z "$FDEBUG"; then
      FDEBUG="$1"
   fi
   AC_SUBST(FDEBUG)
])dnl

dnl **********************************************************************
dnl * CASC_OPT_DEBUG_CHOICES replaces the obsolete macro
dnl * CASC_CHOOSE_OPT_OR_DEBUG, which still remains for older configure.in
dnl * files which still use it. 
dnl *
dnl * Before this macro is called in configure.in, the macros
dnl * CASC_SET_COPT and CASC_SET_CDEBUG and/or their C++ and Fortran
dnl * counterparts should be invoked to set both optimization and
dnl * debugging flags.  The effect of this macro is to turn off one set of
dnl * flags when the other is selected by the user.  This macro invokes
dnl * the macro AC_ARG_ENABLE to give the configure script the 
dnl * command-line options `--enable-opt' and `--enable-debug'.  When 
dnl * `--enable-opt' appears on the command line, all of the previously
dnl * set debugging flags are turned off, and when `--enable-debug' is
dnl * used, all of the previously set optimization flags are turned off.
dnl * If both flags are used, then none of the previously set flags are
dnl * turned off.   
dnl *
dnl * Also, the variable OPTCHOICE is set to `O' for optimization and `g'
dnl * for debugging.  OPTCHOICE was added because PETSc libraries are
dnl * installed in mirrored directories called 'libO'and `libg'.  So
dnl * OPTCHOICE can be used later on in configure.in or in Makefile.in to
dnl * reference these PETSc libraries, and it also can be used as a simple
dnl * flag to signal whether debugging or optimization has been chosen.
dnl *
dnl * If neither of the options created by this macro are used, the
dnl * default is that configure behaves as if `--enable-opt' was used, and
dnl * OPTCHOICE is set to `O'.  If both options are used, then OPTCHOICE
dnl * is again set to `O'.
dnl **********************************************************************

AC_DEFUN(CASC_OPT_DEBUG_CHOICES,
[
   dnl *  These are the default settings.  They keep these values when
   dnl *  --enable-opt and --enable-debug are both not chosen.
   casc_opt=yes
   casc_debug=no
   casc_both=no

   dnl * when --enable-opt is chosen, casc_opt is assigned "yes", and
   dnl * casc_enable_opt_called is assigned "yes".  Otherwise,
   dnl * casc_enable_opt_called gets "no".
   AC_ARG_ENABLE(opt,
[  --enable-opt            Sets up compiler flags for optimization], 
                casc_opt=yes; casc_enable_opt_called=yes; ,
                casc_enable_opt_called=no )

   dnl * when --enable-debug is chosen, casc_debug is changed to "yes",
   dnl * and then we check whether --enable-opt was also chosen.  If not
   dnl * casc_opt is assigned "no".  Otherwise casc_opt keeps its default
   dnl * value of "yes" and casc_both is assigned "yes".  No action is
   dnl * take if --enable-debug is not invoked, because casc_debug is
   dnl * already "no" by default.
   AC_ARG_ENABLE(debug,
[  --enable-debug          Sets up compiler flags for debugging],
                 casc_debug=yes
                 if test "$casc_enable_opt_called" = "no"; then
                    casc_opt=no
                 else
                    casc_both=yes
                 fi , )

   dnl * If the choice is optimization, then all of the debug variables
   dnl * are given empty values.  If the choice is debugging, then all of
   dnl * the optimization variables are given empty values.  If both are
   dnl * chosen, then the optimization and debug variables are not
   dnl * touched.  OPTCHOICE is set to 'O' for opt and 'g' for debug.  In
   dnl * the both case, OPTCHOICE is given 'O', which was basically an
   dnl * arbitrary decision.
   if test "$casc_both" = "no"; then
      if test "$casc_opt" = "yes"; then
         CDEBUG=
         CXXDEBUG=
         FDEBUG=
         OPTCHOICE=O
      else
         COPT=
         CXXOPT=
         FOPT=  
         OPTCHOICE=g
      fi
   else
      OPTCHOICE=O
   fi

   AC_SUBST(OPTCHOICE)

])dnl


dnl **********************************************************************
dnl * CASC_CHOOSE_OPT_OR_DEBUG
dnl *
dnl * This macro is obsolete.  CASC_OPT_DEBUG_CHOICES provides a better
dnl * framework for optimization/debugging flags.  This macro is kept for
dnl * backward compatibility.
dnl *
dnl * Before this macro is called in configure.in, the macros
dnl * CASC_SET_COPT and CASC_SET_CDEBUG and/or their C++ and Fortran
dnl * counterparts should be invoked to set both optimization and
dnl * debugging flags.  The effect of this macro is to turn off one set of
dnl * flags when the other is selected by the user.  This macro invokes
dnl * the macro AC_ARG_ENABLE to give the configure script the
dnl * command-line option "--enable-opt-debug=ARG", where ARG can equal
dnl * 'opt', 'debug', or 'both'.  If 'opt' then all debugging compiler
dnl * flags are turned off, and if 'debug' then all optimization compiler
dnl * flags are turned off. 
dnl * and 'libg'.  If ARG is 'both', then neither are turned off.  If an
dnl * invalid value of ARG is given, then neither are turned off, and a
dnl * warning message is printed.  If --enable-opt-debug=ARG is not 
dnl * called, then the default action is to enable optimization flags and
dnl * to turn off debugging flags and set OPTCHOICE to 'O'.
dnl **********************************************************************

AC_DEFUN(CASC_CHOOSE_OPT_OR_DEBUG,
[
dnl   if test -z "$casc_opt_or_debug"; then
dnl      casc_opt_or_debug=opt
dnl   fi
AC_ARG_ENABLE(opt-debug,
[  --enable-opt-debug=ARG
         ARG=debug  --  enable debug flags, disable optimization
         ARG=opt    --  enable optimization flags, disable debugging
         ARG=both   --  both optimization and debugging flags enabled],

   casc_opt_or_debug="$enableval",
   casc_opt_or_debug=opt; OPTCHOICE=O
)

   AC_MSG_CHECKING(optimization/debugging choice)
   case $casc_opt_or_debug in

      opt)
         echo "opt"
         CDEBUG=
         CXXDEBUG=
         FDEBUG=
         OPTCHOICE=O
      ;;
      debug)
         echo "debug"
         COPT=
         CXXOPT=
         FOPT=  
         OPTCHOICE=g
      ;;
      both)
         echo "both"
         OPTCHOICE=O
      ;;
      *)
         echo "$casc_opt_or_debug"
            AC_MSG_WARN(
        [Invalid argument given to the flag --enable-opt-debug.
         The only acceptable choices are '--enable-opt-debug=opt',
         '--enable-opt-debug=debug', and '--enable-opt-debug=both'.  Both 
         optimization flags and debugging flags remain unchanged.])
      ;;
   esac

   AC_SUBST(OPTCHOICE)

])dnl
