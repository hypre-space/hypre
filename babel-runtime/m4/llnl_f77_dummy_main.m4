
# LLNL_F77_DUMMY_MAIN([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -----------------------------------------------------------
#
# Detect name of dummy main routine required by the Fortran libraries,
# (if any) and define F77_DUMMY_MAIN to this name (which should be
# used for a dummy declaration, if it is defined).  On some systems,
# linking a C program to the Fortran library does not work unless you
# supply a dummy function called something like MAIN__.
#
# Execute ACTION-IF-NOT-FOUND if no way of successfully linking a C
# program with the F77 libs is found; default to exiting with an error
# message.  Execute ACTION-IF-FOUND if a dummy routine name is needed
# and found or if it is not needed (default to defining F77_DUMMY_MAIN
# when needed).
#
# What is technically happening is that the Fortran libraries provide
# their own main() function, which usually initializes Fortran I/O and
# similar stuff, and then calls MAIN__, which is the entry point of
# your program.  Usually, a C program will override this with its own
# main() routine, but the linker sometimes complain if you don't
# provide a dummy (never-called) MAIN__ routine anyway.
#
# Of course, programs that want to allow Fortran subroutines to do
# I/O, etcetera, should call their main routine MAIN__() (or whatever)
# instead of main().  A separate autoconf test (LLNL_F77_MAIN) checks
# for the routine to use in this case (since the semantics of the test
# are slightly different).  To link to e.g. purely numerical
# libraries, this is normally not necessary, however, and most C/C++
# programs are reluctant to turn over so much control to Fortran.  =)
#
# The name variants we check for are (in order):
#   MAIN__ (g77, MAIN__ required on some systems; IRIX, MAIN__ optional)
#   MAIN_, __main (SunOS)
#   MAIN _MAIN __MAIN main_ main__ _main (we follow DDD and try these too)
AC_DEFUN([LLNL_F77_DUMMY_MAIN],
[AC_REQUIRE([LLNL_F77_LIBRARY_LDFLAGS])dnl
m4_define([_AC_LANG_PROGRAM_C_F77_HOOKS],
[#ifdef F77_DUMMY_MAIN
#  ifdef __cplusplus
     extern "C"
#  endif
   int F77_DUMMY_MAIN() { return 1; }
#endif
])
AC_CACHE_CHECK([for dummy main to link with Fortran 77 libraries],
               ac_cv_f77_dummy_main,
[AC_LANG_PUSH(C)dnl
 ac_f77_dm_save_LIBS=$LIBS
 LIBS="$LIBS $FLIBS"

 # First, try linking without a dummy main:
 AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                [ac_cv_f77_dummy_main=none],
                [ac_cv_f77_dummy_main=unknown])

 if test $ac_cv_f77_dummy_main = unknown; then
   for ac_func in MAIN__ MAIN_ __main MAIN _MAIN __MAIN main_ main__ _main; do
     AC_LINK_IFELSE([AC_LANG_PROGRAM([[@%:@define F77_DUMMY_MAIN $ac_func]])],
                    [ac_cv_f77_dummy_main=$ac_func; break])
   done
 fi
 rm -f conftest*
 LIBS=$ac_f77_dm_save_LIBS
 AC_LANG_POP(C)dnl
])
F77_DUMMY_MAIN=$ac_cv_f77_dummy_main
AS_IF([test "$F77_DUMMY_MAIN" != unknown],
      [m4_default([$1],
[if test $F77_DUMMY_MAIN != none; then
  AC_DEFINE_UNQUOTED([F77_DUMMY_MAIN], $F77_DUMMY_MAIN,
                     [Define to dummy `main' function (if any) required to
                      link to the Fortran 77 libraries.])
fi])],
      [m4_default([$2],
            [AC_MSG_FAILURE([linking to Fortran libraries from C fails])])])
])# LLNL_F77_DUMMY_MAIN

# LLNL_F77_MAIN
# -----------
# Define F77_MAIN to name of alternate main() function for use with
# the Fortran libraries.  (Typically, the libraries may define their
# own main() to initialize I/O, etcetera, that then call your own
# routine called MAIN__ or whatever.)  See LLNL_F77_DUMMY_MAIN, above.
# If no such alternate name is found, just define F77_MAIN to main.
#
AC_DEFUN([LLNL_F77_MAIN],
[AC_REQUIRE([LLNL_F77_LIBRARY_LDFLAGS])dnl
AC_CACHE_CHECK([for alternate main to link with Fortran 77 libraries],
               ac_cv_f77_main,
[AC_LANG_PUSH(C)dnl
 ac_f77_m_save_LIBS=$LIBS
 LIBS="$LIBS $FLIBS"
 ac_cv_f77_main="main" # default entry point name

 for ac_func in MAIN__ MAIN_ __main MAIN _MAIN __MAIN main_ main__ _main; do
   AC_LINK_IFELSE([AC_LANG_PROGRAM([@%:@undef F77_DUMMY_MAIN
@%:@define main $ac_func])],
                  [ac_cv_f77_main=$ac_func; break])
 done
 rm -f conftest*
 LIBS=$ac_f77_m_save_LIBS
 AC_LANG_POP(C)dnl
])
AC_DEFINE_UNQUOTED([F77_MAIN], $ac_cv_f77_main,
                   [Define to alternate name for `main' routine that is
                    called from a `main' in the Fortran libraries.])
])# LLNL_F77_MAIN

