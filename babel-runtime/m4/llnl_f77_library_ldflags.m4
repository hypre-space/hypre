

# ---------------------------------------- #
# 4d. Fortran 77 compiler characteristics. #
# ---------------------------------------- #

#
# NOTE: LLNL_F77_LIBRARY_LDFLAGS, _LLNL_PROG_F77_V_OUTPUT, and _LLNL_PROG_F77_V
#       are identical to their AC_* cousins except that _LLNL_PROG_F77_V
#       calls AC_LINK_IFELSE instead of AC_COMPILE_IFELSE
#

# _LLNL_PROG_F77_V_OUTPUT([FLAG = $ac_cv_prog_f77_v])
# -------------------------------------------------
# Link a trivial Fortran program, compiling with a verbose output FLAG
# (which default value, $ac_cv_prog_f77_v, is computed by
# _AC_PROG_F77_V), and return the output in $ac_f77_v_output.  This
# output is processed in the way expected by AC_F77_LIBRARY_LDFLAGS,
# so that any link flags that are echoed by the compiler appear as
# space-separated items.
AC_DEFUN([_LLNL_PROG_F77_V_OUTPUT],
[AC_REQUIRE([AC_PROG_F77])dnl
AC_LANG_PUSH(Fortran 77)dnl

AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])

# Compile and link our simple test program by passing a flag (argument
# 1 to this macro) to the Fortran 77 compiler in order to get
# "verbose" output that we can then parse for the Fortran 77 linker
# flags.
ac_save_FFLAGS=$FFLAGS
FFLAGS="$FFLAGS m4_default([$1], [$ac_cv_prog_f77_v])"
(eval echo $as_me:__oline__: \"$ac_link\") >&AS_MESSAGE_LOG_FD
ac_f77_v_output=`eval $ac_link AS_MESSAGE_LOG_FD>&1 2>&1 | grep -v 'Driving:'`
echo "$ac_f77_v_output" >&AS_MESSAGE_LOG_FD
FFLAGS=$ac_save_FFLAGS

rm -f conftest*
AC_LANG_POP(Fortran 77)dnl

# If we are using xlf then replace all the commas with spaces.
if echo $ac_f77_v_output | grep xlfentry >/dev/null 2>&1; then
  ac_f77_v_output=`echo $ac_f77_v_output | sed 's/,/ /g'`
fi

# On HP/UX there is a line like: "LPATH is: /foo:/bar:/baz" where
# /foo, /bar, and /baz are search directories for the Fortran linker.
# Here, we change these into -L/foo -L/bar -L/baz (and put it first):
ac_f77_v_output="`echo $ac_f77_v_output |
	grep 'LPATH is:' |
	sed 's,.*LPATH is\(: *[[^ ]]*\).*,\1,;s,: */, -L/,g'` $ac_f77_v_output"

case $ac_f77_v_output in
  # If we are using xlf then replace all the commas with spaces.
  *xlfentry*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed 's/,/ /g'` ;;

  # With Intel ifc, ignore the quoted -mGLOB_options_string stuff (quoted
  # $LIBS confuse us, and the libraries appear later in the output anyway).
  *mGLOB_options_string*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed 's/\"-mGLOB[[^\"]]*\"/ /g'` ;;

  # If we are using Cray Fortran then delete quotes.
  # Use "\"" instead of '"' for font-lock-mode.
  # FIXME: a more general fix for quoted arguments with spaces?
  *cft90*)
    ac_f77_v_output=`echo $ac_f77_v_output | sed "s/\"//g"` ;;
esac
])# _LLNL_PROG_F77_V_OUTPUT


# _LLNL_PROG_F77_V
# --------------
#
# Determine the flag that causes the Fortran 77 compiler to print
# information of library and object files (normally -v)
# Needed for AC_F77_LIBRARY_FLAGS
# Some compilers don't accept -v (Lahey: -verbose, xlf: -V, Fujitsu: -###)
AC_DEFUN([_LLNL_PROG_F77_V],
[AC_CACHE_CHECK([how to get verbose linking output from $F77],
                [ac_cv_prog_f77_v],
[AC_LANG_ASSERT(Fortran 77)
AC_LINK_IFELSE([AC_LANG_PROGRAM()],
[ac_cv_prog_f77_v=
# Try some options frequently used verbose output
# It is better to try -V before -v for xlf
for ac_verb in -V -v -verbose --verbose -\#\#\#; do
  _LLNL_PROG_F77_V_OUTPUT($ac_verb)
  # look for -l* and *.a constructs in the output
  for ac_arg in $ac_f77_v_output; do
     case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]*)
          ac_cv_prog_f77_v=$ac_verb
          break 2 ;;
     esac
  done
done
if test -z "$ac_cv_prog_f77_v"; then
   AC_MSG_WARN([cannot determine how to obtain linking information from $F77])
fi],
                  [AC_MSG_WARN([compilation failed])])
])])# _LLNL_PROG_F77_V


# LLNL_F77_LIBRARY_LDFLAGS
# ----------------------
#
# Determine the linker flags (e.g. "-L" and "-l") for the Fortran 77
# intrinsic and run-time libraries that are required to successfully
# link a Fortran 77 program or shared library.  The output variable
# FLIBS is set to these flags.
#
# This macro is intended to be used in those situations when it is
# necessary to mix, e.g. C++ and Fortran 77, source code into a single
# program or shared library.
#
# For example, if object files from a C++ and Fortran 77 compiler must
# be linked together, then the C++ compiler/linker must be used for
# linking (since special C++-ish things need to happen at link time
# like calling global constructors, instantiating templates, enabling
# exception support, etc.).
#
# However, the Fortran 77 intrinsic and run-time libraries must be
# linked in as well, but the C++ compiler/linker doesn't know how to
# add these Fortran 77 libraries.  Hence, the macro
# "AC_F77_LIBRARY_LDFLAGS" was created to determine these Fortran 77
# libraries.
#
# This macro was packaged in its current form by Matthew D. Langston.
# However, nearly all of this macro came from the "OCTAVE_FLIBS" macro
# in "octave-2.0.13/aclocal.m4", and full credit should go to John
# W. Eaton for writing this extremely useful macro.  Thank you John.
AC_DEFUN([LLNL_F77_LIBRARY_LDFLAGS],
[AC_LANG_PUSH(Fortran 77)dnl
_LLNL_PROG_F77_V
AC_CACHE_CHECK([for Fortran 77 libraries], ac_cv_flibs,
[if test "x$FLIBS" != "x"; then
  ac_cv_flibs="$FLIBS" # Let the user override the test.
else

_LLNL_PROG_F77_V_OUTPUT

ac_cv_flibs=

# Save positional arguments (if any)
ac_save_positional="$[@]"

set X $ac_f77_v_output
while test $[@%:@] != 1; do
  shift
  ac_arg=$[1]
  case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_flibs, ,
              ac_cv_flibs="$ac_cv_flibs $ac_arg")
          ;;
        -bI:*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_flibs, ,
             [_AC_LINKER_OPTION([$ac_arg], ac_cv_flibs)])
          ;;
          # Ignore these flags.
        -lang* | -lcrt[[012]].o | -lcrtbegin.o | -lc | -lgcc* | -libmil | -LANG:=*)
          ;;
	-lfrtbegin )  ;; #(gkk) Ignore this one too
        -lkernel32)
          test x"$CYGWIN" != xyes && ac_cv_flibs="$ac_cv_flibs $ac_arg"
          ;;
        -[[LRuY]])
          # These flags, when seen by themselves, take an argument.
          # We remove the space between option and argument and re-iterate
          # unless we find an empty arg or a new option (starting with -)
	  case $[2] in
             "" | -*);;
             *)
		ac_arg="$ac_arg$[2]"
		shift; shift
		set X $ac_arg "$[@]"
		;;
	  esac
          ;;
        -YP,*)
          for ac_j in `echo $ac_arg | sed -e 's/-YP,/-L/;s/:/ -L/g'`; do
            _AC_LIST_MEMBER_IF($ac_j, $ac_cv_flibs, ,
                               [ac_arg="$ac_arg $ac_j"
                               ac_cv_flibs="$ac_cv_flibs $ac_j"])
          done
          ;;
        -[[lLR]]*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_flibs, ,
                             ac_cv_flibs="$ac_cv_flibs $ac_arg")
          ;;
          # Ignore everything else.
  esac
done
# restore positional arguments
set X $ac_save_positional; shift

# We only consider "LD_RUN_PATH" on Solaris systems.  If this is seen,
# then we insist that the "run path" must be an absolute path (i.e. it
# must begin with a "/").
case `(uname -sr) 2>/dev/null` in
   "SunOS 5"*)
      ac_ld_run_path=`echo $ac_f77_v_output |
                        sed -n 's,^.*LD_RUN_PATH *= *\(/[[^ ]]*\).*$,-R\1,p'`
      test "x$ac_ld_run_path" != x &&
        _AC_LINKER_OPTION([$ac_ld_run_path], ac_cv_flibs)
      ;;
   "Darwin 7"*)
      if test -n "$ac_cv_flibs"; then
	for ac_arg in $ac_cv_flibs; do
	  case $ac_arg in
	  -L*)
	    tmp_path="$tmp_path "`echo $ECHO_N $ac_arg | sed -e 's/^-L//'`
	    ;;
	  -lSystem) ;; # ignore this one
	  -lm)
	    modified_flibs="$modified_flibs $ac_arg"
	    ;;
	  -l*)
	    found="no"
	    if test -n "$tmp_path"; then
	      libname=`echo $ECHO_N $ac_arg | sed -e 's/^-l//'`
	      for tp in $tmp_path; do
		if test $found = "no"; then
		  if test -d $tp -a -r $tp; then
		    shortpath=`cd $tp 2>/dev/null && pwd`
		  else
		    shortpath=$tp
		  fi
		  if test -r "$shortpath/lib$libname.a" ; then
		    modified_flibs="$modified_flibs $shortpath/lib$libname.a"
		    found="yes"
		  elif test -r "$shortpath/lib$libname.so" ; then
		    modified_flibs="$modified_flibs $shortpath/lib$libname.so"
		    found="yes"
		  elif test -r "$shortpath/lib$libname.dylib" ; then
		    modified_flibs="$modified_flibs $shortpath/lib$libname.dylib"
		    found="yes"
		  fi
		fi
	      done
	    fi
	    if test $found = "no"; then
	      modified_flibs="$modified_flibs $ac_arg"
	    fi
	    ;;
	  esac
	done
	ac_cv_flibs="$modified_flibs"
      fi
      ;;
esac
fi # test "x$FLIBS" = "x"
])
FLIBS="$ac_cv_flibs"
AC_SUBST(FLIBS)
AC_LANG_POP(Fortran 77)dnl
])# LLNL_F77_LIBRARY_LDFLAGS

