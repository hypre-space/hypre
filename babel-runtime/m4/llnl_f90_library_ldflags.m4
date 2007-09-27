# copied wholesale from Autoconf 2.59's AC_FC_LIBRARY_LDFLAGS macro.
# with one little change for Absoft needing the -lU77 flags


# _LLNL_F90_LIBRARY_LDFLAGS
# ----------------------
#
# Determine the linker flags (e.g. "-L" and "-l") for the Fortran
# intrinsic and run-time libraries that are required to successfully
# link a Fortran program or shared library.  The output variable
# FLIBS/FCLIBS is set to these flags.
#
# This macro is intended to be used in those situations when it is
# necessary to mix, e.g. C++ and Fortran, source code into a single
# program or shared library.
#
# For example, if object files from a C++ and Fortran compiler must
# be linked together, then the C++ compiler/linker must be used for
# linking (since special C++-ish things need to happen at link time
# like calling global constructors, instantiating templates, enabling
# exception support, etc.).
#
# However, the Fortran intrinsic and run-time libraries must be
# linked in as well, but the C++ compiler/linker doesn't know how to
# add these Fortran libraries.  Hence, the macro
# "AC_F77_LIBRARY_LDFLAGS" was created to determine these Fortran
# libraries.
#
# This macro was packaged in its current form by Matthew D. Langston.
# However, nearly all of this macro came from the "OCTAVE_FLIBS" macro
# in "octave-2.0.13/aclocal.m4", and full credit should go to John
# W. Eaton for writing this extremely useful macro.  Thank you John.
AC_DEFUN([_LLNL_F90_LIBRARY_LDFLAGS],
[_AC_FORTRAN_ASSERT()dnl
_AC_PROG_FC_V
[]_AC_LANG_PREFIX[]LIBS_NOSORT=true
AC_CACHE_CHECK([for Fortran libraries of $[]_AC_FC[] (LLNL)], ac_cv_[]_AC_LANG_ABBREV[]_libs,
[if test "x$[]_AC_LANG_PREFIX[]LIBS" != "x"; then
  ac_cv_[]_AC_LANG_ABBREV[]_libs="$[]_AC_LANG_PREFIX[]LIBS" # Let the user override the test.
  []_AC_LANG_PREFIX[]LIBS_NOSORT=true
  _AS_ECHO_N([(user override) ])
else
  []_AC_LANG_PREFIX[]LIBS_NOSORT=false

_AC_PROG_FC_V_OUTPUT

ac_cv_[]_AC_LANG_ABBREV[]_libs=

# Save positional arguments (if any)
ac_save_positional="$[@]"

set X $ac_[]_AC_LANG_ABBREV[]_v_output
while test $[@%:@] != 1; do
  shift
  ac_arg=$[1]
  case $ac_arg in
	*libgcc.a | *libgcc_s.a)
	  ;;
        [[\\/]]*.a | ?:[[\\/]]*.a)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_[]_AC_LANG_ABBREV[]_libs, ,
              ac_cv_[]_AC_LANG_ABBREV[]_libs="$ac_cv_[]_AC_LANG_ABBREV[]_libs $ac_arg")
          ;;
        -bI:*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_[]_AC_LANG_ABBREV[]_libs, ,
             [_AC_LINKER_OPTION([$ac_arg], ac_cv_[]_AC_LANG_ABBREV[]_libs)])
          ;;
          # Ignore these flags.
        -lang* | -lcrt[[012]].o | -lcrtbegin.o | -lc | -lgcc* | -libmil | -LANG:=* | -lgfortranbegin )
          ;;
        -lkernel32)
          test x"$CYGWIN" != xyes && ac_cv_[]_AC_LANG_ABBREV[]_libs="$ac_cv_[]_AC_LANG_ABBREV[]_libs $ac_arg"
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
            _AC_LIST_MEMBER_IF($ac_j, $ac_cv_[]_AC_LANG_ABBREV[]_libs, ,
                               [ac_arg="$ac_arg $ac_j"
                               ac_cv_[]_AC_LANG_ABBREV[]_libs="$ac_cv_[]_AC_LANG_ABBREV[]_libs $ac_j"])
          done
          ;;
	-lcxa | -lunwind)
	  case $FC in
	  *ifort* | *ifc*)  ;;
	  *)
            _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_[]_AC_LANG_ABBREV[]_libs, ,
                               ac_cv_[]_AC_LANG_ABBREV[]_libs="$ac_cv_[]_AC_LANG_ABBREV[]_libs $ac_arg")
	  ;;
	  esac
	  ;;
        -[[lLR]]*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_[]_AC_LANG_ABBREV[]_libs, ,
                             ac_cv_[]_AC_LANG_ABBREV[]_libs="$ac_cv_[]_AC_LANG_ABBREV[]_libs $ac_arg")
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
      ac_ld_run_path=`echo $ac_[]_AC_LANG_ABBREV[]_v_output |
                        sed -n 's,^.*LD_RUN_PATH *= *\(/[[^ ]]*\).*$,-R\1,p'`
      test "x$ac_ld_run_path" != x &&
        _AC_LINKER_OPTION([$ac_ld_run_path], ac_cv_[]_AC_LANG_ABBREV[]_libs)
      # add -mimpure-text
      ac_cv_[]_AC_LANG_ABBREV[]_libs="-mimpure-text $ac_cv_[]_AC_LANG_ABBREV[]_libs"
      ;;
esac

fi # test "x$[]_AC_LANG_PREFIX[]LIBS" = "x"

dnl echo "*****"
dnl echo "*****  testing for libU77.a in $ac_cv_[]_AC_LANG_ABBREV[]_libs"
dnl echo "*****"
add_u77=
for arg in $ac_cv_[]_AC_LANG_ABBREV[]_libs; do
dnl echo "****"
dnl echo "**** $arg"
dnl echo "****"
case "$arg" in 
   -[[LR]]*)
	dir=`echo $arg | sed 's/^-L//;s/^-R//;'`
	dnl echo "*****"
	dnl echo "***** testing for $dir/libU77.a"
	dnl echo "*****"
	if test -e $dir/libU77.a; then
		dnl echo "********"
		dnl echo "******** YES"
		dnl echo "********"
 	     add_u77="-lU77"
_AC_LIST_MEMBER_IF($add_u77, $ac_cv_[]_AC_LANG_ABBREV[]_libs, ,
              ac_cv_[]_AC_LANG_ABBREV[]_libs="$add_u77 $ac_cv_[]_AC_LANG_ABBREV[]_libs")
        fi
	;;
esac
done

])
[]_AC_LANG_PREFIX[]LIBS="$ac_cv_[]_AC_LANG_ABBREV[]_libs"
AC_SUBST([]_AC_LANG_PREFIX[]LIBS)
])# _LLNL_F90_LIBRARY_LDFLAGS

# LLNL_F90_LIBRARY_LDFLAGS
# ----------------------
AC_DEFUN([LLNL_F90_LIBRARY_LDFLAGS],
[dnl AC_REQUIRE([AC_PROG_FC])dnl
AC_LANG_PUSH(Fortran)dnl
AC_ARG_VAR([FCLIBS],[Linker flags needed to link against F90 code])
_LLNL_F90_LIBRARY_LDFLAGS
AC_LANG_POP(Fortran)dnl
])# LLNL_F90_LIBRARY_LDFLAGS

