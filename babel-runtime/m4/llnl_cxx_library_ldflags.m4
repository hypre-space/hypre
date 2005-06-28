
dnl @synopsis LLNL_CXX_LIBRARY_LDFLAGS
dnl
dnl Determine the linker flags (e.g., `-L' and `-l') for the C++ run-time
dnl libaries that are required to successfully link a C++ program or shared
dnl library.  The output variable CXXLIBS is set to these flags.  This macro
dnl is intended for situations for which it is necessary to mix different
dnl languages into a single program or shared library.
dnl
dnl @author Gary Kumfert
AC_DEFUN([LLNL_CXX_LIBRARY_LDFLAGS],
[AC_REQUIRE([AC_PROG_CXX])dnl
AC_LANG_PUSH(C++)dnl
AC_CACHE_CHECK([for C++ libraries],ac_cv_cxx_libs,
[if test "x$CXXLIBS" != "x"; then
  ac_cv_cxx_libs="$CXXLIBS" # Let the user override the test
else 
changequote(, )dnl
echo "int main() { return 0; }" > conftest.C
cxx_output=`${CXX} -v -o conftest conftest.C 2>&1`

cxx_libs=
cxx_flags=


want_arg=
for arg in $cxx_output; do
  old_want_arg=$want_arg
  want_arg=


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
          orig_arg=$arg
          arg=-L`dirname $arg`
          exists=false
          for f in $cxx_flags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            cxx_flags="$cxx_flags $arg"
          fi
          arg=`basename $orig_arg .a`
          arg=`echo $arg | sed 's/^lib/-l/'`
        ;;
        -lang* | -lcrt[012].o)
          arg=
        ;;
        -[lLR])
          want_arg=$arg
          arg=
        ;;
        -[lLR]*)
          exists=false
          for f in $cxx_flags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            cxx_flags="$cxx_flags $arg"
          fi
        ;;
        *)
          arg=
        ;;
      esac
    ;;
    -[lLR])
      arg="$old_want_arg $arg"
    ;;
  esac

  if test -n "$arg"; then
    exists=false
    for f in $cxx_libs; do
      if test x$arg = x$f; then
        exists=true
      fi
    done
    if $exists; then
      arg=
    else
      cxx_libs="$cxx_libs $arg"
    fi
  fi
done

changequote([, ])dnl
ac_cv_cxx_libs="$cxx_libs"
fi #if test "x$CXXLIBS" = "x"
])
CXXLIBS="$ac_cv_cxx_libs"
AC_SUBST(CXXLIBS)
AC_LANG_POP(C++)dnl
])
