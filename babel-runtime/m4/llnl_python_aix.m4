dnl @synopsis LLNL_XML_EXTRA
dnl
dnl @author Tom Epperly
dnl
dnl When linking Python C extensions on AIX, one must include the 
dnl library dependencies from libxml2.
AC_DEFUN([LLNL_PYTHON_AIX],[
AC_REQUIRE([AC_CANONICAL_HOST])dnl
AC_REQUIRE([LLNL_LIBXML_CONFIG]) dnl
AC_MSG_CHECKING([for extra python setup arguments])
PYTHON_SETUP_ARGS=""
if test "$XML2_CONFIG" != "no"; then
  case $host_os in
  aix*)
    libxml_libs=`$XML2_CONFIG --libs`
    for f in $libxml_libs; do
      case $f in
      -L*)
	libxml_dir=`echo "$f" | sed -e 's/^-L//'` 
	PYTHON_SETUP_ARGS="$PYTHON_SETUP_ARGS --library-dirs=$libxml_dir"
	;;
      -l*)
	libxml_lib=`echo "$f" | sed -e 's/^-l//'`
	PYTHON_SETUP_ARGS="$PYTHON_SETUP_ARGS --extra-library=$libxml_lib"
	;;
      esac
    done
    ;;
  esac
fi
if test -z "$PYTHON_SETUP_ARGS" ; then
  AC_MSG_RESULT([none])
else
  AC_MSG_RESULT($PYTHON_SETUP_ARGS)
fi
AC_SUBST(PYTHON_SETUP_ARGS)
])