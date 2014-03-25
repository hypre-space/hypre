

dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_check_classpath.m4***
dnl @synopsis AC_CHECK_CLASSPATH
dnl
dnl AC_CHECK_CLASSPATH just displays the CLASSPATH, for the edification
dnl of the user.
dnl
dnl Note: This is part of the set of autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download the whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl The general documentation, as well as the sample configure.in, is
dnl included in the AC_PROG_JAVA macro.
dnl
dnl @author Stephane Bortzmeyer <bortzmeyer@pasteur.fr>
dnl @version $Id$
dnl
AC_DEFUN([LLNL_CHECK_CLASSPATH],[
AC_CACHE_CHECK([for your CLASSPATH],
	llnl_cv_java_classpath,
[if test "x$CLASSPATH" = x; then
        llnl_cv_java_classpath="none"
else
        llnl_cv_java_classpath="$CLASSPATH"
fi
])])


