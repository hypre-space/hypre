

dnl @synopsis LLNL_PROG_JAVA
dnl
dnl Here is a summary of the main macros:
dnl
dnl AC_PROG_JAVAC: finds a Java compiler.
dnl
dnl LLNL_PROG_JAVA: finds a Java virtual machine.
dnl
dnl AC_CHECK_CLASS: finds if we have the given class (beware of CLASSPATH!).
dnl
dnl AC_CHECK_RQRD_CLASS: finds if we have the given class and stops otherwise.
dnl
dnl AC_TRY_COMPILE_JAVA: attempt to compile user given source.
dnl
dnl AC_TRY_RUN_JAVA: attempt to compile and run user given source.
dnl
dnl AC_JAVA_OPTIONS: adds Java configure options.
dnl
dnl LLNL_PROG_JAVA tests an existing Java virtual machine. It uses the
dnl environment variable JAVA then tests in sequence various common Java
dnl virtual machines. For political reasons, it starts with the free ones.
dnl You *must* call [AC_PROG_JAVAC] before.
dnl
dnl If you want to force a specific VM:
dnl
dnl - at the configure.in level, set JAVA=yourvm before calling LLNL_PROG_JAVA
dnl   (but after AC_INIT)
dnl
dnl - at the configure level, setenv JAVA
dnl
dnl You can use the JAVA variable in your Makefile.in, with @JAVA@.
dnl
dnl *Warning*: its success or failure can depend on a proper setting of the
dnl CLASSPATH env. variable.
dnl
dnl TODO: allow to exclude virtual machines (rationale: most Java programs
dnl cannot run with some VM like kaffe).
dnl
dnl Note: This is part of the set of autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download the whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl
dnl A Web page, with a link to the latest CVS snapshot is at
dnl <http://www.internatif.org/bortzmeyer/autoconf-Java/>.
dnl
dnl This is a sample configure.in
dnl Process this file with autoconf to produce a configure script.
dnl
dnl    AC_INIT(UnTag.java)
dnl
dnl    dnl Checks for programs.
dnl    AC_CHECK_CLASSPATH
dnl    AC_PROG_JAVAC
dnl    LLNL_PROG_JAVA
dnl
dnl    dnl Checks for classes
dnl    AC_CHECK_RQRD_CLASS(org.xml.sax.Parser)
dnl    AC_CHECK_RQRD_CLASS(com.jclark.xml.sax.Driver)
dnl
dnl    AC_OUTPUT(Makefile)
dnl
dnl @author Stephane Bortzmeyer <bortzmeyer@pasteur.fr>
dnl @version $Id$
dnl
AC_DEFUN([LLNL_PROG_JAVA],[
AC_REQUIRE([AC_EXEEXT])dnl
if test x$JAVAPREFIX = x; then
        test x$JAVA = x && AC_CHECK_PROGS(JAVA, kaffe$EXEEXT java$EXEEXT)
else
        test x$JAVA = x && AC_CHECK_PROGS(JAVA, kaffe$EXEEXT java$EXEEXT, $JAVAPREFIX)
fi
test x$JAVA = x && AC_MSG_ERROR([no acceptable Java virtual machine found in \$PATH])
LLNL_PROG_JAVA_WORKS
AC_PROVIDE([$0])dnl
])




dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_java_works.m4***
dnl @synopsis LLNL_PROG_JAVA_WORKS
dnl
dnl Internal use ONLY.
dnl
dnl Note: This is part of the set of autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download the whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl The general documentation, as well as the sample configure.in, is
dnl included in the LLNL_PROG_JAVA macro.
dnl
dnl @author Stephane Bortzmeyer <bortzmeyer@pasteur.fr>
dnl @version $Id$
dnl 
dnl Additional Mods by Gary Kumfert
dnl
AC_DEFUN([LLNL_PROG_JAVA_WORKS], [
AC_REQUIRE([AC_PROG_JAVAC])
AC_CACHE_CHECK(if $JAVA works, ac_cv_prog_java_works, [
JAVA_TEST=Test.java
CLASS_TEST=Test.class
TEST=Test
changequote(, )dnl
cat << \EOF > $JAVA_TEST
/* [#]line __oline__ "configure" */
public class Test {
public static void main (String args[]) {
        System.exit (0);
} }
EOF
changequote([, ])dnl
if test -n "$JAVAPREFIX"; then
  myjavac=$JAVAPREFIX/$JAVAC
  myjava=$JAVAPREFIX/$JAVA
else
  myjavac=$JAVAC
  myjava=$JAVA
fi 
if AC_TRY_COMMAND($myjavac $JAVACFLAGS $JAVA_TEST) && test -s $CLASS_TEST; then
        :
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat $JAVA_TEST >&AC_FD_CC
  AC_MSG_ERROR(The Java compiler $JAVAC failed (see config.log, check the CLASSPATH?))
fi
if AC_TRY_COMMAND($myjava $JAVAFLAGS $TEST) >/dev/null 2>&1; then
  ac_cv_prog_java_works=yes
else
  ac_cv_prog_java_works=no
  echo "configure: failed program was:" >&AC_FD_CC
  cat $JAVA_TEST >&AC_FD_CC
  AC_MSG_ERROR(The Java VM $JAVA failed (see config.log, check the CLASSPATH?))
fi
rm -fr $JAVA_TEST $CLASS_TEST Test.uue
])
AC_PROVIDE([$0])dnl
]
)


