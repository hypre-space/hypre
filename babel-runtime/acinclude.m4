

dnl *** file: config/autoconf-archive-macros/acx_restrict.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C_Support/acx_restrict.m4***
dnl @synopsis ACX_C_RESTRICT
dnl
dnl This macro determines whether the C compiler supports the "restrict"
dnl keyword introduced in ANSI C99, or an equivalent.  Does nothing if
dnl the compiler accepts the keyword.  Otherwise, if the compiler supports
dnl an equivalent (like gcc's __restrict__) defines "restrict" to be that.
dnl Otherwise, defines "restrict" to be empty.
dnl
dnl @version $Id$
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>

AC_DEFUN([ACX_C_RESTRICT],
[AC_CACHE_CHECK([for C restrict keyword], acx_cv_c_restrict,
[acx_cv_c_restrict=unsupported
 AC_LANG_SAVE
 AC_LANG_C
 # Try the official restrict keyword, then gcc's __restrict__, then
 # SGI's __restrict.  __restrict has slightly different semantics than
 # restrict (it's a bit stronger, in that __restrict pointers can't
 # overlap even with non __restrict pointers), but I think it should be
 # okay under the circumstances where restrict is normally used.
 for acx_kw in restrict __restrict__ __restrict; do
   AC_TRY_COMPILE([], [float * $acx_kw x;], [acx_cv_c_restrict=$acx_kw; break])
 done
 AC_LANG_RESTORE
])
 if test "$acx_cv_c_restrict" != "restrict"; then
   acx_kw="$acx_cv_c_restrict"
   if test "$acx_kw" = unsupported; then acx_kw=""; fi
   AC_DEFINE_UNQUOTED(restrict, $acx_kw, [Define to equivalent of C99 restrict keyword, or to nothing if this is not supported.  Do not define if restrict is supported directly.])
 fi
])




dnl *** file: config/autoconf-archive-macros/ac_java_options.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_java_options.m4***
dnl @synopsis AC_JAVA_OPTIONS
dnl
dnl AC_JAVA_OPTIONS adds configure command line options used for Java m4
dnl macros. This Macro is optional.
dnl
dnl Note: This is part of the set of autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download the whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl The general documentation, as well as the sample configure.in, is
dnl included in the AC_PROG_JAVA macro.
dnl
dnl @author Devin Weaver <ktohg@tritarget.com>
dnl @version $Id$
dnl
AC_DEFUN([AC_JAVA_OPTIONS],[
AC_ARG_WITH(java-prefix,
                        [  --with-java-prefix=PFX  prefix where Java runtime is installed (optional)])
AC_ARG_WITH(javac-flags,
                        [  --with-javac-flags=FLAGS flags to pass to the Java compiler (optional)])
AC_ARG_WITH(java-flags,
                        [  --with-java-flags=FLAGS flags to pass to the Java VM (optional)])
JAVAPREFIX=$with_java_prefix
JAVACFLAGS=$with_javac_flags
JAVAFLAGS=$with_java_flags
AC_SUBST(JAVAPREFIX)dnl
AC_SUBST(JAVACFLAGS)dnl
AC_SUBST(JAVAFLAGS)dnl
AC_SUBST(JAVA)dnl
AC_SUBST(JAVAC)dnl
])




dnl *** file: config/autoconf-archive-macros/ac_prog_java.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_java.m4***
dnl @synopsis AC_PROG_JAVA
dnl
dnl Here is a summary of the main macros:
dnl
dnl AC_PROG_JAVAC: finds a Java compiler.
dnl
dnl AC_PROG_JAVA: finds a Java virtual machine.
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
dnl AC_PROG_JAVA tests an existing Java virtual machine. It uses the
dnl environment variable JAVA then tests in sequence various common Java
dnl virtual machines. For political reasons, it starts with the free ones.
dnl You *must* call [AC_PROG_JAVAC] before.
dnl
dnl If you want to force a specific VM:
dnl
dnl - at the configure.in level, set JAVA=yourvm before calling AC_PROG_JAVA
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
dnl    AC_PROG_JAVA
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
AC_DEFUN([AC_PROG_JAVA],[
AC_REQUIRE([AC_EXEEXT])dnl
if test x$JAVAPREFIX = x; then
        test x$JAVA = x && AC_CHECK_PROGS(JAVA, kaffe$EXEEXT java$EXEEXT)
else
        test x$JAVA = x && AC_CHECK_PROGS(JAVA, kaffe$EXEEXT java$EXEEXT, $JAVAPREFIX)
fi
test x$JAVA = x && AC_MSG_ERROR([no acceptable Java virtual machine found in \$PATH])
AC_PROG_JAVA_WORKS
AC_PROVIDE([$0])dnl
])




dnl *** file: config/autoconf-archive-macros/ac_prog_java_works.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_java_works.m4***
dnl @synopsis AC_PROG_JAVA_WORKS
dnl
dnl Internal use ONLY.
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
AC_DEFUN([AC_PROG_JAVA_WORKS], [
AC_CHECK_PROG(uudecode, uudecode$EXEEXT, yes)
if test x$uudecode = xyes; then
AC_CACHE_CHECK([if uudecode can decode base 64 file], ac_cv_prog_uudecode_base64, [
dnl /**
dnl  * Test.java: used to test if java compiler works.
dnl  */
dnl public class Test
dnl {
dnl
dnl public static void
dnl main( String[] argv )
dnl {
dnl     System.exit (0);
dnl }
dnl
dnl }
cat << \EOF > Test.uue
begin-base64 644 Test.class
yv66vgADAC0AFQcAAgEABFRlc3QHAAQBABBqYXZhL2xhbmcvT2JqZWN0AQAE
bWFpbgEAFihbTGphdmEvbGFuZy9TdHJpbmc7KVYBAARDb2RlAQAPTGluZU51
bWJlclRhYmxlDAAKAAsBAARleGl0AQAEKEkpVgoADQAJBwAOAQAQamF2YS9s
YW5nL1N5c3RlbQEABjxpbml0PgEAAygpVgwADwAQCgADABEBAApTb3VyY2VG
aWxlAQAJVGVzdC5qYXZhACEAAQADAAAAAAACAAkABQAGAAEABwAAACEAAQAB
AAAABQO4AAyxAAAAAQAIAAAACgACAAAACgAEAAsAAQAPABAAAQAHAAAAIQAB
AAEAAAAFKrcAErEAAAABAAgAAAAKAAIAAAAEAAQABAABABMAAAACABQ=
====
EOF
if uudecode$EXEEXT Test.uue; then
        ac_cv_prog_uudecode_base64=yes
else
        echo "configure: __oline__: uudecode had trouble decoding base 64 file 'Test.uue'" >&AC_FD_CC
        echo "configure: failed file was:" >&AC_FD_CC
        cat Test.uue >&AC_FD_CC
        ac_cv_prog_uudecode_base64=no
fi
rm -f Test.uue])
fi
if test x$ac_cv_prog_uudecode_base64 != xyes; then
        rm -f Test.class
        AC_MSG_WARN([I have to compile Test.class from scratch])
        if test x$ac_cv_prog_javac_works = xno; then
                AC_MSG_ERROR([Cannot compile java source. $JAVAC does not work properly])
        fi
        if test x$ac_cv_prog_javac_works = x; then
                AC_PROG_JAVAC
        fi
fi
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
if test x$ac_cv_prog_uudecode_base64 != xyes; then
        if AC_TRY_COMMAND($JAVAC $JAVACFLAGS $JAVA_TEST) && test -s $CLASS_TEST; then
                :
        else
          echo "configure: failed program was:" >&AC_FD_CC
          cat $JAVA_TEST >&AC_FD_CC
          AC_MSG_ERROR(The Java compiler $JAVAC failed (see config.log, check the CLASSPATH?))
        fi
fi
if AC_TRY_COMMAND($JAVA $JAVAFLAGS $TEST) >/dev/null 2>&1; then
  ac_cv_prog_java_works=yes
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat $JAVA_TEST >&AC_FD_CC
  AC_MSG_ERROR(The Java VM $JAVA failed (see config.log, check the CLASSPATH?))
fi
rm -fr $JAVA_TEST $CLASS_TEST Test.uue
])
AC_PROVIDE([$0])dnl
]
)




dnl *** file: config/autoconf-archive-macros/ac_check_class.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_check_class.m4***
dnl @synopsis AC_CHECK_CLASS
dnl
dnl AC_CHECK_CLASS tests the existence of a given Java class, either in
dnl a jar or in a '.class' file.
dnl
dnl *Warning*: its success or failure can depend on a proper setting of the
dnl CLASSPATH env. variable.
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
AC_DEFUN([AC_CHECK_CLASS],[
AC_REQUIRE([AC_PROG_JAVA])
ac_var_name=`echo $1 | sed 's/\./_/g'`
dnl Normaly I'd use a AC_CACHE_CHECK here but since the variable name is
dnl dynamic I need an extra level of extraction
AC_MSG_CHECKING([for $1 class])
AC_CACHE_VAL(ac_cv_class_$ac_var_name, [
if test x$ac_cv_prog_uudecode_base64 = xyes; then
dnl /**
dnl  * Test.java: used to test dynamicaly if a class exists.
dnl  */
dnl public class Test
dnl {
dnl
dnl public static void
dnl main( String[] argv )
dnl {
dnl     Class lib;
dnl     if (argv.length < 1)
dnl      {
dnl             System.err.println ("Missing argument");
dnl             System.exit (77);
dnl      }
dnl     try
dnl      {
dnl             lib = Class.forName (argv[0]);
dnl      }
dnl     catch (ClassNotFoundException e)
dnl      {
dnl             System.exit (1);
dnl      }
dnl     lib = null;
dnl     System.exit (0);
dnl }
dnl
dnl }
cat << \EOF > Test.uue
begin-base64 644 Test.class
yv66vgADAC0AKQcAAgEABFRlc3QHAAQBABBqYXZhL2xhbmcvT2JqZWN0AQAE
bWFpbgEAFihbTGphdmEvbGFuZy9TdHJpbmc7KVYBAARDb2RlAQAPTGluZU51
bWJlclRhYmxlDAAKAAsBAANlcnIBABVMamF2YS9pby9QcmludFN0cmVhbTsJ
AA0ACQcADgEAEGphdmEvbGFuZy9TeXN0ZW0IABABABBNaXNzaW5nIGFyZ3Vt
ZW50DAASABMBAAdwcmludGxuAQAVKExqYXZhL2xhbmcvU3RyaW5nOylWCgAV
ABEHABYBABNqYXZhL2lvL1ByaW50U3RyZWFtDAAYABkBAARleGl0AQAEKEkp
VgoADQAXDAAcAB0BAAdmb3JOYW1lAQAlKExqYXZhL2xhbmcvU3RyaW5nOylM
amF2YS9sYW5nL0NsYXNzOwoAHwAbBwAgAQAPamF2YS9sYW5nL0NsYXNzBwAi
AQAgamF2YS9sYW5nL0NsYXNzTm90Rm91bmRFeGNlcHRpb24BAAY8aW5pdD4B
AAMoKVYMACMAJAoAAwAlAQAKU291cmNlRmlsZQEACVRlc3QuamF2YQAhAAEA
AwAAAAAAAgAJAAUABgABAAcAAABtAAMAAwAAACkqvgSiABCyAAwSD7YAFBBN
uAAaKgMyuAAeTKcACE0EuAAaAUwDuAAasQABABMAGgAdACEAAQAIAAAAKgAK
AAAACgAAAAsABgANAA4ADgATABAAEwASAB4AFgAiABgAJAAZACgAGgABACMA
JAABAAcAAAAhAAEAAQAAAAUqtwAmsQAAAAEACAAAAAoAAgAAAAQABAAEAAEA
JwAAAAIAKA==
====
EOF
                if uudecode$EXEEXT Test.uue; then
                        :
                else
                        echo "configure: __oline__: uudecode had trouble decoding base 64 file 'Test.uue'" >&AC_FD_CC
                        echo "configure: failed file was:" >&AC_FD_CC
                        cat Test.uue >&AC_FD_CC
                        ac_cv_prog_uudecode_base64=no
                fi
        rm -f Test.uue
        if AC_TRY_COMMAND($JAVA $JAVAFLAGS Test $1) >/dev/null 2>&1; then
                eval "ac_cv_class_$ac_var_name=yes"
        else
                eval "ac_cv_class_$ac_var_name=no"
        fi
        rm -f Test.class
else
        AC_TRY_COMPILE_JAVA([$1], , [eval "ac_cv_class_$ac_var_name=yes"],
                [eval "ac_cv_class_$ac_var_name=no"])
fi
eval "ac_var_val=$`eval echo ac_cv_class_$ac_var_name`"
eval "HAVE_$ac_var_name=$`echo ac_cv_class_$ac_var_val`"
HAVE_LAST_CLASS=$ac_var_val
if test x$ac_var_val = xyes; then
        ifelse([$2], , :, [$2])
else
        ifelse([$3], , :, [$3])
fi
])
dnl for some reason the above statment didn't fall though here?
dnl do scripts have variable scoping?
eval "ac_var_val=$`eval echo ac_cv_class_$ac_var_name`"
AC_MSG_RESULT($ac_var_val)
])




dnl *** file: config/autoconf-archive-macros/ac_check_classpath.m4


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
AC_DEFUN([AC_CHECK_CLASSPATH],[
if test "x$CLASSPATH" = x; then
        echo "You have no CLASSPATH, I hope it is good"
else
        echo "You have CLASSPATH $CLASSPATH, hope it is correct"
fi
])




dnl *** file: config/autoconf-archive-macros/ac_prog_javac.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_javac.m4***
dnl @synopsis AC_PROG_JAVAC
dnl
dnl AC_PROG_JAVAC tests an existing Java compiler. It uses the environment
dnl variable JAVAC then tests in sequence various common Java compilers. For
dnl political reasons, it starts with the free ones.
dnl
dnl If you want to force a specific compiler:
dnl
dnl - at the configure.in level, set JAVAC=yourcompiler before calling
dnl AC_PROG_JAVAC
dnl
dnl - at the configure level, setenv JAVAC
dnl
dnl You can use the JAVAC variable in your Makefile.in, with @JAVAC@.
dnl
dnl *Warning*: its success or failure can depend on a proper setting of the
dnl CLASSPATH env. variable.
dnl
dnl TODO: allow to exclude compilers (rationale: most Java programs cannot compile
dnl with some compilers like guavac).
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
AC_DEFUN([AC_PROG_JAVAC],[
AC_REQUIRE([AC_EXEEXT])dnl
if test "x$JAVAPREFIX" = x; then
        test "x$JAVAC" = x && AC_CHECK_PROGS(JAVAC, "gcj$EXEEXT -C" guavac$EXEEXT jikes$EXEEXT javac$EXEEXT)
else
        test "x$JAVAC" = x && AC_CHECK_PROGS(JAVAC, "gcj$EXEEXT -C" guavac$EXEEXT jikes$EXEEXT javac$EXEEXT, $JAVAPREFIX)
fi
test "x$JAVAC" = x && AC_MSG_ERROR([no acceptable Java compiler found in \$PATH])
AC_PROG_JAVAC_WORKS
AC_PROVIDE([$0])dnl
])




dnl *** file: config/autoconf-archive-macros/ac_prog_javac_works.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_javac_works.m4***
dnl @synopsis AC_PROG_JAVAC_WORKS
dnl
dnl Internal use ONLY.
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
AC_DEFUN([AC_PROG_JAVAC_WORKS],[
AC_CACHE_CHECK([if $JAVAC works], ac_cv_prog_javac_works, [
JAVA_TEST=Test.java
CLASS_TEST=Test.class
cat << \EOF > $JAVA_TEST
/* [#]line __oline__ "configure" */
public class Test {
}
EOF
if AC_TRY_COMMAND($JAVAC $JAVACFLAGS $JAVA_TEST) >/dev/null 2>&1; then
  ac_cv_prog_javac_works=yes
else
  AC_MSG_ERROR([The Java compiler $JAVAC failed (see config.log, check the CLASSPATH?)])
  echo "configure: failed program was:" >&AC_FD_CC
  cat $JAVA_TEST >&AC_FD_CC
fi
rm -f $JAVA_TEST $CLASS_TEST
])
AC_PROVIDE([$0])dnl
])




dnl *** file: config/autoconf-archive-macros/ac_prog_jar.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_jar.m4***
dnl @synopsis AC_PROG_JAR
dnl
dnl AC_PROG_JAR tests for an existing jar program. It uses the environment
dnl variable JAR then tests in sequence various common jar programs.
dnl
dnl If you want to force a specific compiler:
dnl
dnl - at the configure.in level, set JAR=yourcompiler before calling
dnl AC_PROG_JAR
dnl
dnl - at the configure level, setenv JAR
dnl
dnl You can use the JAR variable in your Makefile.in, with @JAR@.
dnl
dnl Note: This macro depends on the autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download that whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl
dnl The general documentation of those macros, as well as the sample
dnl configure.in, is included in the AC_PROG_JAVA macro.
dnl
dnl @author Egon Willighagen <egonw@sci.kun.nl>
dnl @version $Id$
dnl
AC_DEFUN([AC_PROG_JAR],[
AC_REQUIRE([AC_EXEEXT])dnl
if test "x$JAVAPREFIX" = x; then
        test "x$JAR" = x && AC_CHECK_PROGS(JAR, jar$EXEEXT)
else
        test "x$JAR" = x && AC_CHECK_PROGS(JAR, jar, $JAVAPREFIX)
fi
test "x$JAR" = x && AC_MSG_ERROR([no acceptable jar program found in \$PATH])
AC_PROVIDE([$0])dnl
])




dnl *** file: config/autoconf-archive-macros/ac_prog_javadoc.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_prog_javadoc.m4***
dnl @synopsis AC_PROG_JAVADOC
dnl
dnl AC_PROG_JAVADOC tests for an existing javadoc generator. It uses the environment
dnl variable JAVADOC then tests in sequence various common javadoc generator.
dnl
dnl If you want to force a specific compiler:
dnl
dnl - at the configure.in level, set JAVADOC=yourgenerator before calling
dnl AC_PROG_JAVADOC
dnl
dnl - at the configure level, setenv JAVADOC
dnl
dnl You can use the JAVADOC variable in your Makefile.in, with @JAVADOC@.
dnl
dnl Note: This macro depends on the autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download that whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl
dnl The general documentation of those macros, as well as the sample
dnl configure.in, is included in the AC_PROG_JAVA macro.
dnl
dnl @author Egon Willighagen <egonw@sci.kun.nl>
dnl @version $Id$
dnl
AC_DEFUN([AC_PROG_JAVADOC],[
AC_REQUIRE([AC_EXEEXT])dnl
if test "x$JAVAPREFIX" = x; then
        test "x$JAVADOC" = x && AC_CHECK_PROGS(JAVADOC, javadoc$EXEEXT)
else
        test "x$JAVADOC" = x && AC_CHECK_PROGS(JAVADOC, javadoc, $JAVAPREFIX)
fi
test "x$JAVADOC" = x && AC_MSG_ERROR([no acceptable javadoc generator found in \$PATH])
AC_PROVIDE([$0])dnl
])





dnl *** file: config/autoconf-archive-macros/ac_try_compile_java.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/Java_Support/ac_try_compile_java.m4***
dnl @synopsis AC_TRY_COMPILE_JAVA
dnl
dnl AC_TRY_COMPILE_JAVA attempt to compile user given source.
dnl
dnl *Warning*: its success or failure can depend on a proper setting of the
dnl CLASSPATH env. variable.
dnl
dnl Note: This is part of the set of autoconf M4 macros for Java programs.
dnl It is VERY IMPORTANT that you download the whole set, some
dnl macros depend on other. Unfortunately, the autoconf archive does not
dnl support the concept of set of macros, so I had to break it for
dnl submission.
dnl The general documentation, as well as the sample configure.in, is
dnl included in the AC_PROG_JAVA macro.
dnl
dnl @author Devin Weaver <ktohg@tritarget.com>
dnl @version $Id$
dnl
AC_DEFUN([AC_TRY_COMPILE_JAVA],[
AC_REQUIRE([AC_PROG_JAVAC])dnl
cat << \EOF > Test.java
/* [#]line __oline__ "configure" */
ifelse([$1], , , [import $1;])
public class Test {
[$2]
}
EOF
if AC_TRY_COMMAND($JAVAC $JAVACFLAGS Test.java) && test -s Test.class
then
dnl Don't remove the temporary files here, so they can be examined.
  ifelse([$3], , :, [$3])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat Test.java >&AC_FD_CC
ifelse([$4], , , [  rm -fr Test*
  $4
])dnl
fi
rm -fr Test*])




dnl *** file: config/autoconf-archive-macros/ac_cxx_namespaces.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_namespaces.m4***
dnl @synopsis AC_CXX_NAMESPACES
dnl
dnl If the compiler can prevent names clashes using namespaces, define
dnl HAVE_NAMESPACES.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_NAMESPACES],
[AC_CACHE_CHECK(whether the compiler implements namespaces,
ac_cv_cxx_namespaces,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([namespace Outer { namespace Inner { int i = 0; }}],
                [using namespace Outer::Inner; return i;],
 ac_cv_cxx_namespaces=yes, ac_cv_cxx_namespaces=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_namespaces" = yes; then
  AC_DEFINE(HAVE_NAMESPACES,,[define if the compiler implements namespaces])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_std.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_std.m4***
dnl @synopsis AC_CXX_HAVE_STD
dnl
dnl If the compiler supports ISO C++ standard library (i.e., can include the
dnl files iostream, map, iomanip and cmath}), define HAVE_STD.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_STD],
[AC_CACHE_CHECK(whether the compiler supports ISO C++ standard library,
ac_cv_cxx_have_std,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <iostream>
#include <map>
#include <iomanip>
#include <cmath>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[return 0;],
 ac_cv_cxx_have_std=yes, ac_cv_cxx_have_std=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_std" = yes; then
  AC_DEFINE(HAVE_STD,,[define if the compiler supports ISO C++ standard library])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_stl.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_stl.m4***
dnl @synopsis AC_CXX_HAVE_STL
dnl
dnl If the compiler supports the Standard Template Library, define HAVE_STL.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_STL],
[AC_CACHE_CHECK(whether the compiler supports Standard Template Library,
ac_cv_cxx_have_stl,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <list>
#include <deque>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[list<int> x; x.push_back(5);
list<int>::iterator iter = x.begin(); if (iter != x.end()) ++iter; return 0;],
 ac_cv_cxx_have_stl=yes, ac_cv_cxx_have_stl=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_stl" = yes; then
  AC_DEFINE(HAVE_STL,,[define if the compiler supports Standard Template Library])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_numeric_limits.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_numeric_limits.m4***
dnl @synopsis AC_CXX_HAVE_NUMERIC_LIMITS
dnl
dnl If the compiler has numeric_limits<T>, define HAVE_NUMERIC_LIMITS.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_NUMERIC_LIMITS],
[AC_CACHE_CHECK(whether the compiler has numeric_limits<T>,
ac_cv_cxx_have_numeric_limits,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <limits>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[double e = numeric_limits<double>::epsilon(); return 0;],
 ac_cv_cxx_have_numeric_limits=yes, ac_cv_cxx_have_numeric_limits=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_numeric_limits" = yes; then
  AC_DEFINE(HAVE_NUMERIC_LIMITS,,[define if the compiler has numeric_limits<T>])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_complex_math_in_namespace_std.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_complex_math_in_namespace_std.m4***
dnl @synopsis AC_CXX_COMPLEX_MATH_IN_NAMESPACE_STD
dnl
dnl If the C math functions are in the cmath header file and std:: namespace,
dnl define HAVE_MATH_FN_IN_NAMESPACE_STD.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_COMPLEX_MATH_IN_NAMESPACE_STD],
[AC_CACHE_CHECK(whether complex math functions are in std::,
ac_cv_cxx_complex_math_in_namespace_std,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <complex>
namespace S { using namespace std;
              complex<float> pow(complex<float> x, complex<float> y)
              { return std::pow(x,y); }
            };
],[using namespace S; complex<float> x = 1.0, y = 1.0; S::pow(x,y); return 0;],
 ac_cv_cxx_complex_math_in_namespace_std=yes, ac_cv_cxx_complex_math_in_namespace_std=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_complex_math_in_namespace_std" = yes; then
  AC_DEFINE(HAVE_COMPLEX_MATH_IN_NAMESPACE_STD,,
            [define if complex math functions are in std::])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_complex.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_complex.m4***
dnl @synopsis AC_CXX_HAVE_COMPLEX
dnl
dnl If the compiler has complex<T>, define HAVE_COMPLEX.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_COMPLEX],
[AC_CACHE_CHECK(whether the compiler has complex<T>,
ac_cv_cxx_have_complex,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 AC_TRY_COMPILE([#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[complex<float> a; complex<double> b; return 0;],
 ac_cv_cxx_have_complex=yes, ac_cv_cxx_have_complex=no)
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_complex" = yes; then
  AC_DEFINE(HAVE_COMPLEX,,[define if the compiler has complex<T>])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_complex_math1.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_complex_math1.m4***
dnl @synopsis AC_CXX_HAVE_COMPLEX_MATH1
dnl
dnl If the compiler has the complex math functions cos, cosh, exp, log,
dnl pow, sin, sinh, sqrt, tan and tanh, define HAVE_COMPLEX_MATH1.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_COMPLEX_MATH1],
[AC_CACHE_CHECK(whether the compiler has complex math functions,
ac_cv_cxx_have_complex_math1,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -lm"
 AC_TRY_LINK([#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[complex<double> x(1.0, 1.0), y(1.0, 1.0);
cos(x); cosh(x); exp(x); log(x); pow(x,1); pow(x,double(2.0));
pow(x, y); pow(double(2.0), x); sin(x); sinh(x); sqrt(x); tan(x); tanh(x);
return 0;],
 ac_cv_cxx_have_complex_math1=yes, ac_cv_cxx_have_complex_math1=no)
 LIBS="$ac_save_LIBS"
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_complex_math1" = yes; then
  AC_DEFINE(HAVE_COMPLEX_MATH1,,[define if the compiler has complex math functions])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_complex_math2.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_complex_math2.m4***
dnl @synopsis AC_CXX_HAVE_COMPLEX_MATH2
dnl
dnl If the compiler has the complex math functions acos, asin,
dnl atan, atan2 and log10, define HAVE_COMPLEX_MATH2.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_COMPLEX_MATH2],
[AC_CACHE_CHECK(whether the compiler has more complex math functions,
ac_cv_cxx_have_complex_math2,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -lm"
 AC_TRY_LINK([#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif],[complex<double> x(1.0, 1.0), y(1.0, 1.0);
acos(x); asin(x); atan(x); atan2(x,y); atan2(x, double(3.0));
atan2(double(3.0), x); log10(x); return 0;],
 ac_cv_cxx_have_complex_math2=yes, ac_cv_cxx_have_complex_math2=no)
 LIBS="$ac_save_LIBS"
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_complex_math2" = yes; then
  AC_DEFINE(HAVE_COMPLEX_MATH2,,[define if the compiler has more complex math functions])
fi
])




dnl *** file: config/autoconf-archive-macros/ac_cxx_have_ieee_math.m4


dnl *** Downloaded from http://gnu.wwc.edu/software/ac-archive/C++_Support/ac_cxx_have_ieee_math.m4***
dnl @synopsis AC_CXX_HAVE_IEEE_MATH
dnl
dnl If the compiler has the double math functions acosh,
dnl asinh, atanh, expm1, erf, erfc, isnan, j0, j1, lgamma, logb,
dnl log1p, rint, y0 and y1, define HAVE_IEEE_MATH.
dnl
dnl @version $Id$
dnl @author Luc Maisonobe
dnl
AC_DEFUN([AC_CXX_HAVE_IEEE_MATH],
[AC_CACHE_CHECK(whether the compiler supports IEEE math library,
ac_cv_cxx_have_ieee_math,
[AC_LANG_SAVE
 AC_LANG_CPLUSPLUS
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -lm"
 AC_TRY_LINK([
#ifndef _ALL_SOURCE
 #define _ALL_SOURCE
#endif
#ifndef _XOPEN_SOURCE
 #define _XOPEN_SOURCE
#endif
#ifndef _XOPEN_SOURCE_EXTENDED
 #define _XOPEN_SOURCE_EXTENDED 1
#endif
#include <math.h>],[double x = 1.0; double y = 1.0;
acosh(x); asinh(x); atanh(x); expm1(x); erf(x); erfc(x); isnan(x);
j0(x); j1(x); lgamma(x); logb(x); log1p(x); rint(x); y0(x); y1(x);
return 0;],
 ac_cv_cxx_have_ieee_math=yes, ac_cv_cxx_have_ieee_math=no)
 LIBS="$ac_save_LIBS"
 AC_LANG_RESTORE
])
if test "$ac_cv_cxx_have_ieee_math" = yes; then
  AC_DEFINE(HAVE_IEEE_MATH,,[define if the compiler supports IEEE math library])
fi
])




dnl *** file: config/llnl-ac-macros/llnl_check_autoconf.m4

dnl
dnl @synopsis LLNL_CHECK_AUTOCONF(VERSION)
dnl
dnl Check whether autoconf is the specified version.
dnl
dnl @author Gary Kumfert
AC_DEFUN(LLNL_CHECK_AUTOCONF, [
  AC_MSG_CHECKING(for autoconf/autoheader version)
  changequote(,)
  llnl_autoconf_version=`autoconf --version 2>/dev/null | sed '1s/.* \([^ ]*\)$/\1/g;1q'`
  changequote([,])
  if test "X$1" = "X$llnl_autoconf_version"; then
    AC_MSG_RESULT([$llnl_autoconf_version (enabled)])
  else
    AUTOCONF="$SHELL $am_aux_dir/disabled autoconf"
    AUTOHEADER="$SHELL $am_aux_dir/disabled autoheader"
    AC_MSG_RESULT([$llnl_autoconf_version (disabled)])
  fi
])


dnl *** file: config/llnl-ac-macros/llnl_check_automake.m4

dnl
dnl @synopsis LLNL_CHECK_AUTOMAKE(VERSION)
dnl
dnl Check whether automake is the specified version.
dnl
dnl @author Gary Kumfert
AC_DEFUN(LLNL_CHECK_AUTOMAKE, [
  AC_MSG_CHECKING(for automake/aclocal version)
  changequote(,)
  llnl_automake_version=`automake --version 2>/dev/null | sed '1s/.* \([^ ]*\)$/\1/g;1q'`
  changequote([,])
  if test "X$1" = "X$llnl_automake_version"; then
    AC_MSG_RESULT([$llnl_automake_version (enabled)])
  else
    AUTOMAKE="$SHELL $am_aux_dir/disabled automake"
    ACLOCAL="$SHELL $am_aux_dir/disabled autoconf"
    AC_MSG_RESULT([$llnl_automake_version (disabled)])
  fi
])


dnl *** file: config/llnl-ac-macros/llnl_check_int32_t.m4

dnl @synopsis LLNL_CHECK_INT32_T
dnl  
dnl Checks for a int32_t in inttypes.h, systypes.h,
dnl stdlib.h and stddef.h.  If none is found int32_t
dnl is defined as some 32bit signed integer searched
dnl by alternative means.
dnl
dnl @author Gary Kumfert

AC_DEFUN(LLNL_CHECK_INT32_T, 
[AC_REQUIRE([LLNL_FIND_32BIT_SIGNED_INT])
 AC_REQUIRE([AC_HEADER_STDC])
 AC_CACHE_CHECK(for int32_t, llnl_cv_int32_t,
 [AC_LANG_SAVE
  AC_LANG_C
  AC_TRY_COMPILE([#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#include <sys/types.h>
#if STDC_HEADERS
#include <stdlib.h>
#include <stddef.h>
#endif
],[int32_t t],llnl_cv_int32_t=yes,llnl_cv_int32_t=no)
  AC_LANG_RESTORE
])
if test "$llnl_cv_int32_t" = no; then 
  if test "$llnl_cv_find_32bit_signed_int" = "unresolved"; then
    AC_MSG_ERROR([Cannot find int32_t or an alternative 4 byte integer])
  else 
    AC_MSG_WARN([Using $llnl_cv_find_32bit_signed_int instead of int32_t])
    AC_DEFINE_UNQUOTED(int32_t, $llnl_cv_find_32bit_signed_int,
	  [used when a compiler does not recognize int32_t])
  fi
fi
])


dnl *** file: config/llnl-ac-macros/llnl_check_int64_t.m4

dnl @synopsis LLNL_CHECK_INT64_T
dnl  
dnl Checks for a int64_t in inttypes.h, systypes.h,
dnl stdlib.h and stddef.h.  If none is found int64_t
dnl is defined as some 64bit signed integer searched
dnl by alternative means.
dnl
dnl @author Gary Kumfert

AC_DEFUN(LLNL_CHECK_INT64_T, 
[AC_REQUIRE([LLNL_FIND_64BIT_SIGNED_INT])
 AC_REQUIRE([AC_HEADER_STDC])
 AC_CACHE_CHECK(for int64_t, llnl_cv_int64_t,
 [AC_LANG_SAVE
  AC_LANG_C
  AC_TRY_COMPILE([#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif
#include <sys/types.h>
#if STDC_HEADERS
#include <stdlib.h>
#include <stddef.h>
#endif
],[int64_t t],llnl_cv_int64_t=yes,llnl_cv_int64_t=no)
  AC_LANG_RESTORE
])
if test "$llnl_cv_int64_t" = "no"; then 
  if test "$llnl_cv_find_64bit_signed_int" = "unresolved"; then
    AC_MSG_ERROR([Cannot find int64_t or an alternative 8 byte integer])
  else 
    AC_MSG_WARN([Using $llnl_cv_find_64bit_signed_int instead of int64_t])
    AC_DEFINE_UNQUOTED(int64_t, $llnl_cv_find_64bit_signed_int,
	  [used when a compiler does not recognize int64_t])
  fi
fi
])


dnl *** file: config/llnl-ac-macros/llnl_check_long_long.m4

dnl @synopsis LLNL_CHECK_LONG_LONG
dnl
dnl checks for a `long long' type
dnl 
dnl @version 
dnl @author Gary Kumfert, LLNL
AC_DEFUN(LLNL_CHECK_LONG_LONG,
[AC_CACHE_CHECK(for type long long,
 ac_cv_c_long_long,
 AC_TRY_RUN([int main() {
 exit(sizeof(long long) < sizeof(long)); }],
 ac_cv_c_long_long=yes, ac_cv_c_long_long=no)
 if test $ac_cv_c_long_long = yes; then
   AC_DEFINE(HAVE_LONG_LONG,,[define if long long is a built in type])
 fi
)])




dnl *** file: config/llnl-ac-macros/llnl_cxx_library_ldflags.m4

dnl @synopsis LLNL_CXX_LIBRARY_LDFLAGS
dnl
dnl Determine the linker flags (e.g., `-L' and `-l') for the C++ run-time
dnl libaries that are required to successfully link a C++ program or shared
dnl library.  The output variable CXXLIBS is set to these flags.  This macro
dnl is intended for situations for which it is necessary to mix different
dnl languages into a single program or shared library.
dnl
dnl @author Gary Kumfert
AC_DEFUN(LLNL_CXX_LIBRARY_LDFLAGS,
[AC_REQUIRE([AC_PROG_CXX])
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
          arg=`echo $arg | sed 's/^lib/-l'/'`
        ;;
        -lang* | -lcrt0.o)
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


dnl *** file: config/llnl-ac-macros/llnl_cxx_old_header_suffix.m4

dnl 
dnl @synopsis LLNL_CXX_OLD_HEADER_SUFFIX
dnl 
dnl If the CXX compiler requires *.h includes define `LLNL_CXX_OLD_HEADER_SUFFIX'
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
AC_DEFUN(LLNL_CXX_OLD_HEADER_SUFFIX,
[
AC_MSG_CHECKING([if ${CXX} requires requires old .h-style header includes])
AC_CACHE_VAL(llnl_cv_old_cxx_header_suffix,
[
AC_LANG_SAVE
AC_LANG_CPLUSPLUS
AC_TRY_COMPILE([#include <iostream>],
[ using namespace std; cout << "Hello World!";],
llnl_cv_old_cxx_header_suffix=no,
llnl_cv_old_cxx_header_suffix=yes)
AC_LANG_RESTORE
])
AC_MSG_RESULT($llnl_cv_old_cxx_header_suffix)
if test "$llnl_cv_old_cxx_header_suffix" = yes; then
AC_DEFINE(REQUIRE_OLD_CXX_HEADER_SUFFIX,,[define if C++ requires old .h-style header includes])
fi])	


dnl *** file: config/llnl-ac-macros/llnl_f77_c_config.m4
dnl
dnl @synopsis LLNL_F77_C_CONFIG
dnl
dnl
dnl @author
dnl
AC_DEFUN([LLNL_F77_C_CONFIG],[
AC_REQUIRE([AC_CANONICAL_TARGET])dnl
AC_REQUIRE([AC_PROG_F77])dnl
AC_REQUIRE([LLNL_F77_NAME_MANGLING])
dnl set some resonable defaults
sidl_cv_f77_str="str" dnl can also be "struct"
sidl_cv_f77_str_len="far" dnl can also be "near", only meaningful if $sidl_cv_str="str"
sidl_cv_f77_str_struct="str_len" dnl can also be "len_str", only meaningful if $sidl_cv_str="struct"
sidl_cv_f77_char="as_string"
dnl sidl_cv_f77_true=1
dnl sidl_cv_f77_false=0
sidl_cv_f77_str_minsize=512

AC_CACHE_CHECK([the integer value of F77's .true.], sidl_cv_f77_true,[dnl
AC_LANG_PUSH(Fortran 77)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(,[
        logical log
        integer value
        equivalence (log, value)
        log = .true.
        write (*,*) value
],[
sidl_cv_f77_true=`./conftest$ac_exeext`
],[AC_MSG_WARN([Unable to determine integer value of F77 .true.])
	echo "the program generates"  `./conftest$ac_exeext`])
AC_LANG_POP(Fortran 77)dnl])

AC_CACHE_CHECK([the integer value of F77's .false.], sidl_cv_f77_false,[dnl
AC_LANG_PUSH(Fortran 77)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(,[
        logical log
        integer value
        equivalence (log, value)
        log = .false.
	write (*,*) value
],[
sidl_cv_f77_false=`./conftest$ac_exeext`
],[AC_MSG_WARN([Unable to determine integer value of F77 .false.])])
AC_LANG_POP(Fortran 77)dnl
])

dnl set number of underscores
if test -z "$sidl_cv_f77_number_underscores"; then
   AC_MSG_ERROR([Number of underscores not determined])
elif test $sidl_cv_f77_number_underscores -eq 2; then
   AC_DEFINE(SIDL_F77_TWO_UNDERSCORE,,[two underscores after F77 symbols])
elif test $sidl_cv_f77_number_underscores -eq 1; then
   AC_DEFINE(SIDL_F77_ONE_UNDERSCORE,,[one underscore after F77 symbols])
else
  if test $sidl_cv_f77_number_underscores -ne 0; then
     AC_WARN([number of underscores after F77 symbols undetermined, assuming zero])
  fi;
   AC_DEFINE(SIDL_F77_ZERO_UNDERSCORE,,[no underscores after F77 symbols])
fi;
dnl set case
if test "$sidl_cv_f77_case" = "mixed"; then
   AC_DEFINE(SIDL_F77_MIXED_CASE,,[F77 symbols are mixed case])
elif test "$sidl_cv_f77_case" = "upper"; then
   AC_DEFINE(SIDL_F77_UPPER_CASE,,[F77 symbols are upper case])
else
   if test "$sidl_cv_f77_case" != "lower"; then
      AC_WARN([case of f77 symbols undetermined, assuming lower case])
   fi  
   AC_DEFINE(SIDL_F77_LOWER_CASE,,[F77 symbols are lower case])
fi;
dnl strings
if test "$sidl_cv_f77_str" = "struct"; then 
   if test "$sidl_cv_f77_str_struct" = "len_str"; then
      AC_DEFINE(SIDL_F77_STR_STRUCT_LEN_STR,,[F77 strings as length-char* structs])
   else 
      if test "$sidl_cv_f77_str_struct" != "str_len"; then
         AC_WARN([string structs as length-charptr or char_ptr/length undetermined, assuming the latter])   
      fi;
      AC_DEFINE(SIDL_F77_STR_STRUCT_STR_LEN,,[F77 strings as char*-length structs])
   fi;
else
   if test "$sidl_cv_f77_str" != "str"; then 
      AC_WARN([strings passed as structs or char*/length undetermined, assumming the latter])
   fi;
   if test "$sidl_cv_f77_str_len" = "near"; then
      AC_DEFINE(SIDL_F77_STR_LEN_NEAR,,[F77 strings lengths at end])
   else
      if test "$sidl_cv_f77_str_len" != "far"; then
         AC_WARN([string length immediately following char* or at end undetermined, assuming at end])
      fi
      AC_DEFINE(SIDL_F77_STR_LEN_FAR,,[F77 strings lengths immediately follow string])
   fi;
fi;
if test "$sidl_cv_f77_char" = "as_string"; then
   AC_DEFINE(SIDL_F77_CHAR_AS_STRING,,[F77 char args are strings])
fi; 
AC_DEFINE_UNQUOTED(SIDL_F77_TRUE,$sidl_cv_f77_true,[F77 logical true value])
AC_DEFINE_UNQUOTED(SIDL_F77_FALSE,$sidl_cv_f77_false,[F77 logical false value])
AC_DEFINE_UNQUOTED(SIDL_F77_STR_MINSIZE,$sidl_cv_f77_str_minsize,[Minimum size for out strings])
])


dnl *** file: config/llnl-ac-macros/llnl_f77_name_mangling.m4

# LLNL_F77_NAME_MANGLING
# ---------------------
# Test for the name mangling scheme used by the Fortran 77 compiler.
#
# Sets ac_cv_f77_mangling. The value contains three fields, separated
# by commas:
#
# lower case / upper case:
#    case translation of the Fortran 77 symbols
# underscore / no underscore:
#    whether the compiler appends "_" to symbol names
# extra underscore / no extra underscore:
#    whether the compiler appends an extra "_" to symbol names already
#    containing at least one underscore
#
AC_DEFUN([LLNL_F77_NAME_MANGLING],
[AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])dnl
AC_REQUIRE([AC_F77_DUMMY_MAIN])dnl
AC_CACHE_CHECK([for Fortran 77 name-mangling scheme],
               ac_cv_f77_mangling,
[AC_LANG_PUSH(Fortran 77)dnl
AC_COMPILE_IFELSE(
[      subroutine Foobar()
      return
      end
      subroutine Foo_bar()
      return
      end],
[mv conftest.$ac_objext cf77_test.$ac_objext

  AC_LANG_PUSH(C)dnl

  ac_save_LIBS=$LIBS
  LIBS="cf77_test.$ac_objext $LIBS $FLIBS"

  ac_success=no
  for ac_foobar in foobar Foobar FOOBAR; do
    for ac_underscore in "" "_"; do
      ac_func="$ac_foobar$ac_underscore"
      AC_TRY_LINK_FUNC($ac_func,
         [ac_success=yes; break 2])
    done
  done

  if test "$ac_success" = "yes"; then
     case $ac_foobar in
        foobar)
	   sidl_cv_f77_case="lower"
           ac_foo_bar=foo_bar
           ;;
        FOOBAR)
	   sidl_cv_f77_case="upper"
           ac_foo_bar=FOO_BAR
           ;;
        Foobar)
	   sidl_cv_f77_case="mixed"
           ac_foo_bar=Foo_bar
           ;;
     esac

     ac_success_extra=no
     for ac_extra in "" "_"; do
        ac_func="$ac_foo_bar$ac_underscore$ac_extra"
        AC_TRY_LINK_FUNC($ac_func,
        [ac_success_extra=yes; break])
     done

     if test "$ac_success_extra" = "yes"; then
	ac_cv_f77_mangling="$sidl_cv_f77_case case"
        if test -z "$ac_underscore"; then
           ac_cv_f77_mangling="$ac_cv_f77_mangling, no underscore"
	   sidl_cv_f77_number_underscores=0
	else
           ac_cv_f77_mangling="$ac_cv_f77_mangling, underscore"
	   sidl_cv_f77_number_underscores=1
        fi
        if test -z "$ac_extra"; then
           ac_cv_f77_mangling="$ac_cv_f77_mangling, no extra underscore"
	else
           ac_cv_f77_mangling="$ac_cv_f77_mangling, extra underscore"
	   sidl_cv_f77_number_underscores=`expr $sidl_cv_f77_number_underscores + 1`
        fi
      else
	ac_cv_f77_mangling="unknown"
      fi
  else
     ac_cv_f77_mangling="unknown"
  fi

  if test "$ac_cv_f77_mangling" = "unknown"; then
    AC_MSG_ERROR([Failed to determine how F77 mangles linker symbols.])
  fi
  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f cf77_test* conftest*])
AC_LANG_POP(Fortran 77)dnl
])
])# LLNL_F77_NAME_MANGLING


dnl *** file: config/llnl-ac-macros/llnl_f90_c_config.m4
dnl
dnl @synopsis LLNL_F90_C_CONFIG
dnl
dnl
dnl @author
dnl
dnl Note:  Clone of F77 version.

AC_DEFUN([LLNL_F90_C_CONFIG],[
AC_REQUIRE([AC_CANONICAL_TARGET])dnl
AC_REQUIRE([AC_PROG_F90])dnl
AC_REQUIRE([LLNL_F90_NAME_MANGLING])
dnl set some resonable defaults
sidl_cv_f90_str="str" dnl can also be "struct"
sidl_cv_f90_str_len="far" dnl can also be "near", only meaningful if $sidl_cv_str="str"
sidl_cv_f90_str_struct="str_len" dnl can also be "len_str", only meaningful if $sidl_cv_str="struct"
sidl_cv_f90_char="as_string"
dnl sidl_cv_f90_true=1
dnl sidl_cv_f90_false=0
sidl_cv_f90_str_minsize=512

AC_CACHE_CHECK([the integer value of F90's .true.], sidl_cv_f90_true,[dnl
AC_LANG_PUSH(Fortran 90)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(,[
  logical log
  integer value
  equivalence (log, value)
  log = .true.
  write (*,*) value
],[dnl
sidl_cv_f90_true=`./conftest$ac_exeext`
],[AC_MSG_WARN([Unable to determine integer value of F90 .true.])])
AC_LANG_POP(Fortran 90)dnl])

AC_CACHE_CHECK([the integer value of F90's .false.], sidl_cv_f90_false,[dnl
AC_LANG_PUSH(Fortran 90)dnl
dnl should be AC_TRY_RUN, but the macro destroys conftest$ac_exeext too soon
dnl ignore the warnings this issues from automake: F77 does not use the 1st argument (includes)
AC_TRY_LINK(,[
  logical log
  integer value
  equivalence (log, value)
  log = .false.
  write (*,*) value
],[dnl
sidl_cv_f90_false=`./conftest$ac_exeext`
],[AC_MSG_WARN([Unable to determine integer value of F90 .false.])])
AC_LANG_POP(Fortran 90)dnl
])

dnl set number of underscores
if test -z "$sidl_cv_f90_number_underscores"; then
   AC_MSG_ERROR([Number of F90 underscores not determined])
elif test $sidl_cv_f90_number_underscores -eq 2; then
   AC_DEFINE(SIDL_F90_TWO_UNDERSCORE,,[two underscores after Fortran 90 symbols])
elif test $sidl_cv_f90_number_underscores -eq 1; then
   AC_DEFINE(SIDL_F90_ONE_UNDERSCORE,,[one underscore after Fortran 90 symbols])
else
  if test $sidl_cv_f90_number_underscores -ne 0; then
     AC_WARN([number of underscores after Fortran 90 symbols undetermined, assuming zero])
  fi;
   AC_DEFINE(SIDL_F90_ZERO_UNDERSCORE,,[no underscores after Fortran 90 symbols])
fi;
dnl set case
if test "$sidl_cv_f90_case" = "mixed"; then
   AC_DEFINE(SIDL_F90_MIXED_CASE,,[Fortran 90 symbols are mixed case])
elif test "$sidl_cv_f90_case" = "upper"; then
   AC_DEFINE(SIDL_F90_UPPER_CASE,,[Fortran 90 symbols are upper case])
else
   if test "$sidl_cv_f90_case" != "lower"; then
      AC_WARN([case of Fortran 90 symbols undetermined, assuming lower case])
   fi  
   AC_DEFINE(SIDL_F90_LOWER_CASE,,[Fortran 90 symbols are lower case])
fi;
dnl strings
if test "$sidl_cv_f90_str" = "struct"; then 
   if test "$sidl_cv_f90_str_struct" = "len_str"; then
      AC_DEFINE(SIDL_F90_STR_STRUCT_LEN_STR,,[Fortran 90 strings as length-char* structs])
   else 
      if test "$sidl_cv_f90_str_struct" != "str_len"; then
         AC_WARN([string structs as length-charptr or char_ptr/length undetermined, assuming the latter])   
      fi;
      AC_DEFINE(SIDL_F90_STR_STRUCT_STR_LEN,,[Fortran 90 strings as char*-length structs])
   fi;
else
   if test "$sidl_cv_f90_str" != "str"; then 
      AC_WARN([strings passed as structs or char*/length undetermined, assumming the latter])
   fi;
   if test "$sidl_cv_f90_str_len" = "near"; then
      AC_DEFINE(SIDL_F90_STR_LEN_NEAR,,[Fortran 90 strings lengths at end])
   else
      if test "$sidl_cv_f90_str_len" != "far"; then
         AC_WARN([string length immediately following char* or at end undetermined, assuming at end])
      fi
      AC_DEFINE(SIDL_F90_STR_LEN_FAR,,[Fortran 90 strings lengths immediately follow string])
   fi;
fi;
if test "$sidl_cv_f90_char" = "as_string"; then
   AC_DEFINE(SIDL_F90_CHAR_AS_STRING,,[Fortran 90 char args are strings])
fi; 
AC_DEFINE_UNQUOTED(SIDL_F90_TRUE,$sidl_cv_f90_true,[Fortran 90 logical true value])
AC_DEFINE_UNQUOTED(SIDL_F90_FALSE,$sidl_cv_f90_false,[Fortran 90 logical false value])
AC_DEFINE_UNQUOTED(SIDL_F90_STR_MINSIZE,$sidl_cv_f90_str_minsize,[Minimum size for out strings])
])


dnl *** file: config/llnl-ac-macros/llnl_f90_name_mangling.m4

# LLNL_F90_NAME_MANGLING
# ---------------------
# Test for the name mangling scheme used by the Fortran 90 compiler.
#
# Sets ac_cv_f90_mangling. The value contains three fields, separated
# by commas:
#
# lower case / upper case:
#    case translation of the Fortran 90 symbols
# underscore / no underscore:
#    whether the compiler appends "_" to symbol names
# extra underscore / no extra underscore:
#    whether the compiler appends an extra "_" to symbol names already
#    containing at least one underscore
#
# Note:  Clone of F77 version.
#
AC_DEFUN([LLNL_F90_NAME_MANGLING],
[AC_REQUIRE([AC_F90_LIBRARY_LDFLAGS])dnl
AC_REQUIRE([AC_F90_DUMMY_MAIN])dnl
AC_CACHE_CHECK([for Fortran 90 name-mangling scheme],
               ac_cv_f90_mangling,
[AC_LANG_PUSH(Fortran 90)dnl
AC_COMPILE_IFELSE(
[subroutine Foobar()
return
end subroutine Foobar
subroutine Foo_bar()
return
end subroutine Foo_bar],
[mv conftest.$ac_objext cf90_test.$ac_objext

  AC_LANG_PUSH(C)dnl

  ac_save_LIBS=$LIBS
  LIBS="cf90_test.$ac_objext $LIBS $F90LIBS"

  ac_success=no
  for ac_foobar in foobar Foobar FOOBAR; do
    for ac_underscore in "" "_"; do
      ac_func="$ac_foobar$ac_underscore"
      AC_TRY_LINK_FUNC($ac_func,
         [ac_success=yes; break 2])
    done
  done

  if test "$ac_success" = "yes"; then
     case $ac_foobar in
        foobar)
	   sidl_cv_f90_case="lower"
           ac_foo_bar=foo_bar
           ;;
        FOOBAR)
	   sidl_cv_f90_case="upper"
           ac_foo_bar=FOO_BAR
           ;;
        Foobar)
	   sidl_cv_f90_case="mixed"
           ac_foo_bar=Foo_bar
           ;;
     esac

     ac_success_extra=no
     for ac_extra in "" "_"; do
        ac_func="$ac_foo_bar$ac_underscore$ac_extra"
        AC_TRY_LINK_FUNC($ac_func,
        [ac_success_extra=yes; break])
     done

     if test "$ac_success_extra" = "yes"; then
	ac_cv_f90_mangling="$sidl_cv_f90_case case"
        if test -z "$ac_underscore"; then
           ac_cv_f90_mangling="$ac_cv_f90_mangling, no underscore"
	   sidl_cv_f90_number_underscores=0
	else
           ac_cv_f90_mangling="$ac_cv_f90_mangling, underscore"
	   sidl_cv_f90_number_underscores=1
        fi
        if test -z "$ac_extra"; then
           ac_cv_f90_mangling="$ac_cv_f90_mangling, no extra underscore"
	else
           ac_cv_f90_mangling="$ac_cv_f90_mangling, extra underscore"
	   sidl_cv_f90_number_underscores=`expr $sidl_cv_f90_number_underscores + 1`
        fi
      else
	ac_cv_f90_mangling="unknown"
      fi
  else
     ac_cv_f90_mangling="unknown"
  fi

  if test "$ac_cv_f90_mangling" = "unknown"; then
    AC_MSG_ERROR([Failed to determine how F90 mangles linker symbols.])
  fi
  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f cf90_test* conftest*])
AC_LANG_POP(Fortran 90)dnl
])
])# LLNL_F90_NAME_MANGLING


dnl *** file: config/llnl-ac-macros/llnl_find_32bit_signed_int.m4

dnl @synopsis LLNL_FIND_32BIT_SIGNED_INT
dnl
dnl @author Gary Kumfert, LLNL
AC_DEFUN(LLNL_FIND_32BIT_SIGNED_INT,
[AC_CACHE_CHECK(for 32 bit signed int,
 llnl_cv_find_32bit_signed_int, 
 [AC_REQUIRE([LLNL_CHECK_LONG_LONG])
  if test $ac_cv_sizeof_int -eq 4; then
    llnl_cv_find_32bit_signed_int=int;
  elif test $ac_cv_sizeof_short -eq 4; then
    llnl_cv_find_32bit_signed_int=short;
  elif test $ac_cv_sizeof_long -eq 4; then 
    llnl_cv_find_32bit_signed_int=long;
  elif test $ac_cv_sizeof_long_long -eq 4; then
    llnl_cv_find_32bit_signed_int="long long";
  else
    llnl_cv_find_32bit_signed_int=unresolved
  fi
])
 if test "$llnl_cv_find_32bit_signed_int" = "unresolved"; then
   AC_MSG_WARN([Could not identify a suitable 4 byte signed integer type])
 fi
])


dnl *** file: config/llnl-ac-macros/llnl_find_64bit_signed_int.m4

dnl @synopsis LLNL_FIND_64BIT_SIGNED_INT
dnl
dnl @author Gary Kumfert, LLNL
AC_DEFUN(LLNL_FIND_64BIT_SIGNED_INT,
[AC_CACHE_CHECK(for 64 bit signed integer,
 llnl_cv_find_64bit_signed_int, 
 [AC_REQUIRE([LLNL_CHECK_LONG_LONG])
  if test $ac_cv_sizeof_int -eq 8; then
    llnl_cv_find_64bit_signed_int=int;
  elif test $ac_cv_sizeof_short -eq 8; then
    llnl_cv_find_64bit_signed_int=short;
  elif test $ac_cv_sizeof_long -eq 8; then 
    llnl_cv_find_64bit_signed_int=long;
  elif test $ac_cv_sizeof_long_long -eq 8; then
    llnl_cv_find_64bit_signed_int="long long";
  else
    llnl_cv_find_64bit_signed_int="unresolved";
  fi
])
 if test "$llnl_cv_find_64bit_signed_int" = "unresolved"; then
   AC_MSG_WARN([Could not identify a suitable 8 byte signed integer type])
 fi 
])


dnl *** file: config/llnl-ac-macros/llnl_fortran90.m4
# This file is part of Autoconf.                       -*- Autoconf -*-
# Programming languages support.
# Copyright 2000, 2001
# Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
# 02111-1307, USA.
#
# As a special exception, the Free Software Foundation gives unlimited
# permission to copy, distribute and modify the configure scripts that
# are the output of Autoconf.  You need not follow the terms of the GNU
# General Public License when using or distributing such scripts, even
# though portions of the text of Autoconf appear in them.  The GNU
# General Public License (GPL) does govern all other use of the material
# that constitutes the Autoconf program.
#
# Certain portions of the Autoconf source text are designed to be copied
# (in certain cases, depending on the input) into the output of
# Autoconf.  We call these the "data" portions.  The rest of the Autoconf
# source text consists of comments plus executable code that decides which
# of the data portions to output in any given case.  We call these
# comments and executable code the "non-data" portions.  Autoconf never
# copies any of the non-data portions into its output.
#
# This special exception to the GPL applies to versions of Autoconf
# released by the Free Software Foundation.  When you make and
# distribute a modified version of Autoconf, you may extend this special
# exception to the GPL to apply to your modified version as well, *unless*
# your modified version has the potential to copy into its output some
# of the text that was the non-data portion of the version that you started
# with.  (In other words, unless your change moves or copies text from
# the non-data portions to the data portions.)  If your modification has
# such potential, you must delete any notice of this special exception
# to the GPL from your modified version.
#
# Written by Akim Demaille, Christian Marquardt, Martin Wilks (and probably
# many others). 


#
# *********************************************************************** #
# NOTE:  This was renamed from aclang-fortran.m4 to llnl_fortran90.m4 for
# for incorporation into the Babel build since Autoconf/FSF people have
# chosen not to add it to their distribution (as of November 2002).
# We have had to make minor fixes, change some macros to be more like their
# current F77 counterparts, and add a few new F77-like macros (marked "LLNL").
# To get everthing to work, we had to use the "latest" M4 (an alpha version) 
# and automake (1.7).  Which means both automake 1.7 and autoconf 2.54 also 
# had to be rebuilt with the new M4. 
# *********************************************************************** #
#


# Table of Contents:
#
# 1. Language selection
#    and routines to produce programs in a given language.
#  a. Fortran 77 (to be moved from aclang.m4)
#  b. Fortran 90
#  c. Fortran 95
#
# 2. Producing programs in a given language.
#  a. Fortran 77 (to be moved from aclang.m4)
#  b. Fortran 90
#  c. Fortran 95
#
# 3. Looking for a compiler
#    And possibly the associated preprocessor.
#  a. Fortran 77 (to be moved from aclang.m4)
#  b. Fortran 90
#  c. Fortran 95
#
# 4. Compilers' characteristics.
#  a. Fortran 77 (to be moved from aclang.m4)
#  b. Fortran 90
#  c. Fortran 95



## ----------------------- ##
## 1. Language selection.  ##
## ----------------------- ##

# ----------------------------- #
# 1b. The Fortran 90 language.  #
# ----------------------------- #

# AC_LANG(Fortran 90)
# -------------------
m4_define([AC_LANG(Fortran 90)],
[ac_ext=f90
ac_compile='$F90 -c $F90FLAGS conftest.$ac_ext >&AS_MESSAGE_LOG_FD'
ac_link='$F90 -o conftest$ac_exeext $F90FLAGS $LD90FLAGS $LDFLAGS conftest.$ac_ext $LIBS >&AS_MESSAGE_LOG_FD'
ac_compiler_gnu=$ac_cv_f90_compiler_gnu
])

##
##  LLNL:  Added the following to mimic latest for Fortran 77
##
# AC_LANG_FORTRAN90
# -----------------
AU_DEFUN([AC_LANG_FORTRAN90], [AC_LANG(Fortran 90)])


# _AC_LANG_ABBREV(Fortran 90)
# ---------------------------
m4_define([_AC_LANG_ABBREV(Fortran 90)], [f90])


# ----------------------------- #
# 1c. The Fortran 95 language.  #
# ----------------------------- #

# AC_LANG(Fortran 95)
# -------------------
m4_define([AC_LANG(Fortran 95)],
[ac_ext=f95
ac_compile='$F95 -c $F95FLAGS conftest.$ac_ext >&AS_MESSAGE_LOG_FD'
ac_link='$F95 -o conftest$ac_exeext $F95FLAGS $LD95FLAGS $LDFLAGS conftest.$ac_ext $LIBS >&AS_MESSAGE_LOG_FD'
ac_compiler_gnu=$ac_cv_f95_compiler_gnu
])


##
##  LLNL:  Added the following to mimic latest for Fortran 77
##
# AC_LANG_FORTRAN95
# -----------------
AU_DEFUN([AC_LANG_FORTRAN95], [AC_LANG(Fortran 95)])


# _AC_LANG_ABBREV(Fortran 95)
# ---------------------------
m4_define([_AC_LANG_ABBREV(Fortran 95)], [f95])


## ---------------------- ##
## 2.Producing programs.  ##
## ---------------------- ##

# ------------------------ #
# 2b. Fortran 90 sources.  #
# ------------------------ #

# AC_LANG_SOURCE(Fortran 90)(BODY)
# --------------------------------
m4_copy([AC_LANG_SOURCE(Fortran 77)], [AC_LANG_SOURCE(Fortran 90)])


# AC_LANG_PROGRAM(Fortran 90)([PROLOGUE], [BODY])
# -----------------------------------------------
## LLNL - Discarding the PROLOGUE just like F77
##
m4_define([AC_LANG_PROGRAM(Fortran 90)], [
m4_ifval([$1],
       [m4_warn([syntax], [$0: ignoring PROLOGUE: $1])])dnl
program main
$2
end program main
])

# AC_LANG_CALL(Fortran 90)(PROLOGUE, FUNCTION)
# --------------------------------------------
m4_define([AC_LANG_CALL(Fortran 90)],
[AC_LANG_PROGRAM([$1],
[call $2])])


# ------------------------ #
# 2c. Fortran 95 sources.  #
# ------------------------ #

# AC_LANG_SOURCE(Fortran 95)(BODY)
# --------------------------------
m4_copy([AC_LANG_SOURCE(Fortran 90)], [AC_LANG_SOURCE(Fortran 95)])

# AC_LANG_PROGRAM(Fortran 95)([PROLOGUE], [BODY])
# -----------------------------------------------
m4_copy([AC_LANG_PROGRAM(Fortran 90)], [AC_LANG_PROGRAM(Fortran 95)])

# AC_LANG_CALL(Fortran 95)(PROLOGUE, FUNCTION)
# --------------------------------------------
m4_copy([AC_LANG_CALL(Fortran 90)], [AC_LANG_CALL(Fortran 95)])


## -------------------------------------------- ##
## 3. Looking for Compilers and Preprocessors.  ##
## -------------------------------------------- ##

# ----------------------------- #
# 3b. The Fortran 90 compiler.  #
# ----------------------------- #


# AC_LANG_PREPROC(Fortran 90)
# ---------------------------
# Find the Fortran 90 preprocessor.  Must be AC_DEFUN'd to be AC_REQUIRE'able.
AC_DEFUN([AC_LANG_PREPROC(Fortran 90)],
[m4_warn([syntax],
         [$0: No preprocessor defined for ]_AC_LANG)])


# AC_LANG_COMPILER(Fortran 90)
# ----------------------------
# Find the Fortran 90 compiler.  Must be AC_DEFUN'd to be
# AC_REQUIRE'able.
AC_DEFUN([AC_LANG_COMPILER(Fortran 90)],
[AC_REQUIRE([AC_PROG_F90])])

##
## LLNL:  Adding ac_cv_prog_g90 like Fortran 77
##
# ac_cv_prog_g90
# --------------
# We used to name the cache variable this way.
AU_DEFUN([ac_cv_prog_g90],
[ac_cv_f90_compiler_gnu])


# AC_PROG_F90([COMPILERS...])
# ---------------------------
# COMPILERS is a space separated list of Fortran 90 compilers to search
# for.
#
# Compilers are ordered by
#  1. F90, F95
#  2. Good/tested native compilers, bad/untested native compilers
#
# pgf90 is the Portland Group F90 compilers.
# xlf90/xlf95 are IBM (AIX) F90/F95 compilers.
# lf95 is the Lahey-Fujitsu compiler.
# epcf90 is the "Edinburgh Portable Compiler" F90.
# fort is the Compaq Fortran 90 (now 95) compiler for Tru64 and Linux/Alpha.
AC_DEFUN([AC_PROG_F90],
[AC_LANG_PUSH(Fortran 90)dnl
AC_ARG_VAR([F90],      [Fortran 90 compiler command])dnl
AC_ARG_VAR([F90FLAGS], [Fortran 90 compiler flags])dnl
_AC_ARG_VAR_LDFLAGS()dnl
AC_CHECK_TOOLS(F90,
      [m4_default([$1],
                  [f90 xlf90 pgf90 epcf90 f95 xlf95 lf95 fort g95])])

#
# LLNL:  Added to be consistent with F77
# Provide some information about the compiler.
echo "$as_me:__oline__:" \
     "checking for _AC_LANG compiler version" >&AS_MESSAGE_LOG_FD
ac_compiler=`set X $ac_compile; echo $[2]`
_AC_EVAL([$ac_compiler --version </dev/null >&AS_MESSAGE_LOG_FD])
_AC_EVAL([$ac_compiler -v </dev/null >&AS_MESSAGE_LOG_FD])
_AC_EVAL([$ac_compiler -V </dev/null >&AS_MESSAGE_LOG_FD])

m4_expand_once([_AC_COMPILER_EXEEXT])[]dnl
m4_expand_once([_AC_COMPILER_OBJEXT])[]dnl
# If we don't use `.F90' as extension, the preprocessor is not run on the
# input file.
ac_save_ext=$ac_ext
ac_ext=F90
_AC_LANG_COMPILER_GNU
ac_ext=$ac_save_ext
G90=`test $ac_compiler_gnu = yes && echo yes`
AC_LANG_POP(Fortran 90)dnl
])# AC_PROG_F90


##
## LLNL:  Should equiv of F77's AC_PROG_F77_G and AC_PROG_F77_C_O be added
##  to check the use of '-g' and '-c -o' options?

# ----------------------------- #
# 3c. The Fortran 95 compiler.  #
# ----------------------------- #


# AC_LANG_PREPROC(Fortran 95)
# ---------------------------
# Find the Fortran 95 preprocessor.  Must be AC_DEFUN'd to be AC_REQUIRE'able.
AC_DEFUN([AC_LANG_PREPROC(Fortran 95)],
[m4_warn([syntax],
         [$0: No preprocessor defined for ]_AC_LANG)])


# AC_LANG_COMPILER(Fortran 95)
# ----------------------------
# Find the Fortran 95 compiler.  Must be AC_DEFUN'd to be
# AC_REQUIRE'able.
AC_DEFUN([AC_LANG_COMPILER(Fortran 95)],
[AC_REQUIRE([AC_PROG_F95])])

##
## LLNL:  Adding ac_cv_prog_g95 like Fortran 77
##
# ac_cv_prog_g95
# --------------
# We used to name the cache variable this way.
AU_DEFUN([ac_cv_prog_g95],
[ac_cv_f95_compiler_gnu])


# AC_PROG_F95([COMPILERS...])
# ---------------------------
# COMPILERS is a space separated list of Fortran 95 compilers to search
# for.
#
# Compilers are ordered by
#  1. Good/tested native compilers, bad/untested native compilers
#
# xlf95 is the IBM (AIX) F95 compiler.
# lf95 is the Lahey-Fujitsu compiler.
# fort is the Compaq Fortran 90 (now 95) compiler for Tru64 and Linux/Alpha.
AC_DEFUN([AC_PROG_F95],
[AC_LANG_PUSH(Fortran 95)dnl
AC_ARG_VAR([F95],      [Fortran 95 compiler command])dnl
AC_ARG_VAR([F95FLAGS], [Fortran 95 compiler flags])dnl
_AC_ARG_VAR_LDFLAGS()dnl
AC_CHECK_TOOLS(F95,
      [m4_default([$1],
                  [f95 xlf95 lf95 fort g95])])
#
#  LLNL:  Making consistent with F77
# Provide some information about the compiler.
echo "$as_me:__oline__:" \
     "checking for _AC_LANG compiler version" >&AS_MESSAGE_LOG_FD
ac_compiler=`set X $ac_compile; echo $[2]`
_AC_EVAL([$ac_compiler --version </dev/null >&AS_MESSAGE_LOG_FD])
_AC_EVAL([$ac_compiler -v </dev/null >&AS_MESSAGE_LOG_FD])
_AC_EVAL([$ac_compiler -V </dev/null >&AS_MESSAGE_LOG_FD])

m4_expand_once([_AC_COMPILER_EXEEXT])[]dnl
m4_expand_once([_AC_COMPILER_OBJEXT])[]dnl
# If we don't use `.F95' as extension, the preprocessor is not run on the
# input file.
ac_save_ext=$ac_ext
ac_ext=F95
_AC_LANG_COMPILER_GNU
ac_ext=$ac_save_ext
G95=`test $ac_compiler_gnu = yes && echo yes`
AC_LANG_POP(Fortran 95)dnl
])# AC_PROG_F95


##
## LLNL:  Should equiv of F77's AC_PROG_F77_G and AC_PROG_F77_C_O be added
##  to check the use of '-g' and '-c -o' options?


## ------------------------------- ##
## 4. Compilers' characteristics.  ##
## ------------------------------- ##


# ---------------------------------------- #
# 4b. Fortan 90 compiler characteristics.  #
# ---------------------------------------- #


# _AC_PROG_F90_V_OUTPUT([FLAG = $ac_cv_prog_f90_v])
# -------------------------------------------------
# Link a trivial Fortran program, compiling with a verbose output FLAG
# (which default value, $ac_cv_prog_f90_v, is computed by
# _AC_PROG_F90_V), and return the output in $ac_f90_v_output.  This
# output is processed in the way expected by AC_F90_LIBRARY_LDFLAGS,
# so that any link flags that are echoed by the compiler appear as
# space-separated items.
AC_DEFUN([_AC_PROG_F90_V_OUTPUT],
[AC_REQUIRE([AC_PROG_F90])dnl
AC_LANG_PUSH(Fortran 90)dnl

AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])

# Compile and link our simple test program by passing a flag (argument
# 1 to this macro) to the Fortran 90 compiler in order to get
# "verbose" output that we can then parse for the Fortran 90 linker
# flags.
ac_save_F90FLAGS=$F90FLAGS
F90FLAGS="$F90FLAGS m4_default([$1], [$ac_cv_prog_f90_v])"
(eval echo $as_me:__oline__: \"$ac_link\") >&AS_MESSAGE_LOG_FD
ac_f90_v_output=`eval $ac_link AS_MESSAGE_LOG_FD>&1 2>&1 | grep -v 'Driving:'`
echo "$ac_f90_v_output" >&AS_MESSAGE_LOG_FD
F90FLAGS=$ac_save_F90FLAGS

rm -f conftest.*
AC_LANG_POP(Fortran 90)dnl

# If we are using xlf then replace all the commas with spaces.
if echo $ac_f90_v_output | grep xlfentry >/dev/null 2>&1; then
  ac_f90_v_output=`echo $ac_f90_v_output | sed 's/,/ /g'`
fi

# If we are using Cray Fortran then delete quotes.
# Use "\"" instead of '"' for font-lock-mode.
# FIXME: a more general fix for quoted arguments with spaces?
if echo $ac_f90_v_output | grep cft90 >/dev/null 2>&1; then
  ac_f90_v_output=`echo $ac_f90_v_output | sed "s/\"//g"`
fi[]dnl
])# _AC_PROG_F90_V_OUTPUT


# _AC_PROG_F90_V
# --------------
#
# Determine the flag that causes the Fortran 90 compiler to print
# information of library and object files (normally -v)
# Needed for AC_F90_LIBRARY_FLAGS
# Some compilers don't accept -v (Lahey: -verbose, xlf: -V, Fujitsu: -###)
AC_DEFUN([_AC_PROG_F90_V],
[AC_CACHE_CHECK([how to get verbose linking output from $F90],
                [ac_cv_prog_f90_v],
[AC_LANG_ASSERT(Fortran 90)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
[ac_cv_prog_f90_v=
# Try some options frequently used verbose output
for ac_verb in -v -verbose --verbose -V -\#\#\#; do
  _AC_PROG_F90_V_OUTPUT($ac_verb)
  # look for -l* and *.a constructs in the output
  for ac_arg in $ac_f90_v_output; do
     case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]*)
          ac_cv_prog_f90_v=$ac_verb
          break 2 ;;
     esac
  done
done
if test -z "$ac_cv_prog_f90_v"; then
   AC_MSG_WARN([cannot determine how to obtain linking information from $F90])
fi],
                  [AC_MSG_WARN([compilation failed])])
])])# _AC_PROG_F90_V


# AC_F90_LIBRARY_LDFLAGS
# ----------------------
#
# Determine the linker flags (e.g. "-L" and "-l") for the Fortran 90
# intrinsic and run-time libraries that are required to successfully
# link a Fortran 90 program or shared library.  The output variable
# F90LIBS is set to these flags.
#
# This macro is intended to be used in those situations when it is
# necessary to mix, e.g. C++ and Fortran 90, source code into a single
# program or shared library.
#
# For example, if object files from a C++ and Fortran 90 compiler must
# be linked together, then the C++ compiler/linker must be used for
# linking (since special C++-ish things need to happen at link time
# like calling global constructors, instantiating templates, enabling
# exception support, etc.).
#
# However, the Fortran 90 intrinsic and run-time libraries must be
# linked in as well, but the C++ compiler/linker doesn't know how to
# add these Fortran 90 libraries.  Hence, the macro
# "AC_F90_LIBRARY_LDFLAGS" was created to determine these Fortran 90
# libraries.
#
# This macro was copied from the Fortran 77 version by Matthew D. Langston.
AC_DEFUN([AC_F90_LIBRARY_LDFLAGS],
[AC_LANG_PUSH(Fortran 90)dnl
_AC_PROG_F90_V
AC_CACHE_CHECK([for Fortran 90 libraries], ac_cv_f90libs,
[if test "x$F90LIBS" != "x"; then
  ac_cv_f90libs="$F90LIBS" # Let the user override the test.
else

_AC_PROG_F90_V_OUTPUT

ac_cv_f90libs=

# Save positional arguments (if any)
ac_save_positional="$[@]"

set X $ac_f90_v_output
while test $[@%:@] != 1; do
  shift
  ac_arg=$[1]
  case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_f90libs, ,
              ac_cv_f90libs="$ac_cv_f90libs $ac_arg")
          ;;
        -bI:*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_f90libs, ,
             [_AC_LINKER_OPTION([$ac_arg], ac_cv_f90libs)])
          ;;
          # Ignore these flags.
        -lang* | -lcrt0.o | -lc | -lgcc | -LANG:=*)
          ;;
        -lkernel32)
          test x"$CYGWIN" != xyes && ac_cv_f90libs="$ac_cv_f90libs $ac_arg"
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
            _AC_LIST_MEMBER_IF($ac_j, $ac_cv_f90libs, ,
                            [ac_arg="$ac_arg $ac_j"
                             ac_cv_f90libs="$ac_cv_f90libs $ac_j"])
          done
          ;;
        -[[lLR]]*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_f90libs, ,
                          ac_cv_f90libs="$ac_cv_f90libs $ac_arg")
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
      ac_ld_run_path=`echo $ac_f90_v_output |
                        sed -n 's,^.*LD_RUN_PATH *= *\(/[[^ ]]*\).*$,-R\1,p'`
      test "x$ac_ld_run_path" != x &&
        _AC_LINKER_OPTION([$ac_ld_run_path], ac_cv_f90libs)
      ;;
esac
fi # test "x$F90LIBS" = "x"
])
F90LIBS="$ac_cv_f90libs"
AC_SUBST(F90LIBS)
AC_LANG_POP(Fortran 90)dnl
])# AC_F90_LIBRARY_LDFLAGS


##
##  LLNL:  Added F90 Dummy Main.
##
# AC_F90_DUMMY_MAIN([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -----------------------------------------------------------
#
# Detect name of dummy main routine required by the Fortran libraries,
# (if any) and define F90_DUMMY_MAIN to this name (which should be
# used for a dummy declaration, if it is defined).  On some systems,
# linking a C program to the Fortran library does not work unless you
# supply a dummy function called something like MAIN__.
#
# Execute ACTION-IF-NOT-FOUND if no way of successfully linking a C
# program with the F90 libs is found; default to exiting with an error
# message.  Execute ACTION-IF-FOUND if a dummy routine name is needed
# and found or if it is not needed (default to defining F90_DUMMY_MAIN
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
# instead of main().  A separate autoconf test (AC_F90_MAIN) checks
# for the routine to use in this case (since the semantics of the test
# are slightly different).  To link to e.g. purely numerical
# libraries, this is normally not necessary, however, and most C/C++
# programs are reluctant to turn over so much control to Fortran.  =)
#
# The name variants we check for are (in order):
#   MAIN__ (g90, MAIN__ required on some systems; IRIX, MAIN__ optional)
#   MAIN_, __main (SunOS)
#   MAIN _MAIN __MAIN main_ main__ _main (we follow DDD and try these too)
AC_DEFUN([AC_F90_DUMMY_MAIN],
[AC_REQUIRE([AC_F90_LIBRARY_LDFLAGS])dnl
m4_define([_AC_LANG_PROGRAM_C_F90_HOOKS],
[#ifdef F90_DUMMY_MAIN
#  ifdef __cplusplus
     extern "C"
#  endif
   int F90_DUMMY_MAIN() { return 1; }
#endif
])
AC_CACHE_CHECK([for dummy main to link with Fortran 90 libraries],
               ac_cv_f90_dummy_main,
[AC_LANG_PUSH(C)dnl
 ac_f90_dm_save_LIBS=$LIBS
 LIBS="$LIBS $FLIBS"

 # First, try linking without a dummy main:
 AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                [ac_cv_f90_dummy_main=none],
                [ac_cv_f90_dummy_main=unknown])

 if test $ac_cv_f90_dummy_main = unknown; then
   for ac_func in MAIN__ MAIN_ __main MAIN _MAIN __MAIN main_ main__ _main; do
     AC_LINK_IFELSE([AC_LANG_PROGRAM([[@%:@define F90_DUMMY_MAIN $ac_func]])],
                    [ac_cv_f90_dummy_main=$ac_func; break])
   done
 fi
 rm -f conftest*
 LIBS=$ac_f90_dm_save_LIBS
 AC_LANG_POP(C)dnl
])
F90_DUMMY_MAIN=$ac_cv_f90_dummy_main
AS_IF([test "$F90_DUMMY_MAIN" != unknown],
      [m4_default([$1],
[if test $F90_DUMMY_MAIN != none; then
  AC_DEFINE_UNQUOTED([F90_DUMMY_MAIN], $F90_DUMMY_MAIN,
                     [Define to dummy `main' function (if any) required to
                      link to the Fortran 90 libraries.])
fi])],
      [m4_default([$2],
                [AC_MSG_ERROR([linking to Fortran libraries from C fails])])])
])# AC_F90_DUMMY_MAIN


##
##  LLNL:  Added F90 Main.
##
# AC_F90_MAIN
# -----------
# Define F90_MAIN to name of alternate main() function for use with
# the Fortran libraries.  (Typically, the libraries may define their
# own main() to initialize I/O, etcetera, that then call your own
# routine called MAIN__ or whatever.)  See AC_F90_DUMMY_MAIN, above.
# If no such alternate name is found, just define F90_MAIN to main.
#
AC_DEFUN([AC_F90_MAIN],
[AC_REQUIRE([AC_F90_LIBRARY_LDFLAGS])dnl
AC_CACHE_CHECK([for alternate main to link with Fortran 90 libraries],
               ac_cv_f90_main,
[AC_LANG_PUSH(C)dnl
 ac_f90_m_save_LIBS=$LIBS
 LIBS="$LIBS $FLIBS"
 ac_cv_f90_main="main" # default entry point name

 for ac_func in MAIN__ MAIN_ __main MAIN _MAIN __MAIN main_ main__ _main; do
   AC_LINK_IFELSE([AC_LANG_PROGRAM([@%:@undef F90_DUMMY_MAIN
@%:@define main $ac_func])],
                  [ac_cv_f90_main=$ac_func; break])
 done
 rm -f conftest*
 LIBS=$ac_f90_m_save_LIBS
 AC_LANG_POP(C)dnl
])
AC_DEFINE_UNQUOTED([F90_MAIN], $ac_cv_f90_main,
                   [Define to alternate name for `main' routine that is
                    called from a `main' in the Fortran libraries.])
])# AC_F90_MAIN


# _AC_F90_NAME_MANGLING
# ---------------------
# Test for the name mangling scheme used by the Fortran 90 compiler.
#
# Sets ac_cv_f90_mangling. The value contains three fields, separated
# by commas:
#
# lower case / upper case:
#    case translation of the Fortan 90 symbols
# underscore / no underscore:
#    whether the compiler appends "_" to symbol names
# extra underscore / no extra underscore:
#    whether the compiler appends an extra "_" to symbol names already
#    containing at least one underscore
#
AC_DEFUN([_AC_F90_NAME_MANGLING],
[AC_REQUIRE([AC_F90_LIBRARY_LDFLAGS])dnl
AC_CACHE_CHECK([for Fortran 90 name-mangling scheme],
               ac_cv_f90_mangling,
[AC_LANG_PUSH(Fortran 90)dnl
AC_COMPILE_IFELSE(
[subroutine foobar()
return
end
subroutine foo_bar()
return
end],
[mv conftest.$ac_objext cf90_test.$ac_objext

  AC_LANG_PUSH(C)dnl

  ac_save_LIBS=$LIBS
  LIBS="cf90_test.$ac_objext $F90LIBS $LIBS"

  ac_success=no
  for ac_foobar in foobar FOOBAR; do
    for ac_underscore in "" "_"; do
      ac_func="$ac_foobar$ac_underscore"
      AC_TRY_LINK_FUNC($ac_func,
         [ac_success=yes; break 2])
    done
  done

  if test "$ac_success" = "yes"; then
     case $ac_foobar in
        foobar)
           ac_case=lower
           ac_foo_bar=foo_bar
           ;;
        FOOBAR)
           ac_case=upper
           ac_foo_bar=FOO_BAR
           ;;
     esac

     ac_success_extra=no
     for ac_extra in "" "_"; do
        ac_func="$ac_foo_bar$ac_underscore$ac_extra"
        AC_TRY_LINK_FUNC($ac_func,
        [ac_success_extra=yes; break])
     done

     if test "$ac_success_extra" = "yes"; then
        ac_cv_f90_mangling="$ac_case case"
        if test -z "$ac_underscore"; then
           ac_cv_f90_mangling="$ac_cv_f90_mangling, no underscore"
        else
           ac_cv_f90_mangling="$ac_cv_f90_mangling, underscore"
        fi
        if test -z "$ac_extra"; then
           ac_cv_f90_mangling="$ac_cv_f90_mangling, no extra underscore"
        else
           ac_cv_f90_mangling="$ac_cv_f90_mangling, extra underscore"
        fi
      else
        ac_cv_f90_mangling="unknown"
      fi
  else
     ac_cv_f90_mangling="unknown"
  fi

  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f cf90_test* conftest*])
AC_LANG_POP(Fortran 90)dnl
])
])# _AC_F90_NAME_MANGLING

# The replacement is empty.
AU_DEFUN([AC_F90_NAME_MANGLING], [])


# AC_F90_WRAPPERS
# ---------------
# Defines C macros F90_FUNC(name,NAME) and F90_FUNC_(name,NAME) to
# properly mangle the names of C identifiers, and C identifiers with
# underscores, respectively, so that they match the name mangling
# scheme used by the Fortran 90 compiler.
AC_DEFUN([AC_F90_WRAPPERS],
[AC_REQUIRE([_AC_F90_NAME_MANGLING])dnl
AH_TEMPLATE([F90_FUNC],
    [Define to a macro mangling the given C identifier (in lower and upper
     case), which must not contain underscores, for linking with Fortran 90.])dnl
AH_TEMPLATE([F90_FUNC_],
    [As F90_FUNC, but for C identifiers containing underscores.])dnl
case $ac_cv_f90_mangling in
  "lower case, no underscore, no extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [name])
          AC_DEFINE([F90_FUNC_(name,NAME)], [name]) ;;
  "lower case, no underscore, extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [name])
          AC_DEFINE([F90_FUNC_(name,NAME)], [name ## _]) ;;
  "lower case, underscore, no extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [name ## _])
          AC_DEFINE([F90_FUNC_(name,NAME)], [name ## _]) ;;
  "lower case, underscore, extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [name ## _])
          AC_DEFINE([F90_FUNC_(name,NAME)], [name ## __]) ;;
  "upper case, no underscore, no extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [NAME])
          AC_DEFINE([F90_FUNC_(name,NAME)], [NAME]) ;;
  "upper case, no underscore, extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [NAME])
          AC_DEFINE([F90_FUNC_(name,NAME)], [NAME ## _]) ;;
  "upper case, underscore, no extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [NAME ## _])
          AC_DEFINE([F90_FUNC_(name,NAME)], [NAME ## _]) ;;
  "upper case, underscore, extra underscore")
          AC_DEFINE([F90_FUNC(name,NAME)],  [NAME ## _])
          AC_DEFINE([F90_FUNC_(name,NAME)], [NAME ## __]) ;;
  *)
          AC_MSG_WARN([unknown Fortran 90 name-mangling scheme])
          ;;
esac
])# AC_F90_WRAPPERS


# AC_F90_FUNC(NAME, [SHELLVAR = NAME])
# ------------------------------------
# For a Fortran subroutine of given NAME, define a shell variable
# $SHELLVAR to the Fortran 90 mangled name.  If the SHELLVAR
# argument is not supplied, it defaults to NAME.
AC_DEFUN([AC_F90_FUNC],
[AC_REQUIRE([_AC_F90_NAME_MANGLING])dnl
case $ac_cv_f90_mangling in
  upper*) ac_val="m4_toupper([$1])" ;;
  lower*) ac_val="m4_tolower([$1])" ;;
  *)      ac_val="unknown" ;;
esac
case $ac_cv_f90_mangling in *," underscore"*) ac_val="$ac_val"_ ;; esac
m4_if(m4_index([$1],[_]),-1,[],
[case $ac_cv_f90_mangling in *," extra underscore"*) ac_val="$ac_val"_ ;; esac
])
m4_default([$2],[$1])="$ac_val"
])# AC_F90_FUNC


# ---------------------------------------- #
# 4c. Fortan 95 compiler characteristics.  #
# ---------------------------------------- #


# _AC_PROG_F95_V_OUTPUT([FLAG = $ac_cv_prog_f95_v])
# -------------------------------------------------
# Link a trivial Fortran program, compiling with a verbose output FLAG
# (which default value, $ac_cv_prog_f95_v, is computed by
# _AC_PROG_F95_V), and return the output in $ac_f95_v_output.  This
# output is processed in the way expected by AC_F95_LIBRARY_LDFLAGS,
# so that any link flags that are echoed by the compiler appear as
# space-separated items.
AC_DEFUN([_AC_PROG_F95_V_OUTPUT],
[AC_REQUIRE([AC_PROG_F95])dnl
AC_LANG_PUSH(Fortran 95)dnl

AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])

# Compile and link our simple test program by passing a flag (argument
# 1 to this macro) to the Fortran 95 compiler in order to get
# "verbose" output that we can then parse for the Fortran 95 linker
# flags.
ac_save_F95FLAGS=$F95FLAGS
F95FLAGS="$F95FLAGS m4_default([$1], [$ac_cv_prog_f95_v])"
(eval echo $as_me:__oline__: \"$ac_link\") >&AS_MESSAGE_LOG_FD
ac_f95_v_output=`eval $ac_link AS_MESSAGE_LOG_FD>&1 2>&1 | grep -v 'Driving:'`
echo "$ac_f95_v_output" >&AS_MESSAGE_LOG_FD
F95FLAGS=$ac_save_F95FLAGS

rm -f conftest.*
AC_LANG_POP(Fortran 95)dnl

# If we are using xlf then replace all the commas with spaces.
if echo $ac_f95_v_output | grep xlfentry >/dev/null 2>&1; then
  ac_f95_v_output=`echo $ac_f95_v_output | sed 's/,/ /g'`
fi

# If we are using Cray Fortran then delete quotes.
# Use "\"" instead of '"' for font-lock-mode.
# FIXME: a more general fix for quoted arguments with spaces?
if echo $ac_f95_v_output | grep cft95 >/dev/null 2>&1; then
  ac_f95_v_output=`echo $ac_f95_v_output | sed "s/\"//g"`
fi[]dnl
])# _AC_PROG_F95_V_OUTPUT


# _AC_PROG_F95_V
# --------------
#
# Determine the flag that causes the Fortran 95 compiler to print
# information of library and object files (normally -v)
# Needed for AC_F95_LIBRARY_FLAGS
# Some compilers don't accept -v (Lahey: -verbose, xlf: -V, Fujitsu: -###)
AC_DEFUN([_AC_PROG_F95_V],
[AC_CACHE_CHECK([how to get verbose linking output from $F95],
                [ac_cv_prog_f95_v],
[AC_LANG_ASSERT(Fortran 95)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
[ac_cv_prog_f95_v=
# Try some options frequently used verbose output
for ac_verb in -v -verbose --verbose -V -\#\#\#; do
  _AC_PROG_F95_V_OUTPUT($ac_verb)
  # look for -l* and *.a constructs in the output
  for ac_arg in $ac_f95_v_output; do
     case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]*)
          ac_cv_prog_f95_v=$ac_verb
          break 2 ;;
     esac
  done
done
if test -z "$ac_cv_prog_f95_v"; then
   AC_MSG_WARN([cannot determine how to obtain linking information from $F95])
fi],
                  [AC_MSG_WARN([compilation failed])])
])])# _AC_PROG_F95_V


# AC_F95_LIBRARY_LDFLAGS
# ----------------------
#
# Determine the linker flags (e.g. "-L" and "-l") for the Fortran 95
# intrinsic and run-time libraries that are required to successfully
# link a Fortran 95 program or shared library.  The output variable
# F95LIBS is set to these flags.
#
# This macro is intended to be used in those situations when it is
# necessary to mix, e.g. C++ and Fortran 95, source code into a single
# program or shared library.
#
# For example, if object files from a C++ and Fortran 95 compiler must
# be linked together, then the C++ compiler/linker must be used for
# linking (since special C++-ish things need to happen at link time
# like calling global constructors, instantiating templates, enabling
# exception support, etc.).
#
# However, the Fortran 95 intrinsic and run-time libraries must be
# linked in as well, but the C++ compiler/linker doesn't know how to
# add these Fortran 95 libraries.  Hence, the macro
# "AC_F95_LIBRARY_LDFLAGS" was created to determine these Fortran 95
# libraries.
#
# This macro was copied from the Fortran 77 version by Matthew D. Langston.
AC_DEFUN([AC_F95_LIBRARY_LDFLAGS],
[AC_LANG_PUSH(Fortran 95)dnl
_AC_PROG_F95_V
AC_CACHE_CHECK([for Fortran 95 libraries], ac_cv_flibs,
[if test "x$F95LIBS" != "x"; then
  ac_cv_f95libs="$F95LIBS" # Let the user override the test.
else

_AC_PROG_F95_V_OUTPUT

ac_cv_f95libs=

# Save positional arguments (if any)
ac_save_positional="$[@]"

set X $ac_f95_v_output
while test $[@%:@] != 1; do
  shift
  ac_arg=$[1]
  case $ac_arg in
        [[\\/]]*.a | ?:[[\\/]]*.a)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_f95libs, ,
              ac_cv_f95libs="$ac_cv_f95libs $ac_arg")
          ;;
        -bI:*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_f95libs, ,
             [_AC_LINKER_OPTION([$ac_arg], ac_cv_f95libs)])
          ;;
          # Ignore these flags.
        -lang* | -lcrt0.o | -lc | -lgcc | -LANG:=*)
          ;;
        -lkernel32)
          test x"$CYGWIN" != xyes && ac_cv_f95libs="$ac_cv_f95libs $ac_arg"
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
            _AC_LIST_MEMBER_IF($ac_j, $ac_cv_f95libs, ,
                            [ac_arg="$ac_arg $ac_j"
                             ac_cv_f95libs="$ac_cv_f95libs $ac_j"])
          done
          ;;
        -[[lLR]]*)
          _AC_LIST_MEMBER_IF($ac_arg, $ac_cv_f95libs, ,
                          ac_cv_f95libs="$ac_cv_f95libs $ac_arg")
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
      ac_ld_run_path=`echo $ac_f95_v_output |
                        sed -n 's,^.*LD_RUN_PATH *= *\(/[[^ ]]*\).*$,-R\1,p'`
      test "x$ac_ld_run_path" != x &&
        _AC_LINKER_OPTION([$ac_ld_run_path], ac_cv_f95libs)
      ;;
esac
fi # test "x$F95LIBS" = "x"
])
F95LIBS="$ac_cv_f95libs"
AC_SUBST(F95LIBS)
AC_LANG_POP(Fortran 95)dnl
])# AC_F95_LIBRARY_LDFLAGS


##
##  LLNL:  Added F95 Dummy Main.
##
# AC_F95_DUMMY_MAIN([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -----------------------------------------------------------
#
# Detect name of dummy main routine required by the Fortran libraries,
# (if any) and define F95_DUMMY_MAIN to this name (which should be
# used for a dummy declaration, if it is defined).  On some systems,
# linking a C program to the Fortran library does not work unless you
# supply a dummy function called something like MAIN__.
#
# Execute ACTION-IF-NOT-FOUND if no way of successfully linking a C
# program with the F95 libs is found; default to exiting with an error
# message.  Execute ACTION-IF-FOUND if a dummy routine name is needed
# and found or if it is not needed (default to defining F95_DUMMY_MAIN
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
# instead of main().  A separate autoconf test (AC_F95_MAIN) checks
# for the routine to use in this case (since the semantics of the test
# are slightly different).  To link to e.g. purely numerical
# libraries, this is normally not necessary, however, and most C/C++
# programs are reluctant to turn over so much control to Fortran.  =)
#
# The name variants we check for are (in order):
#   MAIN__ (g95, MAIN__ required on some systems; IRIX, MAIN__ optional)
#   MAIN_, __main (SunOS)
#   MAIN _MAIN __MAIN main_ main__ _main (we follow DDD and try these too)
AC_DEFUN([AC_F95_DUMMY_MAIN],
[AC_REQUIRE([AC_F95_LIBRARY_LDFLAGS])dnl
m4_define([_AC_LANG_PROGRAM_C_F95_HOOKS],
[#ifdef F95_DUMMY_MAIN
#  ifdef __cplusplus
     extern "C"
#  endif
   int F95_DUMMY_MAIN() { return 1; }
#endif
])
AC_CACHE_CHECK([for dummy main to link with Fortran 95 libraries],
               ac_cv_f95_dummy_main,
[AC_LANG_PUSH(C)dnl
 ac_f95_dm_save_LIBS=$LIBS
 LIBS="$LIBS $FLIBS"

 # First, try linking without a dummy main:
 AC_LINK_IFELSE([AC_LANG_PROGRAM([], [])],
                [ac_cv_f95_dummy_main=none],
                [ac_cv_f95_dummy_main=unknown])

 if test $ac_cv_f95_dummy_main = unknown; then
   for ac_func in MAIN__ MAIN_ __main MAIN _MAIN __MAIN main_ main__ _main; do
     AC_LINK_IFELSE([AC_LANG_PROGRAM([[@%:@define F95_DUMMY_MAIN $ac_func]])],
                    [ac_cv_f95_dummy_main=$ac_func; break])
   done
 fi
 rm -f conftest*
 LIBS=$ac_f95_dm_save_LIBS
 AC_LANG_POP(C)dnl
])
F95_DUMMY_MAIN=$ac_cv_f95_dummy_main
AS_IF([test "$F95_DUMMY_MAIN" != unknown],
      [m4_default([$1],
[if test $F95_DUMMY_MAIN != none; then
  AC_DEFINE_UNQUOTED([F95_DUMMY_MAIN], $F95_DUMMY_MAIN,
                     [Define to dummy `main' function (if any) required to
                      link to the Fortran 95 libraries.])
fi])],
      [m4_default([$2],
                [AC_MSG_ERROR([linking to Fortran libraries from C fails])])])
])# AC_F95_DUMMY_MAIN


##
##  LLNL:  Added F95 Main.
##
# AC_F95_MAIN
# -----------
# Define F95_MAIN to name of alternate main() function for use with
# the Fortran libraries.  (Typically, the libraries may define their
# own main() to initialize I/O, etcetera, that then call your own
# routine called MAIN__ or whatever.)  See AC_F95_DUMMY_MAIN, above.
# If no such alternate name is found, just define F95_MAIN to main.
#
AC_DEFUN([AC_F95_MAIN],
[AC_REQUIRE([AC_F95_LIBRARY_LDFLAGS])dnl
AC_CACHE_CHECK([for alternate main to link with Fortran 95 libraries],
               ac_cv_f95_main,
[AC_LANG_PUSH(C)dnl
 ac_f95_m_save_LIBS=$LIBS
 LIBS="$LIBS $FLIBS"
 ac_cv_f95_main="main" # default entry point name

 for ac_func in MAIN__ MAIN_ __main MAIN _MAIN __MAIN main_ main__ _main; do
   AC_LINK_IFELSE([AC_LANG_PROGRAM([@%:@undef F95_DUMMY_MAIN
@%:@define main $ac_func])],
                  [ac_cv_f95_main=$ac_func; break])
 done
 rm -f conftest*
 LIBS=$ac_f95_m_save_LIBS
 AC_LANG_POP(C)dnl
])
AC_DEFINE_UNQUOTED([F95_MAIN], $ac_cv_f95_main,
                   [Define to alternate name for `main' routine that is
                    called from a `main' in the Fortran libraries.])
])# AC_F95_MAIN



# _AC_F95_NAME_MANGLING
# ---------------------
# Test for the name mangling scheme used by the Fortran 95 compiler.
#
# Sets ac_cv_f95_mangling. The value contains three fields, separated
# by commas:
#
# lower case / upper case:
#    case translation of the Fortan 95 symbols
# underscore / no underscore:
#    whether the compiler appends "_" to symbol names
# extra underscore / no extra underscore:
#    whether the compiler appends an extra "_" to symbol names already
#    containing at least one underscore
#
AC_DEFUN([_AC_F95_NAME_MANGLING],
[AC_REQUIRE([AC_F95_LIBRARY_LDFLAGS])dnl
AC_CACHE_CHECK([for Fortran 95 name-mangling scheme],
               ac_cv_f95_mangling,
[AC_LANG_PUSH(Fortran 95)dnl
AC_COMPILE_IFELSE(
[subroutine foobar()
return
end
subroutine foo_bar()
return
end],
[mv conftest.$ac_objext cf95_test.$ac_objext

  AC_LANG_PUSH(C)dnl

  ac_save_LIBS=$LIBS
  LIBS="cf95_test.$ac_objext $F95LIBS $LIBS"

  ac_success=no
  for ac_foobar in foobar FOOBAR; do
    for ac_underscore in "" "_"; do
      ac_func="$ac_foobar$ac_underscore"
      AC_TRY_LINK_FUNC($ac_func,
         [ac_success=yes; break 2])
    done
  done

  if test "$ac_success" = "yes"; then
     case $ac_foobar in
        foobar)
           ac_case=lower
           ac_foo_bar=foo_bar
           ;;
        FOOBAR)
           ac_case=upper
           ac_foo_bar=FOO_BAR
           ;;
     esac

     ac_success_extra=no
     for ac_extra in "" "_"; do
        ac_func="$ac_foo_bar$ac_underscore$ac_extra"
        AC_TRY_LINK_FUNC($ac_func,
        [ac_success_extra=yes; break])
     done

     if test "$ac_success_extra" = "yes"; then
        ac_cv_f95_mangling="$ac_case case"
        if test -z "$ac_underscore"; then
           ac_cv_f95_mangling="$ac_cv_f95_mangling, no underscore"
        else
           ac_cv_f95_mangling="$ac_cv_f95_mangling, underscore"
        fi
        if test -z "$ac_extra"; then
           ac_cv_f95_mangling="$ac_cv_f95_mangling, no extra underscore"
        else
           ac_cv_f95_mangling="$ac_cv_f95_mangling, extra underscore"
        fi
      else
        ac_cv_f95_mangling="unknown"
      fi
  else
     ac_cv_f95_mangling="unknown"
  fi

  LIBS=$ac_save_LIBS
  AC_LANG_POP(C)dnl
  rm -f cf95_test* conftest*])
AC_LANG_POP(Fortran 95)dnl
])
])# _AC_F95_NAME_MANGLING

# The replacement is empty.
AU_DEFUN([AC_F95_NAME_MANGLING], [])


# AC_F95_WRAPPERS
# ---------------
# Defines C macros F95_FUNC(name,NAME) and F95_FUNC_(name,NAME) to
# properly mangle the names of C identifiers, and C identifiers with
# underscores, respectively, so that they match the name mangling
# scheme used by the Fortran 95 compiler.
AC_DEFUN([AC_F95_WRAPPERS],
[AC_REQUIRE([_AC_F95_NAME_MANGLING])dnl
AH_TEMPLATE([F95_FUNC],
    [Define to a macro mangling the given C identifier (in lower and upper
     case), which must not contain underscores, for linking with Fortran 95.])dnl
AH_TEMPLATE([F95_FUNC_],
    [As F95_FUNC, but for C identifiers containing underscores.])dnl
case $ac_cv_f95_mangling in
  "lower case, no underscore, no extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [name])
          AC_DEFINE([F95_FUNC_(name,NAME)], [name]) ;;
  "lower case, no underscore, extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [name])
          AC_DEFINE([F95_FUNC_(name,NAME)], [name ## _]) ;;
  "lower case, underscore, no extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [name ## _])
          AC_DEFINE([F95_FUNC_(name,NAME)], [name ## _]) ;;
  "lower case, underscore, extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [name ## _])
          AC_DEFINE([F95_FUNC_(name,NAME)], [name ## __]) ;;
  "upper case, no underscore, no extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [NAME])
          AC_DEFINE([F95_FUNC_(name,NAME)], [NAME]) ;;
  "upper case, no underscore, extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [NAME])
          AC_DEFINE([F95_FUNC_(name,NAME)], [NAME ## _]) ;;
  "upper case, underscore, no extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [NAME ## _])
          AC_DEFINE([F95_FUNC_(name,NAME)], [NAME ## _]) ;;
  "upper case, underscore, extra underscore")
          AC_DEFINE([F95_FUNC(name,NAME)],  [NAME ## _])
          AC_DEFINE([F95_FUNC_(name,NAME)], [NAME ## __]) ;;
  *)
          AC_MSG_WARN([unknown Fortran 95 name-mangling scheme])
          ;;
esac
])# AC_F95_WRAPPERS


# AC_F95_FUNC(NAME, [SHELLVAR = NAME])
# ------------------------------------
# For a Fortran subroutine of given NAME, define a shell variable
# $SHELLVAR to the Fortran 95 mangled name.  If the SHELLVAR
# argument is not supplied, it defaults to NAME.
AC_DEFUN([AC_F95_FUNC],
[AC_REQUIRE([_AC_F95_NAME_MANGLING])dnl
case $ac_cv_f95_mangling in
  upper*) ac_val="m4_toupper([$1])" ;;
  lower*) ac_val="m4_tolower([$1])" ;;
  *)      ac_val="unknown" ;;
esac
case $ac_cv_f95_mangling in *," underscore"*) ac_val="$ac_val"_ ;; esac
m4_if(m4_index([$1],[_]),-1,[],
[case $ac_cv_f95_mangling in *," extra underscore"*) ac_val="$ac_val"_ ;; esac
])
m4_default([$2],[$1])="$ac_val"
])# AC_F95_FUNC



dnl *** file: config/llnl-ac-macros/llnl_func_drand_fortyeight.m4

dnl 
dnl @synopsis LLNL_FUNC_DRAND_FORTYEIGHT
dnl 
dnl If the C compiler has drand48() define `LLNL_HAVE_FUNC_DRAND_FORTYEIGHT'
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
AC_DEFUN(LLNL_FUNC_DRAND_FORTYEIGHT,
[
AC_MSG_CHECKING([if drand48 is available])
AC_CACHE_VAL(llnl_cv_have_drand_fortyeight,
[
AC_LANG_SAVE
AC_LANG_C
AC_TRY_COMPILE([#include <stdlib.h>],
[ double d = drand48();],
llnl_cv_have_drand_fortyeight=yes,
llnl_cv_have_drand_fortyeight=no)
AC_LANG_RESTORE
])
AC_MSG_RESULT($llnl_cv_have_drand_fortyeight)
if test "$llnl_cv_have_drand_fortyeight" = yes; then
AC_DEFINE(HAVE_FUNCTION_DRAND48,,[define if drand48() is available])
fi])	


dnl *** file: config/llnl-ac-macros/llnl_lib_fmain.m4

dnl 
dnl @synopsis LLNL_LIB_FMAIN
dnl 
dnl Finds the "main" function if the driver is written in fortran
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
dnl
AC_DEFUN(LLNL_LIB_FMAIN,[
AC_REQUIRE([AC_PROG_F77])
AC_CACHE_CHECK(if $CC linker needs a special library for $F77 main, llnl_lib_fmain, [
echo "      END" > conftest.f
foutput=`${F77} -v -o conftest conftest.f 2>&1`
xlf_p=`echo $foutput | grep xlfentry`
if test -n "$xlf_p"; then
  foutput=`echo $foutput | sed 's/,/ /g'`
fi
fmain=no
for arg in $foutput; do
  case "$arg" in
    *for_main.o)
      if test -e $arg; then 
        found=true
        fmain="$arg"
      fi
    ;;
  esac
done
llnl_lib_fmain="$fmain"
if test "X$llnl_lib_fmain" != "Xno" ; then 
  FMAIN="$llnl_lib_fmain"
else
  FMAIN=
fi
rm -f conftest.f conftest
])
AC_SUBST(FMAIN)
])


dnl *** file: config/llnl-ac-macros/llnl_lib_f90main.m4

dnl 
dnl @synopsis LLNL_LIB_F90MAIN
dnl 
dnl Finds the "main" function if the driver is written in fortran
dnl
dnl @version 
dnl @author 
dnl
dnl Note:  Clone of F77 version but tailored to pgf90 needs.
dnl

AC_DEFUN(LLNL_LIB_F90MAIN,[
AC_REQUIRE([AC_PROG_F90])
AC_CACHE_CHECK(if $CC linker needs a special library for $F90 main, llnl_lib_f90main, [
echo "END" > conftest.f90
foutput=`${F90} -v -o conftest conftest.f90 2>&1`
fmain=`echo $foutput | grep f90main`
if test -n "$fmain"; then
  foutput=`echo $foutput | sed 's/,/ /g'`
fi
f90main=no
for arg in $foutput; do
  case "$arg" in
    *f90main.o)
      if test -e $arg; then 
        found=true
        f90main="$arg"
      fi
    ;;
  esac
done
llnl_lib_f90main="$f90main"
if test "X$llnl_lib_f90main" != "Xno" ; then 
  F90MAIN="$llnl_lib_f90main"
else
  F90MAIN=
fi
rm -f conftest.f90 conftest
])
AC_SUBST(F90MAIN)
])


dnl *** file: config/llnl-ac-macros/llnl_prevent_cross_compilation.m4

dnl @synopsis LLNL_PREVENT_CROSS_COMPILATION
dnl  
dnl If compilers fail, they assume cross compilation.
dnl This macro makes turns that assumption to a failure
dnl
dnl @author Gary Kumfert
AC_DEFUN(LLNL_PREVENT_CROSS_COMPILATION,[
if test "X$cross_compiling" = "Xyes"; then
  AC_MSG_ERROR([Compiler installation problem - could not run compilers...])
fi
])


dnl *** file: config/llnl-ac-macros/llnl_prevent_unholy_gnu_nongnu_mix.m4

dnl @synopsis LLNL_PREVENT_UNHOLY_GNU_NONGNU_MIX
dnl
dnl Check for unholy mixture of GNU and non-GNU compilers/linkers/etc
dnl on certain platforms.  Linux is more tolerable than others.
dnl
dnl @author Gary Kumfert 
AC_DEFUN(LLNL_PREVENT_UNHOLY_GNU_NONGNU_MIX,[
  AC_REQUIRE([AC_CANONICAL_HOST])dnl
  AC_REQUIRE([AC_PROG_CPP])dnl
case $host in 
 *linux* | *Linux*) 
	if test "X$GCC" != "X$GXX"; then
	   AC_MSG_WARN([You are mixing GNU and non-GNU C and C++ compilers
			CC=$CC (GNU: ${GCC:-no})
			CXX=$CXX (GNU: ${GXX:-no})
		But this is a linux system, so it may be okay... you have been warned])
	fi 
	;;
 *)
	if test "X$GCC" != "X$GXX"; then
	  AC_MSG_ERROR([You are mixing GNU and non-GNU C and C++ compilers
			CC=$CC (GNU: ${GCC:-no})
			CXX=$CXX (GNU: ${GXX:-no})
		Please adjust your path to pick up the correct compilers or set
		environment variables CC and CXX before running configure...])
	fi
	if test "X$with_gnu_ld" = "Xyes"; then
	  if test "X$GCC" != "Xyes"; then
	    AC_MSG_ERROR([You are mixing a GNU linker with non-GNU compilers
			LD=$LD (GNU: $with_gnu_ld)
			CC=$CC (GNU: ${GCC:-no})
			CXX=$CXX (GNU: ${GXX:-no})
		Please! adjust your path to pick up the correct compilers or set
		environment variables LD, CC, and CXX before running configure...])
	  fi
	fi
	;;
esac
])


dnl *** file: config/llnl-ac-macros/llnl_prog_jdb.m4
dnl
dnl @synopsis LLNL_PROG_JDB
dnl
dnl @author ?
AC_DEFUN([LLNL_PROG_JDB],[
AC_REQUIRE([AC_EXEEXT])dnl
if test "x$JAVAPREFIX" = x; then
        test "x$JDB" = x && AC_CHECK_PROGS(JDB, jdb$EXEEXT)
else
        test "x$JDB" = x && AC_CHECK_PROGS(JDB, jdb, $JAVAPREFIX)
fi
test "x$JDB" = x && AC_MSG_ERROR([no acceptable jdb program found in \$PATH])
AC_PROVIDE([$0])dnl
])


dnl *** file: config/llnl-ac-macros/llnl_prog_javah.m4


dnl dnl @synopsis LLNL_PROG_JAVAH
dnl
dnl LLNL_PROG_JAVAH tests the availability of the javah header generator
dnl and looks for the jni.h header file. If available, JAVAH is set to
dnl the full path of javah.  Unlike Luc's implementation, this doesn't
dnl update CPPFLAGS.  Instead it defines JNI_INCLUDES.
dnl
dnl @author Luc Maisonobe
dnl @version $Id$
dnl
AC_DEFUN([LLNL_PROG_JAVAH],[
AC_REQUIRE([AC_CANONICAL_HOST])dnl
AC_REQUIRE([AC_PROG_CPP])dnl
AC_PATH_PROG(JAVAH,javah)
if test x"`eval 'echo $ac_cv_path_JAVAH'`" != x ; then
  ac_save_CPPFLAGS="$CPPFLAGS"
  if test -n "$JNI_INCLUDES"; then
    CPPFLAGS="$ac_save_CPPFLAGS $JNI_INCLUDES"
  fi 
  AC_TRY_CPP([#include <jni.h>],,[
changequote(, )dnl
    ac_dir=`echo $ac_cv_path_JAVAH | sed 's,\(.*\)/[^/]*/[^/]*$,\1/include,'`
    ac_machdep=`echo $build_os | sed 's,[-0-9].*,,' | sed 's,cygwin,win32,'`
changequote([, ])dnl
    JNI_INCLUDES="$JNI_INCLUDES -I$ac_dir -I$ac_dir/$ac_machdep"
    CPPFLAGS="$ac_save_CPPFLAGS $JNI_INCLUDES"
    AC_TRY_CPP([#include <jni.h>],,[
               AC_MSG_WARN([unable to include <jni.h>])
	       JNI_INCLUDES=])
    ])
    CPPFLAGS="$ac_save_CPPFLAGS"
    AC_SUBST(JNI_INCLUDES)
fi])






dnl *** file: config/llnl-ac-macros/llnl_prog_python.m4

dnl @synopsis LLNL_PROG_PYTHON
dnl
dnl LLNL_PROG_PYTHON tests for an existing Python interpreter. It uses
dnl the environment variable PYTHON and then tries to find python
dnl in standard places.
dnl
dnl @author ?
AC_DEFUN([LLNL_PROG_PYTHON],[
  AC_REQUIRE([AC_EXEEXT])dnl
  AC_CHECK_PROGS(PYTHON, python$EXEEXT)
  if test "x$PYTHON" = x; then
    AC_MSG_WARN([Not building Python support])
  fi
])


dnl *** file: config/llnl-ac-macros/llnl_python_library.m4
dnl @synopsis LLNL_PYTHON_LIBRARY 
dnl
dnl @author ?
AC_DEFUN([LLNL_PYTHON_LIBRARY],[
  AC_REQUIRE([LLNL_PROG_PYTHON])dnl

  if test "X$PYTHON" != "X"; then
    AC_CACHE_CHECK(for Python version, llnl_cv_python_version, [
      llnl_cv_python_version=`$PYTHON -c 'import sys; print sys.version' | sed '1s/^\(...\).*/\1/g;1q'`
    ])
    AC_CACHE_CHECK(for Python library path, llnl_cv_python_library, [
      llnl_python_prefix=`$PYTHON -c 'import sys; print sys.prefix'`
      llnl_cv_python_library="$llnl_python_prefix/lib/python$llnl_cv_python_version"
    ])
    AC_CACHE_CHECK(for Python include path, llnl_cv_python_include, [
      llnl_python_prefix=`$PYTHON -c 'import sys; print sys.prefix'`
      llnl_cv_python_include="$llnl_python_prefix/include/python$llnl_cv_python_version"
    ])
  fi

  AC_DEFINE_UNQUOTED(PYTHON_VERSION,"$llnl_cv_python_version",[A string indicating the Python version number])
  PYTHONLIB="$llnl_cv_python_library"
  PYTHONINC="$llnl_cv_python_include"
  PYTHON_VERSION="$llnl_cv_python_version"
  AC_SUBST(PYTHONLIB)
  AC_SUBST(PYTHONINC)
  AC_SUBST(PYTHON_VERSION)
])


dnl *** file: config/llnl-ac-macros/llnl_python_numeric.m4
dnl @synopsis LLNL_PYTHON_NUMERIC
dnl
dnl @author ?
AC_DEFUN([LLNL_PYTHON_NUMERIC],[
  AC_REQUIRE([LLNL_PROG_PYTHON])dnl
  AC_REQUIRE([LLNL_PYTHON_LIBRARY])dnl
  AC_CACHE_CHECK(for Numerical Python, llnl_cv_python_numerical, [
    llnl_cv_python_numerical=no
    if test "X$PYTHON" != "X"; then
      if AC_TRY_COMMAND($PYTHON -c "import Numeric") > /dev/null 2>&1; then
        if test -f $llnl_cv_python_include/Numeric/arrayobject.h; then
          llnl_cv_python_numerical=yes
        fi
      fi
    fi
  ])
])


dnl *** file: config/llnl-ac-macros/llnl_python_shared_library.m4
dnl
dnl @synopsis LLNL_PYTHON_SHARED_LIBRARY
dnl
dnl @author ?

AC_DEFUN([LLNL_PYTHON_SHARED_LIBRARY],[
  AC_REQUIRE([AC_LTDL_SHLIBEXT])dnl
  AC_REQUIRE([LLNL_PYTHON_LIBRARY])dnl
  AC_REQUIRE([AC_LTDL_SHLIBPATH])dnl
  AC_MSG_CHECKING([if Python shared library is available])

  SHARED_LIB_VAR=${libltdl_cv_shlibpath_var}
  AC_SUBST(SHARED_LIB_VAR)

  llnl_python_shared_library_found=no

  case "$target_os" in
    cygwin*)
      llnl_python_shared_library="libpython$llnl_cv_python_version.dll"
      ;;
    *)
      llnl_python_shared_library="libpython$llnl_cv_python_version$libltdl_cv_shlibext"
      ;;
  esac

  llnl_python_shared_lib_path=`env | grep "^${libltdl_cv_shlibpath_var}=" | sed "s/^${libltdl_cv_shlibpath_var}=//"`
  for f in `echo $llnl_python_shared_lib_path | sed 's/:\|;/  /g'` $llnl_cv_python_library/config /bin; do
    if test -f "$f/$llnl_python_shared_library"; then
      llnl_python_shared_library_found=yes
      llnl_python_shared_library="$f/$llnl_python_shared_library"
      llnl_python_shared_library_dir="$f"
      break
    fi
  done

  if test "$llnl_python_shared_library_found" = "yes"; then
    AC_DEFINE_UNQUOTED(PYTHON_SHARED_LIBRARY,"$llnl_python_shared_library",[Fully qualified string name of the Python shared library])
    AC_DEFINE_UNQUOTED(PYTHON_SHARED_LIBRARY_DIR,"$llnl_python_shared_library_dir",[Directory of the Python shared library])
    PYTHON_SHARED_LIBRARY="$llnl_python_shared_library"
    PYTHON_SHARED_LIBRARY_DIR="$llnl_python_shared_library_dir"
    AC_SUBST(PYTHON_SHARED_LIBRARY)
    AC_SUBST(PYTHON_SHARED_LIBRARY_DIR)
    AC_MSG_RESULT($llnl_python_shared_library)
  else
    AC_MSG_RESULT(no)
  fi
])


dnl *** file: config/llnl-ac-macros/llnl_sort_flibs.m4

dnl 
dnl @synopsis LLNL_SORT_FLIBS
dnl 
dnl With certain Fortran compilers, the FLIBS macro can be out of order.
dnl This macros moves all the arguments beginning with "-l" at the end
dnl but does not alter the relative ordering of "-l" arguments and non-"-l" 
dnl arguments; otherwise,
dnl   If the answer is yes, 
dnl     it defines AR_CXX=$CXX, ARFLAGS_CXX=-xar, and RANLIB_CXX=echo
dnl   otherwise AR_CXX=ar, ARFLAGS_CXX=cuv, RANLIB_CXX=ranlib
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
dnl
AC_DEFUN(LLNL_SORT_FLIBS,[
AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])
flibs1=
flibs2=
for arg in $FLIBS; do
  arg1=
  arg2=
  case "$arg" in 
    -l*)
      arg2=$arg
      ;;
    /*.a)
      arg1=-L`dirname $arg`
      arg2=`basename $arg .a`
      arg2=`echo $arg2 | sed 's/^lib/-l'/'`
      ;;
    /*.so)
      arg1=-L`dirname $arg`
      arg2=`basename $arg .so`
      arg2=`echo $arg2 | sed 's/^lib/-l'/'`
      ;;
    *)
      arg1=$arg
      ;;
  esac; 
  if test -n "$arg1"; then
    exists=false
    for f in $flibs1; do
      if test x$arg1 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      flibs1="$flibs1 $arg1"
    fi
  fi
  if test -n "$arg2"; then
    exists=false
    for f in $flibs2; do
      if test x$arg2 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      flibs2="$flibs2 $arg2"
    fi
  fi
done
FLIBS="$flibs1 $flibs2"
AC_SUBST(FLIBS)
])


dnl *** file: config/llnl-ac-macros/llnl_sort_f90libs.m4

dnl 
dnl @synopsis LLNL_SORT_F90LIBS
dnl 
dnl With certain Fortran compilers, the libs macro can be out of order.
dnl This macros moves all the arguments beginning with "-l" at the end
dnl but does not alter the relative ordering of "-l" arguments and non-"-l" 
dnl arguments; otherwise, 
dnl   If the answer is yes, 
dnl     it defines AR_CXX=$CXX, ARFLAGS_CXX=-xar, and RANLIB_CXX=echo
dnl   otherwise AR_CXX=ar, ARFLAGS_CXX=cuv, RANLIB_CXX=ranlib
dnl
dnl @version 
dnl @author 
dnl
dnl Note:  Clone of F77 version.
dnl

AC_DEFUN(LLNL_SORT_F90LIBS,[
AC_REQUIRE([AC_F90_LIBRARY_LDFLAGS])
f90libs1=
f90libs2=
for arg in $F90LIBS; do
  arg1=
  arg2=
  case "$arg" in 
    -l*)
      arg2=$arg
      ;;
    /*.a)
      arg1=-L`dirname $arg`
      arg2=`basename $arg .a`
      arg2=`echo $arg2 | sed 's/^lib/-l'/'`
      ;;
    /*.so)
      arg1=-L`dirname $arg`
      arg2=`basename $arg .so`
      arg2=`echo $arg2 | sed 's/^lib/-l'/'`
      ;;
    *)
      arg1=$arg
      ;;
  esac; 
  if test -n "$arg1"; then
    exists=false
    for f in $f90libs1; do
      if test x$arg1 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      f90libs1="$f90libs1 $arg1"
    fi
  fi
  if test -n "$arg2"; then
    exists=false
    for f in $f90libs2; do
      if test x$arg2 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      f90libs2="$f90libs2 $arg2"
    fi
  fi
done
F90LIBS="$f90libs1 $f90libs2"
AC_SUBST(F90LIBS)
])


dnl *** file: config/llnl-ac-macros/llnl_confirm_babel_c_support.m4
dnl
dnl @synopsis LLNL_CONFIRM_BABEL_C_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_C_SUPPORT], [
  ############################################################
  #
  # C Compiler
  #
  AC_PROG_CC
  # a. Libraries (existence)
  # b. Header Files.
  AC_HEADER_DIRENT
  AC_HEADER_STDC
  AC_CHECK_HEADERS([inttypes.h malloc.h memory.h stddef.h stdlib.h string.h strings.h unistd.h ctype.h sys/stat.h sys/types.h])
  # c. Typedefs, Structs, Compiler Characteristics
  AC_C_CONST
  AC_TYPE_SIZE_T
  AC_CHECK_TYPES([ptrdiff_t])
  AC_CHECK_SIZEOF(short,2)
  AC_CHECK_SIZEOF(int,4)
  AC_CHECK_SIZEOF(long,8)
  LLNL_CHECK_LONG_LONG
  AC_CHECK_SIZEOF(long long,8)
  LLNL_FIND_32BIT_SIGNED_INT
  LLNL_CHECK_INT32_T
  LLNL_FIND_64BIT_SIGNED_INT
  LLNL_CHECK_INT64_T
  AC_CHECK_SIZEOF(void *,4)
  ACX_C_RESTRICT
  # d. Specific Library Functions.
  # AC_FUNC_MALLOC #define's malloc to rpl_malloc if malloc(0)!=NULL, not all that useful
	           # and actually makes life tough on AIX.
  AC_FUNC_STAT
  AC_CHECK_FUNCS([atexit getcwd memset strchr strdup strrchr])
])


dnl *** file: config/llnl-ac-macros/llnl_confirm_babel_f77_support.m4
dnl
dnl @synopsis LLNL_CONFIRM_BABEL_F77_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for F77 is enabled:
dnl     the cpp macro FORTRAN_DISABLED is undefined
dnl     the automake conditional SUPPORT_FORTRAN is true
dnl
dnl  If Babel support for F77 is disabled:
dnl     the cpp macro FORTRAN_DISABLED is defined as true
dnl     the automake conditional SUPPORT_FORTRAN is false
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_F77_SUPPORT], [
  # fortran77 support is enabled by default if $with_fortran77 is not set.
  if test -z "$with_fortran77"; then
    with_fortran77="yes";
  fi
  # allow fortran77 support to be overridden by the command line.
  AC_ARG_WITH(fortran77, [  --without-fortran77       disable fortran77 support])
  if test "X$with_fortran77" != "Xno"; then
    AC_PROG_F77
    # 5.a. Libraries (existence)
    LLNL_LIB_FMAIN
    AC_F77_LIBRARY_LDFLAGS
    AC_F77_DUMMY_MAIN
    LLNL_SORT_FLIBS
    LLNL_F77_NAME_MANGLING
    LLNL_F77_C_CONFIG
  fi
  AM_CONDITIONAL(SUPPORT_FORTRAN77, test "X$with_fortran77" != "Xno")
  if test "X$with_fortran77" = "Xno"; then
    AC_DEFINE(FORTRAN77_DISABLED, 1, [If defined, Fortran support was disabled at configure time])
    msgs="$msgs
	  Fortran77 disabled by request";
  else
    msgs="$msgs
	  Fortran77 enabled.";
  fi 
])


dnl *** file: config/llnl-ac-macros/llnl_confirm_babel_f90_support.m4
dnl
dnl @synopsis LLNL_CONFIRM_BABEL_F90_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for F90 is enabled:
dnl     the cpp macro FORTRAN90_DISABLED is undefined
dnl     the automake conditional SUPPORT_FORTRAN90 is true
dnl
dnl  If Babel support for F90 is disabled:
dnl     the cpp macro FORTRAN90_DISABLED is defined as true
dnl     the automake conditional SUPPORT_FORTRAN90 is false
dnl
dnl  @author 
dnl
dnl  Note:  Clone of F77 version.

AC_DEFUN([LLNL_CONFIRM_BABEL_F90_SUPPORT], [
  # fortran90 support is enabled by default if $with_fortran90 is not set.
  if test -z "$with_fortran90"; then
    with_fortran90="yes";
  fi
  # allow fortran90 support to be overridden by the command line.
  AC_ARG_WITH(fortran90, [  --without-fortran90       disable fortran90 support])
  if test "X$with_fortran90" != "Xno"; then
    AC_PROG_F90
    if test -n "$F90"; then 
        # 5.a. Libraries (existence)
        LLNL_LIB_F90MAIN
        AC_F90_LIBRARY_LDFLAGS
        AC_F90_DUMMY_MAIN
        LLNL_SORT_F90LIBS
        LLNL_F90_NAME_MANGLING
        LLNL_F90_C_CONFIG
    else
	AC_WARN([Disabling F90 Support])
        with_fortran90="broken"	
    fi
  fi
  if test "X$with_fortran90" = "Xno"; then
    msgs="$msgs
	  Fortran90 disabled by request.";
  elif test "X$with_fortran90" = "Xyes"; then
    msgs="$msgs
	  Fortran90 enabled.";
  else
    msgs="$msgs
	  Fortran90 disabled against user request: no viable compiler found.";
  fi 
  if test "X$with_fortran90" != "Xyes"; then
    AC_DEFINE(FORTRAN90_DISABLED, 1, [If defined, F90 support was disabled at configure time])
  fi
  AM_CONDITIONAL(SUPPORT_FORTRAN90, test "X$with_fortran90" = "Xyes")
])


dnl *** file: config/llnl-ac-macros/llnl_confirm_babel_cxx_support.m4
dnl
dnl @synopsis LLNL_CONFIRM_BABEL_CXX_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for CXX is enabled:
dnl     the cpp macro CXX_DISABLED is undefined
dnl     the automake conditional SUPPORT_CXX is true
dnl
dnl  If Babel support for CXX is disabled:
dnl     the cpp macro CXX_DISABLED is defined as true
dnl     the automake conditional SUPPORT_CXX is false
dnl
dnl  @author Gary Kumfert

dnl this is broken into two tests 'cause ac_cxx_namespaces
dnl consistently gets placed *before* ac_prog_cxx otherwise.
dnl We have to prevent this at all costs!

AC_DEFUN([LLNL_CONFIRM_BABEL_CXX_SUPPORT],[
  if test -z "$CCC"; then
    CCC="g++ KCC CC xlC"
  fi
  # cxx support is enabled by default if $with_cxx is not set.
  if test -z "$with_cxx"; then
    with_cxx="yes";
  fi
  # allow cxx support to be overridden by the command line.
  AC_ARG_WITH(cxx,     [  --without-cxx           disable C++ support])
  if test "X$with_cxx" = "Xno"; then
    AC_MSG_ERROR([Sorry, this package cannot work without C++ enabled.])
  fi
  AC_PROG_CXX
])

AC_DEFUN([LLNL_CONFIRM_BABEL_CXX_SUPPORT2], [
  AC_REQUIRE([LLNL_CONFIRM_BABEL_CXX_SUPPORT])
  if test -n "$CXX"; then
    # 6.a. Libraries (existence) 
    LLNL_CXX_LIBRARY_LDFLAGS
    # 6.b. Header Files
    LLNL_CXX_OLD_HEADER_SUFFIX
    AC_CXX_HAVE_STD
    AC_CXX_HAVE_STL
    AC_CXX_HAVE_NUMERIC_LIMITS
    AC_CXX_COMPLEX_MATH_IN_NAMESPACE_STD
    AC_CXX_HAVE_COMPLEX
    AC_CXX_HAVE_COMPLEX_MATH1
    AC_CXX_HAVE_COMPLEX_MATH2
    AC_CXX_HAVE_IEEE_MATH
  fi
  AM_CONDITIONAL(SUPPORT_CXX, test "X$with_cxx" != "Xno")
  if test "X$with_cxx" = "Xno"; then
    AC_DEFINE(CXX_DISABLED, 1, [If defined, C++ support was disabled at configure time])
    msgs="$msgs 
  	  C++ disabled by request"
  else
    msgs="$msgs
  	  C++ enabled.";
  fi
])


dnl *** file: config/llnl-ac-macros/llnl_confirm_babel_python_support.m4
dnl
dnl @synopsis LLNL_CONFIRM_BABEL_PYTHON_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for PYTHON is enabled:
dnl     the cpp macro PYTHON_DISABLED is undefined
dnl     the automake conditional SUPPORT_PYTHON is true
dnl
dnl  If Babel support for PYTHON is disabled:
dnl     the cpp macro PYTHON_DISABLED is defined as true
dnl     the automake conditional SUPPORT_PYTHON is false
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_PYTHON_SUPPORT], [
  if test -z "$with_python"; then
    with_python=yes
  fi
  AC_ARG_WITH(python,  [  --without-python        disable python support])
  if test "X$with_python" != "Xno"; then
    LLNL_PYTHON_LIBRARY
    LLNL_PYTHON_NUMERIC
    LLNL_PYTHON_SHARED_LIBRARY
    if (test "X$llnl_cv_python_numerical" != "Xyes") || (test "X$enable_shared" = "Xno"); then
       with_python=no;
       AC_MSG_WARN([Configuration for Python failed.  Support for Python disabled!])
       msgs="$msgs
  	  Python support disabled against request, shared libs disabled or NumPy not found."
    elif test "X$llnl_python_shared_library_found" != "Xyes"; then
       AC_MSG_WARN([No Python shared library found.  Support for server-side Python disabled!])
       msgs="$msgs
  	  Server-side Python support disabled against request, can only do client side when no libpython.so found".
    else
       msgs="$msgs
  	  Python enabled.";
    fi
  else
    msgs="$msgs 
  	  Python support disabled by request"
  fi
  # support python in general?
  AM_CONDITIONAL(SUPPORT_PYTHON, test "X$with_python" != "Xno")
  if test "X$with_python" = "Xno"; then
    AC_DEFINE(PYTHON_DISABLED, 1, [If defined, Python support was disabled at configure time])
  fi 
  # support server-side python in particular
  AM_CONDITIONAL(SERVER_PYTHON, (test "X$with_python" != "Xno") && (test "X$llnl_python_shared_library_found" = "Xyes"))
  if (test "X$with_python" = "Xno") || (test "X$llnl_python_shared_library_found" != "Xyes"); then
    AC_DEFINE(PYTHON_SERVER_DISABLED, 1, [If defined, server-side Python support was disabled at configure time])
  fi;
])


dnl *** file: config/llnl-ac-macros/llnl_confirm_babel_java_support.m4
dnl
dnl @synopsis LLNL_CONFIRM_BABEL_JAVA_SUPPORT
dnl
dnl  This is a meta-command that orchestrates a bunch of sub-checks.
dnl  I made it a separate M4 Macro to make synchronization between 
dnl  the main configure script and the runtime configure script easier.
dnl
dnl  If Babel support for JAVA is enabled:
dnl     the cpp macro JAVA_DISABLED is undefined
dnl     the automake conditional SUPPORT_JAVA is true
dnl
dnl  If Babel support for JAVA is disabled:
dnl     the cpp macro JAVA_DISABLED is defined as true
dnl     the automake conditional SUPPORT_JAVA is false
dnl
dnl  @author Gary Kumfert

AC_DEFUN([LLNL_CONFIRM_BABEL_JAVA_SUPPORT], [
  if test -z "$with_java"; then
    with_java=yes
  fi
  AC_ARG_WITH(java,    [  --without-java          disable java support])
  if test "X$with_java" = "Xno"; then
    AC_MSG_WARN([Cannot disable Java entirely, only Java support in Babel.])
    AC_MSG_WARN([This package still needs working Java internally.])
  fi
  # for political reasons, AC_PROG_JAVA checks for gcj first.  
  #These variables override that.
  JAVA=java
  JAVAC=javac
  AC_JAVA_OPTIONS
  AC_CHECK_CLASSPATH
  AC_PROG_JAVAC
  AC_PROG_JAVA
  AC_PROG_JAR
  LLNL_PROG_JDB
  AC_TRY_COMPILE_JAVA
  if test "X$with_java" != "Xno"; then
    AC_PROG_JAVADOC
    LLNL_PROG_JAVAH
    if test "X$ac_cv_header_jni_h" = "Xno"; then
      AC_MSG_WARN([Cannot find jni.h, Java support will be disabled])
      with_java=no
      msgs="$msgs
  	  Java support disabled against request (no jni.h found!)"
    fi;
  else
      msgs="$msgs
  	  Java support disabled by request"
  fi 
  AM_CONDITIONAL(SUPPORT_JAVA, test "X$with_java" != "Xno")
  if test "X$with_java" = "Xno"; then
    AC_DEFINE(JAVA_DISABLED, 1, [If defined, Java support was disabled at configure time])
  else
    msgs="$msgs
  	  Java enabled.";
  fi 
])


dnl *** file: config/llnl-ac-macros/llnl_prog_test_ef.m4
# LLNL_PROG_TEST_EF
#
# Some platforms (sun) doesn't have a default program (called "test")
# that understands the "-ef" option.  test FILE1 -ef FILE2 is true only
# if both files have the same inode.
#
AC_DEFUN([LLNL_PROG_TEST_EF],
[AC_CACHE_CHECK([for a test program that accepts -ef],llnl_cv_prog_test_ef,
[echo "" > conftest1
ln -s conftest1 conftest2
llnl_cv_prog_test_ef=none
for t in $TEST test /bin/test /usr/bin/test /usr/local/bin/test /usr/ucb/bin/test ; do
  if test -x $t; then 
    if $t conftest1 -ef conftest2; then
      llnl_cv_prog_test_ef=$t
      break
    fi;
  fi;
done;
rm conftest1 conftest2
])
if test "$llnl_cv_prog_test_ef" = "none"; then
  AC_MSG_ERROR([Cannot find "test" program such that "test FILE1 -ef FILE2".\n Set TEST environment variable and rerun configure])
else
  TEST_EF=$llnl_cv_prog_test_ef
  AC_SUBST(TEST_EF)
fi
])


dnl *** file: config/llnl-ac-macros/libtool.m4
# libtool.m4 - Configure libtool for the host system. -*-Shell-script-*-
## Copyright 1996, 1997, 1998, 1999, 2000, 2001
## Free Software Foundation, Inc.
## Originally by Gordon Matzigkeit <gord@gnu.ai.mit.edu>, 1996
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
##
## As a special exception to the GNU General Public License, if you
## distribute this file as part of a program that contains a
## configuration script generated by Autoconf, you may include it under
## the same distribution terms that you use for the rest of that program.

# serial 46 AC_PROG_LIBTOOL

AC_DEFUN([AC_PROG_LIBTOOL],
[AC_REQUIRE([AC_LIBTOOL_SETUP])dnl

# This can be used to rebuild libtool when needed
LIBTOOL_DEPS="$ac_aux_dir/ltmain.sh"

# Always use our own libtool.
LIBTOOL='$(SHELL) $(top_builddir)/libtool'
AC_SUBST(LIBTOOL)dnl

# Prevent multiple expansion
define([AC_PROG_LIBTOOL], [])
])

AC_DEFUN([AC_LIBTOOL_SETUP],
[AC_PREREQ(2.13)dnl
AC_REQUIRE([AC_ENABLE_SHARED])dnl
AC_REQUIRE([AC_ENABLE_STATIC])dnl
AC_REQUIRE([AC_ENABLE_FAST_INSTALL])dnl
AC_REQUIRE([AC_CANONICAL_HOST])dnl
AC_REQUIRE([AC_CANONICAL_BUILD])dnl
AC_REQUIRE([AC_PROG_CC])dnl
AC_REQUIRE([AC_PROG_LD])dnl
AC_REQUIRE([AC_PROG_LD_RELOAD_FLAG])dnl
AC_REQUIRE([AC_PROG_NM])dnl
AC_REQUIRE([LT_AC_PROG_SED])dnl

AC_REQUIRE([AC_PROG_LN_S])dnl
AC_REQUIRE([AC_DEPLIBS_CHECK_METHOD])dnl
AC_REQUIRE([AC_OBJEXT])dnl
AC_REQUIRE([AC_EXEEXT])dnl
dnl

_LT_AC_PROG_ECHO_BACKSLASH
# Only perform the check for file, if the check method requires it
case $deplibs_check_method in
file_magic*)
  if test "$file_magic_cmd" = '$MAGIC_CMD'; then
    AC_PATH_MAGIC
  fi
  ;;
esac

AC_CHECK_TOOL(RANLIB, ranlib, :)
AC_CHECK_TOOL(STRIP, strip, :)

ifdef([AC_PROVIDE_AC_LIBTOOL_DLOPEN], enable_dlopen=yes, enable_dlopen=no)
ifdef([AC_PROVIDE_AC_LIBTOOL_WIN32_DLL],
enable_win32_dll=yes, enable_win32_dll=no)

AC_ARG_ENABLE(libtool-lock,
  [  --disable-libtool-lock  avoid locking (might break parallel builds)])
test "x$enable_libtool_lock" != xno && enable_libtool_lock=yes

# Some flags need to be propagated to the compiler or linker for good
# libtool support.
case $host in
*-*-irix6*)
  # Find out which ABI we are using.
  echo '[#]line __oline__ "configure"' > conftest.$ac_ext
  if AC_TRY_EVAL(ac_compile); then
    case `/usr/bin/file conftest.$ac_objext` in
    *32-bit*)
      LD="${LD-ld} -32"
      ;;
    *N32*)
      LD="${LD-ld} -n32"
      ;;
    *64-bit*)
      LD="${LD-ld} -64"
      ;;
    esac
  fi
  rm -rf conftest*
  ;;

*-*-sco3.2v5*)
  # On SCO OpenServer 5, we need -belf to get full-featured binaries.
  SAVE_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -belf"
  AC_CACHE_CHECK([whether the C compiler needs -belf], lt_cv_cc_needs_belf,
    [AC_LANG_SAVE
     AC_LANG_C
     AC_TRY_LINK([],[],[lt_cv_cc_needs_belf=yes],[lt_cv_cc_needs_belf=no])
     AC_LANG_RESTORE])
  if test x"$lt_cv_cc_needs_belf" != x"yes"; then
    # this is probably gcc 2.8.0, egcs 1.0 or newer; no need for -belf
    CFLAGS="$SAVE_CFLAGS"
  fi
  ;;

ifdef([AC_PROVIDE_AC_LIBTOOL_WIN32_DLL],
[*-*-cygwin* | *-*-mingw* | *-*-pw32*)
  AC_CHECK_TOOL(DLLTOOL, dlltool, false)
  AC_CHECK_TOOL(AS, as, false)
  AC_CHECK_TOOL(OBJDUMP, objdump, false)

  # recent cygwin and mingw systems supply a stub DllMain which the user
  # can override, but on older systems we have to supply one
  AC_CACHE_CHECK([if libtool should supply DllMain function], lt_cv_need_dllmain,
    [AC_TRY_LINK([],
      [extern int __attribute__((__stdcall__)) DllMain(void*, int, void*);
      DllMain (0, 0, 0);],
      [lt_cv_need_dllmain=no],[lt_cv_need_dllmain=yes])])

  case $host/$CC in
  *-*-cygwin*/gcc*-mno-cygwin*|*-*-mingw*)
    # old mingw systems require "-dll" to link a DLL, while more recent ones
    # require "-mdll"
    SAVE_CFLAGS="$CFLAGS"
    CFLAGS="$CFLAGS -mdll"
    AC_CACHE_CHECK([how to link DLLs], lt_cv_cc_dll_switch,
      [AC_TRY_LINK([], [], [lt_cv_cc_dll_switch=-mdll],[lt_cv_cc_dll_switch=-dll])])
    CFLAGS="$SAVE_CFLAGS" ;;
  *-*-cygwin* | *-*-pw32*)
    # cygwin systems need to pass --dll to the linker, and not link
    # crt.o which will require a WinMain@16 definition.
    lt_cv_cc_dll_switch="-Wl,--dll -nostartfiles" ;;
  esac
  ;;
  ])
esac

_LT_AC_LTCONFIG_HACK

])

# AC_LIBTOOL_HEADER_ASSERT
# ------------------------
AC_DEFUN([AC_LIBTOOL_HEADER_ASSERT],
[AC_CACHE_CHECK([whether $CC supports assert without backlinking],
    [lt_cv_func_assert_works],
    [case $host in
    *-*-solaris*)
      if test "$GCC" = yes && test "$with_gnu_ld" != yes; then
        case `$CC --version 2>/dev/null` in
        [[12]].*) lt_cv_func_assert_works=no ;;
        *)        lt_cv_func_assert_works=yes ;;
        esac
      fi
      ;;
    esac])

if test "x$lt_cv_func_assert_works" = xyes; then
  AC_CHECK_HEADERS(assert.h)
fi
])# AC_LIBTOOL_HEADER_ASSERT

# _LT_AC_CHECK_DLFCN
# --------------------
AC_DEFUN([_LT_AC_CHECK_DLFCN],
[AC_CHECK_HEADERS(dlfcn.h)
])# _LT_AC_CHECK_DLFCN

# AC_LIBTOOL_SYS_GLOBAL_SYMBOL_PIPE
# ---------------------------------
AC_DEFUN([AC_LIBTOOL_SYS_GLOBAL_SYMBOL_PIPE],
[AC_REQUIRE([AC_CANONICAL_HOST])
AC_REQUIRE([AC_PROG_NM])
AC_REQUIRE([AC_OBJEXT])
# Check for command to grab the raw symbol name followed by C symbol from nm.
AC_MSG_CHECKING([command to parse $NM output])
AC_CACHE_VAL([lt_cv_sys_global_symbol_pipe], [dnl

# These are sane defaults that work on at least a few old systems.
# [They come from Ultrix.  What could be older than Ultrix?!! ;)]

# Character class describing NM global symbol codes.
symcode='[[BCDEGRST]]'

# Regexp to match symbols that can be accessed directly from C.
sympat='\([[_A-Za-z]][[_A-Za-z0-9]]*\)'

# Transform the above into a raw symbol and a C symbol.
symxfrm='\1 \2\3 \3'

# Transform an extracted symbol line into a proper C declaration
lt_cv_global_symbol_to_cdecl="sed -n -e 's/^. .* \(.*\)$/extern char \1;/p'"

# Transform an extracted symbol line into symbol name and symbol address
lt_cv_global_symbol_to_c_name_address="sed -n -e 's/^: \([[^ ]]*\) $/  {\\\"\1\\\", (lt_ptr) 0},/p' -e 's/^$symcode \([[^ ]]*\) \([[^ ]]*\)$/  {\"\2\", (lt_ptr) \&\2},/p'"

# Define system-specific variables.
case $host_os in
aix*)
  symcode='[[BCDT]]'
  ;;
cygwin* | mingw* | pw32*)
  symcode='[[ABCDGISTW]]'
  ;;
hpux*) # Its linker distinguishes data from code symbols
  lt_cv_global_symbol_to_cdecl="sed -n -e 's/^T .* \(.*\)$/extern char \1();/p' -e 's/^$symcode* .* \(.*\)$/extern char \1;/p'"
  lt_cv_global_symbol_to_c_name_address="sed -n -e 's/^: \([[^ ]]*\) $/  {\\\"\1\\\", (lt_ptr) 0},/p' -e 's/^$symcode* \([[^ ]]*\) \([[^ ]]*\)$/  {\"\2\", (lt_ptr) \&\2},/p'"
  ;;
irix* | nonstopux*)
  symcode='[[BCDEGRST]]'
  ;;
osf*)
  symcode='[[BCDEGQRST]]'
  ;;
solaris* | sysv5*)
  symcode='[[BDT]]'
  ;;
sysv4)
  symcode='[[DFNSTU]]'
  ;;
esac

# Handle CRLF in mingw tool chain
opt_cr=
case $host_os in
mingw*)
  opt_cr=`echo 'x\{0,1\}' | tr x '\015'` # option cr in regexp
  ;;
esac

# If we're using GNU nm, then use its standard symbol codes.
if $NM -V 2>&1 | egrep '(GNU|with BFD)' > /dev/null; then
  symcode='[[ABCDGISTW]]'
fi

# Try without a prefix undercore, then with it.
for ac_symprfx in "" "_"; do

  # Write the raw and C identifiers.
lt_cv_sys_global_symbol_pipe="sed -n -e 's/^.*[[ 	]]\($symcode$symcode*\)[[ 	]][[ 	]]*\($ac_symprfx\)$sympat$opt_cr$/$symxfrm/p'"

  # Check to see that the pipe works correctly.
  pipe_works=no
  rm -f conftest*
  cat > conftest.$ac_ext <<EOF
#ifdef __cplusplus
extern "C" {
#endif
char nm_test_var;
void nm_test_func(){}
#ifdef __cplusplus
}
#endif
int main(){nm_test_var='a';nm_test_func();return(0);}
EOF

  if AC_TRY_EVAL(ac_compile); then
    # Now try to grab the symbols.
    nlist=conftest.nm
    if AC_TRY_EVAL(NM conftest.$ac_objext \| $lt_cv_sys_global_symbol_pipe \> $nlist) && test -s "$nlist"; then
      # Try sorting and uniquifying the output.
      if sort "$nlist" | uniq > "$nlist"T; then
	mv -f "$nlist"T "$nlist"
      else
	rm -f "$nlist"T
      fi

      # Make sure that we snagged all the symbols we need.
      if egrep ' nm_test_var$' "$nlist" >/dev/null; then
	if egrep ' nm_test_func$' "$nlist" >/dev/null; then
	  cat <<EOF > conftest.$ac_ext
#ifdef __cplusplus
extern "C" {
#endif

EOF
	  # Now generate the symbol file.
	  eval "$lt_cv_global_symbol_to_cdecl"' < "$nlist" >> conftest.$ac_ext'

	  cat <<EOF >> conftest.$ac_ext
#if defined (__STDC__) && __STDC__
# define lt_ptr void *
#else
# define lt_ptr char *
# define const
#endif

/* The mapping between symbol names and symbols. */
const struct {
  const char *name;
  lt_ptr address;
}
lt_preloaded_symbols[[]] =
{
EOF
	  sed "s/^$symcode$symcode* \(.*\) \(.*\)$/  {\"\2\", (lt_ptr) \&\2},/" < "$nlist" >> conftest.$ac_ext
	  cat <<\EOF >> conftest.$ac_ext
  {0, (lt_ptr) 0}
};

#ifdef __cplusplus
}
#endif
EOF
	  # Now try linking the two files.
	  mv conftest.$ac_objext conftstm.$ac_objext
	  save_LIBS="$LIBS"
	  save_CFLAGS="$CFLAGS"
	  LIBS="conftstm.$ac_objext"
	  CFLAGS="$CFLAGS$no_builtin_flag"
	  if AC_TRY_EVAL(ac_link) && test -s conftest$ac_exeext; then
	    pipe_works=yes
	  fi
	  LIBS="$save_LIBS"
	  CFLAGS="$save_CFLAGS"
	else
	  echo "cannot find nm_test_func in $nlist" >&AC_FD_CC
	fi
      else
	echo "cannot find nm_test_var in $nlist" >&AC_FD_CC
      fi
    else
      echo "cannot run $lt_cv_sys_global_symbol_pipe" >&AC_FD_CC
    fi
  else
    echo "$progname: failed program was:" >&AC_FD_CC
    cat conftest.$ac_ext >&5
  fi
  rm -f conftest* conftst*

  # Do not use the global_symbol_pipe unless it works.
  if test "$pipe_works" = yes; then
    break
  else
    lt_cv_sys_global_symbol_pipe=
  fi
done
])
global_symbol_pipe="$lt_cv_sys_global_symbol_pipe"
if test -z "$lt_cv_sys_global_symbol_pipe"; then
  global_symbol_to_cdecl=
  global_symbol_to_c_name_address=
else
  global_symbol_to_cdecl="$lt_cv_global_symbol_to_cdecl"
  global_symbol_to_c_name_address="$lt_cv_global_symbol_to_c_name_address"
fi
if test -z "$global_symbol_pipe$global_symbol_to_cdec$global_symbol_to_c_name_address";
then
  AC_MSG_RESULT(failed)
else
  AC_MSG_RESULT(ok)
fi
]) # AC_LIBTOOL_SYS_GLOBAL_SYMBOL_PIPE

# _LT_AC_LIBTOOL_SYS_PATH_SEPARATOR
# ---------------------------------
AC_DEFUN([_LT_AC_LIBTOOL_SYS_PATH_SEPARATOR],
[# Find the correct PATH separator.  Usually this is `:', but
# DJGPP uses `;' like DOS.
if test "X${PATH_SEPARATOR+set}" != Xset; then
  UNAME=${UNAME-`uname 2>/dev/null`}
  case X$UNAME in
    *-DOS) lt_cv_sys_path_separator=';' ;;
    *)     lt_cv_sys_path_separator=':' ;;
  esac
  PATH_SEPARATOR=$lt_cv_sys_path_separator
fi
])# _LT_AC_LIBTOOL_SYS_PATH_SEPARATOR

# _LT_AC_PROG_ECHO_BACKSLASH
# --------------------------
# Add some code to the start of the generated configure script which
# will find an echo command which doesn't interpret backslashes.
AC_DEFUN([_LT_AC_PROG_ECHO_BACKSLASH],
[ifdef([AC_DIVERSION_NOTICE], [AC_DIVERT_PUSH(AC_DIVERSION_NOTICE)],
			      [AC_DIVERT_PUSH(NOTICE)])
_LT_AC_LIBTOOL_SYS_PATH_SEPARATOR

# Check that we are running under the correct shell.
SHELL=${CONFIG_SHELL-/bin/sh}

case X$ECHO in
X*--fallback-echo)
  # Remove one level of quotation (which was required for Make).
  ECHO=`echo "$ECHO" | sed 's,\\\\\[$]\\[$]0,'[$]0','`
  ;;
esac

echo=${ECHO-echo}
if test "X[$]1" = X--no-reexec; then
  # Discard the --no-reexec flag, and continue.
  shift
elif test "X[$]1" = X--fallback-echo; then
  # Avoid inline document here, it may be left over
  :
elif test "X`($echo '\t') 2>/dev/null`" = 'X\t'; then
  # Yippee, $echo works!
  :
else
  # Restart under the correct shell.
  exec $SHELL "[$]0" --no-reexec ${1+"[$]@"}
fi

if test "X[$]1" = X--fallback-echo; then
  # used as fallback echo
  shift
  cat <<EOF
$*
EOF
  exit 0
fi

# The HP-UX ksh and POSIX shell print the target directory to stdout
# if CDPATH is set.
if test "X${CDPATH+set}" = Xset; then CDPATH=:; export CDPATH; fi

if test -z "$ECHO"; then
if test "X${echo_test_string+set}" != Xset; then
# find a string as large as possible, as long as the shell can cope with it
  for cmd in 'sed 50q "[$]0"' 'sed 20q "[$]0"' 'sed 10q "[$]0"' 'sed 2q "[$]0"' 'echo test'; do
    # expected sizes: less than 2Kb, 1Kb, 512 bytes, 16 bytes, ...
    if (echo_test_string="`eval $cmd`") 2>/dev/null &&
       echo_test_string="`eval $cmd`" &&
       (test "X$echo_test_string" = "X$echo_test_string") 2>/dev/null
    then
      break
    fi
  done
fi

if test "X`($echo '\t') 2>/dev/null`" = 'X\t' &&
   echo_testing_string=`($echo "$echo_test_string") 2>/dev/null` &&
   test "X$echo_testing_string" = "X$echo_test_string"; then
  :
else
  # The Solaris, AIX, and Digital Unix default echo programs unquote
  # backslashes.  This makes it impossible to quote backslashes using
  #   echo "$something" | sed 's/\\/\\\\/g'
  #
  # So, first we look for a working echo in the user's PATH.

  IFS="${IFS= 	}"; save_ifs="$IFS"; IFS=$PATH_SEPARATOR
  for dir in $PATH /usr/ucb; do
    if (test -f $dir/echo || test -f $dir/echo$ac_exeext) &&
       test "X`($dir/echo '\t') 2>/dev/null`" = 'X\t' &&
       echo_testing_string=`($dir/echo "$echo_test_string") 2>/dev/null` &&
       test "X$echo_testing_string" = "X$echo_test_string"; then
      echo="$dir/echo"
      break
    fi
  done
  IFS="$save_ifs"

  if test "X$echo" = Xecho; then
    # We didn't find a better echo, so look for alternatives.
    if test "X`(print -r '\t') 2>/dev/null`" = 'X\t' &&
       echo_testing_string=`(print -r "$echo_test_string") 2>/dev/null` &&
       test "X$echo_testing_string" = "X$echo_test_string"; then
      # This shell has a builtin print -r that does the trick.
      echo='print -r'
    elif (test -f /bin/ksh || test -f /bin/ksh$ac_exeext) &&
	 test "X$CONFIG_SHELL" != X/bin/ksh; then
      # If we have ksh, try running configure again with it.
      ORIGINAL_CONFIG_SHELL=${CONFIG_SHELL-/bin/sh}
      export ORIGINAL_CONFIG_SHELL
      CONFIG_SHELL=/bin/ksh
      export CONFIG_SHELL
      exec $CONFIG_SHELL "[$]0" --no-reexec ${1+"[$]@"}
    else
      # Try using printf.
      echo='printf %s\n'
      if test "X`($echo '\t') 2>/dev/null`" = 'X\t' &&
	 echo_testing_string=`($echo "$echo_test_string") 2>/dev/null` &&
	 test "X$echo_testing_string" = "X$echo_test_string"; then
	# Cool, printf works
	:
      elif echo_testing_string=`($ORIGINAL_CONFIG_SHELL "[$]0" --fallback-echo '\t') 2>/dev/null` &&
	   test "X$echo_testing_string" = 'X\t' &&
	   echo_testing_string=`($ORIGINAL_CONFIG_SHELL "[$]0" --fallback-echo "$echo_test_string") 2>/dev/null` &&
	   test "X$echo_testing_string" = "X$echo_test_string"; then
	CONFIG_SHELL=$ORIGINAL_CONFIG_SHELL
	export CONFIG_SHELL
	SHELL="$CONFIG_SHELL"
	export SHELL
	echo="$CONFIG_SHELL [$]0 --fallback-echo"
      elif echo_testing_string=`($CONFIG_SHELL "[$]0" --fallback-echo '\t') 2>/dev/null` &&
	   test "X$echo_testing_string" = 'X\t' &&
	   echo_testing_string=`($CONFIG_SHELL "[$]0" --fallback-echo "$echo_test_string") 2>/dev/null` &&
	   test "X$echo_testing_string" = "X$echo_test_string"; then
	echo="$CONFIG_SHELL [$]0 --fallback-echo"
      else
	# maybe with a smaller string...
	prev=:

	for cmd in 'echo test' 'sed 2q "[$]0"' 'sed 10q "[$]0"' 'sed 20q "[$]0"' 'sed 50q "[$]0"'; do
	  if (test "X$echo_test_string" = "X`eval $cmd`") 2>/dev/null
	  then
	    break
	  fi
	  prev="$cmd"
	done

	if test "$prev" != 'sed 50q "[$]0"'; then
	  echo_test_string=`eval $prev`
	  export echo_test_string
	  exec ${ORIGINAL_CONFIG_SHELL-${CONFIG_SHELL-/bin/sh}} "[$]0" ${1+"[$]@"}
	else
	  # Oops.  We lost completely, so just stick with echo.
	  echo=echo
	fi
      fi
    fi
  fi
fi
fi

# Copy echo and quote the copy suitably for passing to libtool from
# the Makefile, instead of quoting the original, which is used later.
ECHO=$echo
if test "X$ECHO" = "X$CONFIG_SHELL [$]0 --fallback-echo"; then
   ECHO="$CONFIG_SHELL \\\$\[$]0 --fallback-echo"
fi

AC_SUBST(ECHO)
AC_DIVERT_POP
])# _LT_AC_PROG_ECHO_BACKSLASH

# _LT_AC_TRY_DLOPEN_SELF (ACTION-IF-TRUE, ACTION-IF-TRUE-W-USCORE,
#                           ACTION-IF-FALSE, ACTION-IF-CROSS-COMPILING)
# ------------------------------------------------------------------
AC_DEFUN([_LT_AC_TRY_DLOPEN_SELF],
[if test "$cross_compiling" = yes; then :
  [$4]
else
  AC_REQUIRE([_LT_AC_CHECK_DLFCN])dnl
  lt_dlunknown=0; lt_dlno_uscore=1; lt_dlneed_uscore=2
  lt_status=$lt_dlunknown
  cat > conftest.$ac_ext <<EOF
[#line __oline__ "configure"
#include "confdefs.h"

#if HAVE_DLFCN_H
#include <dlfcn.h>
#endif

#include <stdio.h>

#ifdef RTLD_GLOBAL
#  define LT_DLGLOBAL		RTLD_GLOBAL
#else
#  ifdef DL_GLOBAL
#    define LT_DLGLOBAL		DL_GLOBAL
#  else
#    define LT_DLGLOBAL		0
#  endif
#endif

/* We may have to define LT_DLLAZY_OR_NOW in the command line if we
   find out it does not work in some platform. */
#ifndef LT_DLLAZY_OR_NOW
#  ifdef RTLD_LAZY
#    define LT_DLLAZY_OR_NOW		RTLD_LAZY
#  else
#    ifdef DL_LAZY
#      define LT_DLLAZY_OR_NOW		DL_LAZY
#    else
#      ifdef RTLD_NOW
#        define LT_DLLAZY_OR_NOW	RTLD_NOW
#      else
#        ifdef DL_NOW
#          define LT_DLLAZY_OR_NOW	DL_NOW
#        else
#          define LT_DLLAZY_OR_NOW	0
#        endif
#      endif
#    endif
#  endif
#endif

#ifdef __cplusplus
extern "C" void exit (int);
#endif

void fnord() { int i=42;}
int main ()
{
  void *self = dlopen (0, LT_DLGLOBAL|LT_DLLAZY_OR_NOW);
  int status = $lt_dlunknown;

  if (self)
    {
      if (dlsym (self,"fnord"))       status = $lt_dlno_uscore;
      else if (dlsym( self,"_fnord")) status = $lt_dlneed_uscore;
      /* dlclose (self); */
    }

    exit (status);
}]
EOF
  if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext} 2>/dev/null; then
    (./conftest; exit; ) 2>/dev/null
    lt_status=$?
    case x$lt_status in
      x$lt_dlno_uscore) $1 ;;
      x$lt_dlneed_uscore) $2 ;;
      x$lt_unknown|x*) $3 ;;
    esac
  else :
    # compilation failed
    $3
  fi
fi
rm -fr conftest*
])# _LT_AC_TRY_DLOPEN_SELF

# AC_LIBTOOL_DLOPEN_SELF
# -------------------
AC_DEFUN([AC_LIBTOOL_DLOPEN_SELF],
[if test "x$enable_dlopen" != xyes; then
  enable_dlopen=unknown
  enable_dlopen_self=unknown
  enable_dlopen_self_static=unknown
else
  lt_cv_dlopen=no
  lt_cv_dlopen_libs=

  case $host_os in
  beos*)
    lt_cv_dlopen="load_add_on"
    lt_cv_dlopen_libs=
    lt_cv_dlopen_self=yes
    ;;

  cygwin* | mingw* | pw32*)
    lt_cv_dlopen="LoadLibrary"
    lt_cv_dlopen_libs=
   ;;

  *)
    AC_CHECK_FUNC([shl_load],
          [lt_cv_dlopen="shl_load"],
      [AC_CHECK_LIB([dld], [shl_load],
            [lt_cv_dlopen="shl_load" lt_cv_dlopen_libs="-dld"],
	[AC_CHECK_FUNC([dlopen],
	      [lt_cv_dlopen="dlopen"],
	  [AC_CHECK_LIB([dl], [dlopen],
	        [lt_cv_dlopen="dlopen" lt_cv_dlopen_libs="-ldl"],
	    [AC_CHECK_LIB([svld], [dlopen],
	          [lt_cv_dlopen="dlopen" lt_cv_dlopen_libs="-lsvld"],
	      [AC_CHECK_LIB([dld], [dld_link],
	            [lt_cv_dlopen="dld_link" lt_cv_dlopen_libs="-dld"])
	      ])
	    ])
	  ])
	])
      ])
    ;;
  esac

  if test "x$lt_cv_dlopen" != xno; then
    enable_dlopen=yes
  else
    enable_dlopen=no
  fi

  case $lt_cv_dlopen in
  dlopen)
    save_CPPFLAGS="$CPPFLAGS"
    AC_REQUIRE([_LT_AC_CHECK_DLFCN])dnl
    test "x$ac_cv_header_dlfcn_h" = xyes && CPPFLAGS="$CPPFLAGS -DHAVE_DLFCN_H"

    save_LDFLAGS="$LDFLAGS"
    eval LDFLAGS=\"\$LDFLAGS $export_dynamic_flag_spec\"

    save_LIBS="$LIBS"
    LIBS="$lt_cv_dlopen_libs $LIBS"

    AC_CACHE_CHECK([whether a program can dlopen itself],
	  lt_cv_dlopen_self, [dnl
	  _LT_AC_TRY_DLOPEN_SELF(
	    lt_cv_dlopen_self=yes, lt_cv_dlopen_self=yes,
	    lt_cv_dlopen_self=no, lt_cv_dlopen_self=cross)
    ])

    if test "x$lt_cv_dlopen_self" = xyes; then
      LDFLAGS="$LDFLAGS $link_static_flag"
      AC_CACHE_CHECK([whether a statically linked program can dlopen itself],
    	  lt_cv_dlopen_self_static, [dnl
	  _LT_AC_TRY_DLOPEN_SELF(
	    lt_cv_dlopen_self_static=yes, lt_cv_dlopen_self_static=yes,
	    lt_cv_dlopen_self_static=no,  lt_cv_dlopen_self_static=cross)
      ])
    fi

    CPPFLAGS="$save_CPPFLAGS"
    LDFLAGS="$save_LDFLAGS"
    LIBS="$save_LIBS"
    ;;
  esac

  case $lt_cv_dlopen_self in
  yes|no) enable_dlopen_self=$lt_cv_dlopen_self ;;
  *) enable_dlopen_self=unknown ;;
  esac

  case $lt_cv_dlopen_self_static in
  yes|no) enable_dlopen_self_static=$lt_cv_dlopen_self_static ;;
  *) enable_dlopen_self_static=unknown ;;
  esac
fi
])# AC_LIBTOOL_DLOPEN_SELF

AC_DEFUN([_LT_AC_LTCONFIG_HACK],
[AC_REQUIRE([AC_LIBTOOL_SYS_GLOBAL_SYMBOL_PIPE])dnl
# Sed substitution that helps us do robust quoting.  It backslashifies
# metacharacters that are still active within double-quoted strings.
Xsed='sed -e s/^X//'
sed_quote_subst='s/\([[\\"\\`$\\\\]]\)/\\\1/g'

# Same as above, but do not quote variable references.
double_quote_subst='s/\([[\\"\\`\\\\]]\)/\\\1/g'

# Sed substitution to delay expansion of an escaped shell variable in a
# double_quote_subst'ed string.
delay_variable_subst='s/\\\\\\\\\\\$/\\\\\\$/g'

# Constants:
rm="rm -f"

# Global variables:
default_ofile=libtool
can_build_shared=yes

# All known linkers require a `.a' archive for static linking (except M$VC,
# which needs '.lib').
libext=a
ltmain="$ac_aux_dir/ltmain.sh"
ofile="$default_ofile"
with_gnu_ld="$lt_cv_prog_gnu_ld"
need_locks="$enable_libtool_lock"

old_CC="$CC"
old_CFLAGS="$CFLAGS"

# Set sane defaults for various variables
test -z "$AR" && AR=ar
test -z "$AR_FLAGS" && AR_FLAGS=cru
test -z "$AS" && AS=as
test -z "$CC" && CC=cc
test -z "$DLLTOOL" && DLLTOOL=dlltool
test -z "$LD" && LD=ld
test -z "$LN_S" && LN_S="ln -s"
test -z "$MAGIC_CMD" && MAGIC_CMD=file
test -z "$NM" && NM=nm
test -z "$OBJDUMP" && OBJDUMP=objdump
test -z "$RANLIB" && RANLIB=:
test -z "$STRIP" && STRIP=:
test -z "$ac_objext" && ac_objext=o

if test x"$host" != x"$build"; then
  ac_tool_prefix=${host_alias}-
else
  ac_tool_prefix=
fi

# Transform linux* to *-*-linux-gnu*, to support old configure scripts.
case $host_os in
linux-gnu*) ;;
linux*) host=`echo $host | sed 's/^\(.*-.*-linux\)\(.*\)$/\1-gnu\2/'`
esac

case $host_os in
aix3*)
  # AIX sometimes has problems with the GCC collect2 program.  For some
  # reason, if we set the COLLECT_NAMES environment variable, the problems
  # vanish in a puff of smoke.
  if test "X${COLLECT_NAMES+set}" != Xset; then
    COLLECT_NAMES=
    export COLLECT_NAMES
  fi
  ;;
esac

# Determine commands to create old-style static archives.
old_archive_cmds='$AR $AR_FLAGS $oldlib$oldobjs$old_deplibs'
old_postinstall_cmds='chmod 644 $oldlib'
old_postuninstall_cmds=

if test -n "$RANLIB"; then
  case $host_os in
  openbsd*)
    old_postinstall_cmds="\$RANLIB -t \$oldlib~$old_postinstall_cmds"
    ;;
  *)
    old_postinstall_cmds="\$RANLIB \$oldlib~$old_postinstall_cmds"
    ;;
  esac
  old_archive_cmds="$old_archive_cmds~\$RANLIB \$oldlib"
fi

# Allow CC to be a program name with arguments.
set dummy $CC
compiler="[$]2"

## FIXME: this should be a separate macro
##
AC_MSG_CHECKING([for objdir])
rm -f .libs 2>/dev/null
mkdir .libs 2>/dev/null
if test -d .libs; then
  objdir=.libs
else
  # MS-DOS does not allow filenames that begin with a dot.
  objdir=_libs
fi
rmdir .libs 2>/dev/null
AC_MSG_RESULT($objdir)
##
## END FIXME


## FIXME: this should be a separate macro
##
AC_ARG_WITH(pic,
[  --with-pic              try to use only PIC/non-PIC objects [default=use both]],
pic_mode="$withval", pic_mode=default)
test -z "$pic_mode" && pic_mode=default

# We assume here that the value for lt_cv_prog_cc_pic will not be cached
# in isolation, and that seeing it set (from the cache) indicates that
# the associated values are set (in the cache) correctly too.
AC_MSG_CHECKING([for $compiler option to produce PIC])
AC_CACHE_VAL(lt_cv_prog_cc_pic,
[ lt_cv_prog_cc_pic=
  lt_cv_prog_cc_shlib=
  lt_cv_prog_cc_wl=
  lt_cv_prog_cc_static=
  lt_cv_prog_cc_no_builtin=
  lt_cv_prog_cc_can_build_shared=$can_build_shared

  if test "$GCC" = yes; then
    lt_cv_prog_cc_wl='-Wl,'
    lt_cv_prog_cc_static='-static'

    case $host_os in
    aix*)
      # Below there is a dirty hack to force normal static linking with -ldl
      # The problem is because libdl dynamically linked with both libc and
      # libC (AIX C++ library), which obviously doesn't included in libraries
      # list by gcc. This cause undefined symbols with -static flags.
      # This hack allows C programs to be linked with "-static -ldl", but
      # not sure about C++ programs.
      lt_cv_prog_cc_static="$lt_cv_prog_cc_static ${lt_cv_prog_cc_wl}-lC"
      ;;
    amigaos*)
      # FIXME: we need at least 68020 code to build shared libraries, but
      # adding the `-m68020' flag to GCC prevents building anything better,
      # like `-m68040'.
      lt_cv_prog_cc_pic='-m68020 -resident32 -malways-restore-a4'
      ;;
    beos* | irix5* | irix6* | nonstopux* | osf3* | osf4* | osf5*)
      # PIC is the default for these OSes.
      ;;
    darwin* | rhapsody*)
      # PIC is the default on this platform
      # Common symbols not allowed in MH_DYLIB files
      lt_cv_prog_cc_pic='-fno-common'
      ;;
    cygwin* | mingw* | pw32* | os2*)
      # This hack is so that the source file can tell whether it is being
      # built for inclusion in a dll (and should export symbols for example).
      lt_cv_prog_cc_pic='-DDLL_EXPORT'
      ;;
    sysv4*MP*)
      if test -d /usr/nec; then
	 lt_cv_prog_cc_pic=-Kconform_pic
      fi
      ;;
    *)
      lt_cv_prog_cc_pic='-fPIC'
      ;;
    esac
  else
    # PORTME Check for PIC flags for the system compiler.
    case $host_os in
    aix3* | aix4* ) #gkk | aix5*)
      lt_cv_prog_cc_wl='-Wl,'
      # All AIX code is PIC.
      if test "$host_cpu" = ia64; then
	# AIX 5 now supports IA64 processor
	lt_cv_prog_cc_static='-Bstatic'
      else
	lt_cv_prog_cc_static='-bnso -bI:/lib/syscalls.exp'
      fi
      ;;
   aix5*) #gkk I added all this
      lt_cv_prog_cc_wl='-Wl,'
      if test "$host_cpu" = ia64; then 
	lt_cv_prog_cc_static='-Bstatic'
      else
        lt_cv_prog_cc_pic='-G'
	lt_cv_prog_cc_static='-bstatic'
      fi
      ;;

    hpux9* | hpux10* | hpux11*)
      # Is there a better lt_cv_prog_cc_static that works with the bundled CC?
      lt_cv_prog_cc_wl='-Wl,'
      lt_cv_prog_cc_static="${lt_cv_prog_cc_wl}-a ${lt_cv_prog_cc_wl}archive"
      lt_cv_prog_cc_pic='+Z'
      ;;

    irix5* | irix6* | nonstopux*)
      lt_cv_prog_cc_wl='-Wl,'
      lt_cv_prog_cc_static='-non_shared'
      # PIC (with -KPIC) is the default.
      ;;

    cygwin* | mingw* | pw32* | os2*)
      # This hack is so that the source file can tell whether it is being
      # built for inclusion in a dll (and should export symbols for example).
      lt_cv_prog_cc_pic='-DDLL_EXPORT'
      ;;

    newsos6)
      lt_cv_prog_cc_pic='-KPIC'
      lt_cv_prog_cc_static='-Bstatic'
      ;;

    osf3* | osf4* | osf5*)
      # All OSF/1 code is PIC.
      lt_cv_prog_cc_wl='-Wl,'
      lt_cv_prog_cc_static='-non_shared'
      ;;

    sco3.2v5*)
      lt_cv_prog_cc_pic='-Kpic'
      lt_cv_prog_cc_static='-dn'
      lt_cv_prog_cc_shlib='-belf'
      ;;

    solaris*)
      lt_cv_prog_cc_pic='-KPIC'
      lt_cv_prog_cc_static='-Bstatic'
      lt_cv_prog_cc_wl='-Wl,'
      ;;

    sunos4*)
      lt_cv_prog_cc_pic='-PIC'
      lt_cv_prog_cc_static='-Bstatic'
      lt_cv_prog_cc_wl='-Qoption ld '
      ;;

    sysv4 | sysv4.2uw2* | sysv4.3* | sysv5*)
      lt_cv_prog_cc_pic='-KPIC'
      lt_cv_prog_cc_static='-Bstatic'
      lt_cv_prog_cc_wl='-Wl,'
      ;;

    uts4*)
      lt_cv_prog_cc_pic='-pic'
      lt_cv_prog_cc_static='-Bstatic'
      ;;

    sysv4*MP*)
      if test -d /usr/nec ;then
	lt_cv_prog_cc_pic='-Kconform_pic'
	lt_cv_prog_cc_static='-Bstatic'
      fi
      ;;

    *)
      lt_cv_prog_cc_can_build_shared=no
      ;;
    esac
  fi
])
if test -z "$lt_cv_prog_cc_pic"; then
  AC_MSG_RESULT([none])
else
  AC_MSG_RESULT([$lt_cv_prog_cc_pic])

  # Check to make sure the pic_flag actually works.
  AC_MSG_CHECKING([if $compiler PIC flag $lt_cv_prog_cc_pic works])
  AC_CACHE_VAL(lt_cv_prog_cc_pic_works, [dnl
    save_CFLAGS="$CFLAGS"
    CFLAGS="$CFLAGS $lt_cv_prog_cc_pic -DPIC"
    AC_TRY_COMPILE([], [], [dnl
      case $host_os in
      hpux9* | hpux10* | hpux11*)
	# On HP-UX, both CC and GCC only warn that PIC is supported... then
	# they create non-PIC objects.  So, if there were any warnings, we
	# assume that PIC is not supported.
	if test -s conftest.err; then
	  lt_cv_prog_cc_pic_works=no
	else
	  lt_cv_prog_cc_pic_works=yes
	fi
	;;
      *)
	lt_cv_prog_cc_pic_works=yes
	;;
      esac
    ], [dnl
      lt_cv_prog_cc_pic_works=no
    ])
    CFLAGS="$save_CFLAGS"
  ])

  if test "X$lt_cv_prog_cc_pic_works" = Xno; then
    lt_cv_prog_cc_pic=
    lt_cv_prog_cc_can_build_shared=no
  else
    lt_cv_prog_cc_pic=" $lt_cv_prog_cc_pic"
  fi

  AC_MSG_RESULT([$lt_cv_prog_cc_pic_works])
fi
##
## END FIXME

# Check for any special shared library compilation flags.
if test -n "$lt_cv_prog_cc_shlib"; then
  AC_MSG_WARN([\`$CC' requires \`$lt_cv_prog_cc_shlib' to build shared libraries])
  if echo "$old_CC $old_CFLAGS " | egrep -e "[[ 	]]$lt_cv_prog_cc_shlib[[ 	]]" >/dev/null; then :
  else
   AC_MSG_WARN([add \`$lt_cv_prog_cc_shlib' to the CC or CFLAGS env variable and reconfigure])
    lt_cv_prog_cc_can_build_shared=no
  fi
fi

## FIXME: this should be a separate macro
##
AC_MSG_CHECKING([if $compiler static flag $lt_cv_prog_cc_static works])
AC_CACHE_VAL([lt_cv_prog_cc_static_works], [dnl
  lt_cv_prog_cc_static_works=no
  save_LDFLAGS="$LDFLAGS"
  LDFLAGS="$LDFLAGS $lt_cv_prog_cc_static"
  AC_TRY_LINK([], [], [lt_cv_prog_cc_static_works=yes])
  LDFLAGS="$save_LDFLAGS"
])

# Belt *and* braces to stop my trousers falling down:
test "X$lt_cv_prog_cc_static_works" = Xno && lt_cv_prog_cc_static=
AC_MSG_RESULT([$lt_cv_prog_cc_static_works])

pic_flag="$lt_cv_prog_cc_pic"
special_shlib_compile_flags="$lt_cv_prog_cc_shlib"
wl="$lt_cv_prog_cc_wl"
link_static_flag="$lt_cv_prog_cc_static"
no_builtin_flag="$lt_cv_prog_cc_no_builtin"
can_build_shared="$lt_cv_prog_cc_can_build_shared"
##
## END FIXME


## FIXME: this should be a separate macro
##
# Check to see if options -o and -c are simultaneously supported by compiler
AC_MSG_CHECKING([if $compiler supports -c -o file.$ac_objext])
AC_CACHE_VAL([lt_cv_compiler_c_o], [
$rm -r conftest 2>/dev/null
mkdir conftest
cd conftest
echo "int some_variable = 0;" > conftest.$ac_ext
mkdir out
# According to Tom Tromey, Ian Lance Taylor reported there are C compilers
# that will create temporary files in the current directory regardless of
# the output directory.  Thus, making CWD read-only will cause this test
# to fail, enabling locking or at least warning the user not to do parallel
# builds.
chmod -w .
save_CFLAGS="$CFLAGS"
CFLAGS="$CFLAGS -o out/conftest2.$ac_objext"
compiler_c_o=no
if { (eval echo configure:__oline__: \"$ac_compile\") 1>&5; (eval $ac_compile) 2>out/conftest.err; } && test -s out/conftest2.$ac_objext; then
  # The compiler can only warn and ignore the option if not recognized
  # So say no if there are warnings
  if test -s out/conftest.err; then
    lt_cv_compiler_c_o=no
  else
    lt_cv_compiler_c_o=yes
  fi
else
  # Append any errors to the config.log.
  cat out/conftest.err 1>&AC_FD_CC
  lt_cv_compiler_c_o=no
fi
CFLAGS="$save_CFLAGS"
chmod u+w .
$rm conftest* out/*
rmdir out
cd ..
rmdir conftest
$rm -r conftest 2>/dev/null
])
compiler_c_o=$lt_cv_compiler_c_o
AC_MSG_RESULT([$compiler_c_o])

if test x"$compiler_c_o" = x"yes"; then
  # Check to see if we can write to a .lo
  AC_MSG_CHECKING([if $compiler supports -c -o file.lo])
  AC_CACHE_VAL([lt_cv_compiler_o_lo], [
  lt_cv_compiler_o_lo=no
  save_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -c -o conftest.lo"
  save_objext="$ac_objext"
  ac_objext=lo
  AC_TRY_COMPILE([], [int some_variable = 0;], [dnl
    # The compiler can only warn and ignore the option if not recognized
    # So say no if there are warnings
    if test -s conftest.err; then
      lt_cv_compiler_o_lo=no
    else
      lt_cv_compiler_o_lo=yes
    fi
  ])
  ac_objext="$save_objext"
  CFLAGS="$save_CFLAGS"
  ])
  compiler_o_lo=$lt_cv_compiler_o_lo
  AC_MSG_RESULT([$compiler_o_lo])
else
  compiler_o_lo=no
fi
##
## END FIXME

## FIXME: this should be a separate macro
##
# Check to see if we can do hard links to lock some files if needed
hard_links="nottested"
if test "$compiler_c_o" = no && test "$need_locks" != no; then
  # do not overwrite the value of need_locks provided by the user
  AC_MSG_CHECKING([if we can lock with hard links])
  hard_links=yes
  $rm conftest*
  ln conftest.a conftest.b 2>/dev/null && hard_links=no
  touch conftest.a
  ln conftest.a conftest.b 2>&5 || hard_links=no
  ln conftest.a conftest.b 2>/dev/null && hard_links=no
  AC_MSG_RESULT([$hard_links])
  if test "$hard_links" = no; then
    AC_MSG_WARN([\`$CC' does not support \`-c -o', so \`make -j' may be unsafe])
    need_locks=warn
  fi
else
  need_locks=no
fi
##
## END FIXME

## FIXME: this should be a separate macro
##
if test "$GCC" = yes; then
  # Check to see if options -fno-rtti -fno-exceptions are supported by compiler
  AC_MSG_CHECKING([if $compiler supports -fno-rtti -fno-exceptions])
  echo "int some_variable = 0;" > conftest.$ac_ext
  save_CFLAGS="$CFLAGS"
  CFLAGS="$CFLAGS -fno-rtti -fno-exceptions -c conftest.$ac_ext"
  compiler_rtti_exceptions=no
  AC_TRY_COMPILE([], [int some_variable = 0;], [dnl
    # The compiler can only warn and ignore the option if not recognized
    # So say no if there are warnings
    if test -s conftest.err; then
      compiler_rtti_exceptions=no
    else
      compiler_rtti_exceptions=yes
    fi
  ])
  CFLAGS="$save_CFLAGS"
  AC_MSG_RESULT([$compiler_rtti_exceptions])

  if test "$compiler_rtti_exceptions" = "yes"; then
    no_builtin_flag=' -fno-builtin -fno-rtti -fno-exceptions'
  else
    no_builtin_flag=' -fno-builtin'
  fi
fi
##
## END FIXME

## FIXME: this should be a separate macro
##
# See if the linker supports building shared libraries.
AC_MSG_CHECKING([whether the linker ($LD) supports shared libraries])

allow_undefined_flag=
no_undefined_flag=
need_lib_prefix=unknown
need_version=unknown
# when you set need_version to no, make sure it does not cause -set_version
# flags to be left without arguments
archive_cmds=
archive_expsym_cmds=
old_archive_from_new_cmds=
old_archive_from_expsyms_cmds=
export_dynamic_flag_spec=
whole_archive_flag_spec=
thread_safe_flag_spec=
hardcode_into_libs=no
hardcode_libdir_flag_spec=
hardcode_libdir_separator=
hardcode_direct=no
hardcode_minus_L=no
hardcode_shlibpath_var=unsupported
runpath_var=
link_all_deplibs=unknown
always_export_symbols=no
export_symbols_cmds='$NM $libobjs $convenience | $global_symbol_pipe | sed '\''s/.* //'\'' | sort | uniq > $export_symbols'
# include_expsyms should be a list of space-separated symbols to be *always*
# included in the symbol list
include_expsyms=
# exclude_expsyms can be an egrep regular expression of symbols to exclude
# it will be wrapped by ` (' and `)$', so one must not match beginning or
# end of line.  Example: `a|bc|.*d.*' will exclude the symbols `a' and `bc',
# as well as any symbol that contains `d'.
exclude_expsyms="_GLOBAL_OFFSET_TABLE_"
# Although _GLOBAL_OFFSET_TABLE_ is a valid symbol C name, most a.out
# platforms (ab)use it in PIC code, but their linkers get confused if
# the symbol is explicitly referenced.  Since portable code cannot
# rely on this symbol name, it's probably fine to never include it in
# preloaded symbol tables.
extract_expsyms_cmds=

case $host_os in
cygwin* | mingw* | pw32*)
  # FIXME: the MSVC++ port hasn't been tested in a loooong time
  # When not using gcc, we currently assume that we are using
  # Microsoft Visual C++.
  if test "$GCC" != yes; then
    with_gnu_ld=no
  fi
  ;;
openbsd*)
  with_gnu_ld=no
  ;;
esac

ld_shlibs=yes
if test "$with_gnu_ld" = yes; then
  # If archive_cmds runs LD, not CC, wlarc should be empty
  wlarc='${wl}'

  # See if GNU ld supports shared libraries.
  case $host_os in
  aix3* | aix4* | aix5*)
    # On AIX, the GNU linker is very broken
    # Note:Check GNU linker on AIX 5-IA64 when/if it becomes available.
    ld_shlibs=no
    cat <<EOF 1>&2

*** Warning: the GNU linker, at least up to release 2.9.1, is reported
*** to be unable to reliably create shared libraries on AIX.
*** Therefore, libtool is disabling shared libraries support.  If you
*** really care for shared libraries, you may want to modify your PATH
*** so that a non-GNU linker is found, and then restart.

EOF
    ;;

  amigaos*)
    archive_cmds='$rm $output_objdir/a2ixlibrary.data~$echo "#define NAME $libname" > $output_objdir/a2ixlibrary.data~$echo "#define LIBRARY_ID 1" >> $output_objdir/a2ixlibrary.data~$echo "#define VERSION $major" >> $output_objdir/a2ixlibrary.data~$echo "#define REVISION $revision" >> $output_objdir/a2ixlibrary.data~$AR $AR_FLAGS $lib $libobjs~$RANLIB $lib~(cd $output_objdir && a2ixlibrary -32)'
    hardcode_libdir_flag_spec='-L$libdir'
    hardcode_minus_L=yes

    # Samuel A. Falvo II <kc5tja@dolphin.openprojects.net> reports
    # that the semantics of dynamic libraries on AmigaOS, at least up
    # to version 4, is to share data among multiple programs linked
    # with the same dynamic library.  Since this doesn't match the
    # behavior of shared libraries on other platforms, we can use
    # them.
    ld_shlibs=no
    ;;

  beos*)
    if $LD --help 2>&1 | egrep ': supported targets:.* elf' > /dev/null; then
      allow_undefined_flag=unsupported
      # Joseph Beckenbach <jrb3@best.com> says some releases of gcc
      # support --undefined.  This deserves some investigation.  FIXME
      archive_cmds='$CC -nostart $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname -o $lib'
    else
      ld_shlibs=no
    fi
    ;;

  cygwin* | mingw* | pw32*)
    # hardcode_libdir_flag_spec is actually meaningless, as there is
    # no search path for DLLs.
    hardcode_libdir_flag_spec='-L$libdir'
    allow_undefined_flag=unsupported
    always_export_symbols=yes

    extract_expsyms_cmds='test -f $output_objdir/impgen.c || \
      sed -e "/^# \/\* impgen\.c starts here \*\//,/^# \/\* impgen.c ends here \*\// { s/^# //;s/^# *$//; p; }" -e d < $''0 > $output_objdir/impgen.c~
      test -f $output_objdir/impgen.exe || (cd $output_objdir && \
      if test "x$HOST_CC" != "x" ; then $HOST_CC -o impgen impgen.c ; \
      else $CC -o impgen impgen.c ; fi)~
      $output_objdir/impgen $dir/$soroot > $output_objdir/$soname-def'

    old_archive_from_expsyms_cmds='$DLLTOOL --as=$AS --dllname $soname --def $output_objdir/$soname-def --output-lib $output_objdir/$newlib'

    # cygwin and mingw dlls have different entry points and sets of symbols
    # to exclude.
    # FIXME: what about values for MSVC?
    dll_entry=__cygwin_dll_entry@12
    dll_exclude_symbols=DllMain@12,_cygwin_dll_entry@12,_cygwin_noncygwin_dll_entry@12~
    case $host_os in
    mingw*)
      # mingw values
      dll_entry=_DllMainCRTStartup@12
      dll_exclude_symbols=DllMain@12,DllMainCRTStartup@12,DllEntryPoint@12~
      ;;
    esac

    # mingw and cygwin differ, and it's simplest to just exclude the union
    # of the two symbol sets.
    dll_exclude_symbols=DllMain@12,_cygwin_dll_entry@12,_cygwin_noncygwin_dll_entry@12,DllMainCRTStartup@12,DllEntryPoint@12

    # recent cygwin and mingw systems supply a stub DllMain which the user
    # can override, but on older systems we have to supply one (in ltdll.c)
    if test "x$lt_cv_need_dllmain" = "xyes"; then
      ltdll_obj='$output_objdir/$soname-ltdll.'"$ac_objext "
      ltdll_cmds='test -f $output_objdir/$soname-ltdll.c || sed -e "/^# \/\* ltdll\.c starts here \*\//,/^# \/\* ltdll.c ends here \*\// { s/^# //; p; }" -e d < $''0 > $output_objdir/$soname-ltdll.c~
	test -f $output_objdir/$soname-ltdll.$ac_objext || (cd $output_objdir && $CC -c $soname-ltdll.c)~'
    else
      ltdll_obj=
      ltdll_cmds=
    fi

    # Extract the symbol export list from an `--export-all' def file,
    # then regenerate the def file from the symbol export list, so that
    # the compiled dll only exports the symbol export list.
    # Be careful not to strip the DATA tag left be newer dlltools.
    export_symbols_cmds="$ltdll_cmds"'
      $DLLTOOL --export-all --exclude-symbols '$dll_exclude_symbols' --output-def $output_objdir/$soname-def '$ltdll_obj'$libobjs $convenience~
      sed -e "1,/EXPORTS/d" -e "s/ @ [[0-9]]*//" -e "s/ *;.*$//" < $output_objdir/$soname-def > $export_symbols'

    # If the export-symbols file already is a .def file (1st line
    # is EXPORTS), use it as is.
    # If DATA tags from a recent dlltool are present, honour them!
    archive_expsym_cmds='if test "x`sed 1q $export_symbols`" = xEXPORTS; then
	cp $export_symbols $output_objdir/$soname-def;
      else
	echo EXPORTS > $output_objdir/$soname-def;
	_lt_hint=1;
	cat $export_symbols | while read symbol; do
	 set dummy \$symbol;
	 case \[$]# in
	   2) echo "   \[$]2 @ \$_lt_hint ; " >> $output_objdir/$soname-def;;
	   4) echo "   \[$]2 \[$]3 \[$]4 ; " >> $output_objdir/$soname-def; _lt_hint=`expr \$_lt_hint - 1`;;
	   *) echo "     \[$]2 @ \$_lt_hint \[$]3 ; " >> $output_objdir/$soname-def;;
	 esac;
	 _lt_hint=`expr 1 + \$_lt_hint`;
	done;
      fi~
      '"$ltdll_cmds"'
      $CC -Wl,--base-file,$output_objdir/$soname-base '$lt_cv_cc_dll_switch' -Wl,-e,'$dll_entry' -o $output_objdir/$soname '$ltdll_obj'$libobjs $deplibs $compiler_flags~
      $DLLTOOL --as=$AS --dllname $soname --exclude-symbols '$dll_exclude_symbols' --def $output_objdir/$soname-def --base-file $output_objdir/$soname-base --output-exp $output_objdir/$soname-exp~
      $CC -Wl,--base-file,$output_objdir/$soname-base $output_objdir/$soname-exp '$lt_cv_cc_dll_switch' -Wl,-e,'$dll_entry' -o $output_objdir/$soname '$ltdll_obj'$libobjs $deplibs $compiler_flags~
      $DLLTOOL --as=$AS --dllname $soname --exclude-symbols '$dll_exclude_symbols' --def $output_objdir/$soname-def --base-file $output_objdir/$soname-base --output-exp $output_objdir/$soname-exp --output-lib $output_objdir/$libname.dll.a~
      $CC $output_objdir/$soname-exp '$lt_cv_cc_dll_switch' -Wl,-e,'$dll_entry' -o $output_objdir/$soname '$ltdll_obj'$libobjs $deplibs $compiler_flags'
    ;;

  netbsd*)
    if echo __ELF__ | $CC -E - | grep __ELF__ >/dev/null; then
      archive_cmds='$LD -Bshareable $libobjs $deplibs $linker_flags -o $lib'
      wlarc=
    else
      archive_cmds='$CC -shared -nodefaultlibs $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname -o $lib'
      archive_expsym_cmds='$CC -shared -nodefaultlibs $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname ${wl}-retain-symbols-file $wl$export_symbols -o $lib'
    fi
    ;;

  solaris* | sysv5*)
    if $LD -v 2>&1 | egrep 'BFD 2\.8' > /dev/null; then
      ld_shlibs=no
      cat <<EOF 1>&2

*** Warning: The releases 2.8.* of the GNU linker cannot reliably
*** create shared libraries on Solaris systems.  Therefore, libtool
*** is disabling shared libraries support.  We urge you to upgrade GNU
*** binutils to release 2.9.1 or newer.  Another option is to modify
*** your PATH or compiler configuration so that the native linker is
*** used, and then restart.

EOF
    elif $LD --help 2>&1 | egrep ': supported targets:.* elf' > /dev/null; then
      archive_cmds='$CC -shared $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname -o $lib'
      archive_expsym_cmds='$CC -shared $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname ${wl}-retain-symbols-file $wl$export_symbols -o $lib'
    else
      ld_shlibs=no
    fi
    ;;

  sunos4*)
    archive_cmds='$LD -assert pure-text -Bshareable -o $lib $libobjs $deplibs $linker_flags'
    wlarc=
    hardcode_direct=yes
    hardcode_shlibpath_var=no
    ;;

  *)
    if $LD --help 2>&1 | egrep ': supported targets:.* elf' > /dev/null; then
      archive_cmds='$CC -shared $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname -o $lib'
      archive_expsym_cmds='$CC -shared $libobjs $deplibs $compiler_flags ${wl}-soname $wl$soname ${wl}-retain-symbols-file $wl$export_symbols -o $lib'
    else
      ld_shlibs=no
    fi
    ;;
  esac

  if test "$ld_shlibs" = yes; then
    runpath_var=LD_RUN_PATH
    hardcode_libdir_flag_spec='${wl}--rpath ${wl}$libdir'
    export_dynamic_flag_spec='${wl}--export-dynamic'
    case $host_os in
    cygwin* | mingw* | pw32*)
      # dlltool doesn't understand --whole-archive et. al.
      whole_archive_flag_spec=
      ;;
    *)
      # ancient GNU ld didn't support --whole-archive et. al.
      if $LD --help 2>&1 | egrep 'no-whole-archive' > /dev/null; then
	whole_archive_flag_spec="$wlarc"'--whole-archive$convenience '"$wlarc"'--no-whole-archive'
      else
	whole_archive_flag_spec=
      fi
      ;;
    esac
  fi
else
  # PORTME fill in a description of your system's linker (not GNU ld)
  case $host_os in
  aix3*)
    allow_undefined_flag=unsupported
    always_export_symbols=yes
    archive_expsym_cmds='$LD -o $output_objdir/$soname $libobjs $deplibs $linker_flags -bE:$export_symbols -T512 -H512 -bM:SRE~$AR $AR_FLAGS $lib $output_objdir/$soname'
    # Note: this linker hardcodes the directories in LIBPATH if there
    # are no directories specified by -L.
    hardcode_minus_L=yes
    if test "$GCC" = yes && test -z "$link_static_flag"; then
      # Neither direct hardcoding nor static linking is supported with a
      # broken collect2.
      hardcode_direct=unsupported
    fi
    ;;

  aix4* | aix5*)
    if test "$host_cpu" = ia64; then
      # On IA64, the linker does run time linking by default, so we don't
      # have to do anything special.
      aix_use_runtimelinking=no
      exp_sym_flag='-Bexport'
      no_entry_flag=""
    else
      aix_use_runtimelinking=no

      # Test if we are trying to use run time linking or normal
      # AIX style linking. If -brtl is somewhere in LDFLAGS, we
      # need to do runtime linking.
      case $host_os in aix4.[[23]]|aix4.[[23]].*|aix5*)
	for ld_flag in $LDFLAGS; do
	  case $ld_flag in
	  *-brtl*)
	    aix_use_runtimelinking=yes
	    break
	  ;;
	  esac
	done
      esac

      aix_use_runtimelinking=yes #gkk I use it always
      link_static_flag="-bstatic" #gkk added this
      exp_sym_flag='-bexport'
      no_entry_flag='-bnoentry'
    fi

    # When large executables or shared objects are built, AIX ld can
    # have problems creating the table of contents.  If linking a library
    # or program results in "error TOC overflow" add -mminimal-toc to
    # CXXFLAGS/CFLAGS for g++/gcc.  In the cases where that is not
    # enough to fix the problem, add -Wl,-bbigtoc to LDFLAGS.

    hardcode_direct=yes
    archive_cmds=''
    hardcode_libdir_separator=':'
    if test "$GCC" = yes; then
      case $host_os in aix4.[[012]]|aix4.[[012]].*)
	collect2name=`${CC} -print-prog-name=collect2`
	if test -f "$collect2name" && \
	  strings "$collect2name" | grep resolve_lib_name >/dev/null
	then
	  # We have reworked collect2
	  hardcode_direct=yes
	else
	  # We have old collect2
	  hardcode_direct=unsupported
	  # It fails to find uninstalled libraries when the uninstalled
	  # path is not listed in the libpath.  Setting hardcode_minus_L
	  # to unsupported forces relinking
	  hardcode_minus_L=yes
	  hardcode_libdir_flag_spec='-L$libdir'
	  hardcode_libdir_separator=
	fi
      esac

      shared_flag='-shared'
    else
      # not using gcc
      if test "$host_cpu" = ia64; then
	shared_flag='${wl}-G'
      else
	if test "$aix_use_runtimelinking" = yes; then
	  shared_flag='${wl}-G'
	else
	  shared_flag='${wl}-bM:SRE'
	fi
      fi
    fi

    # It seems that -bexpall can do strange things, so it is better to
    # generate a list of symbols to export.
    always_export_symbols=yes
    if test "$aix_use_runtimelinking" = yes; then
      # Warning - without using the other runtime loading flags (-brtl),
      # -berok will link without error, but may produce a broken library.
      allow_undefined_flag='-berok'
      hardcode_libdir_flag_spec='${wl}-blibpath:$libdir:/usr/lib:/lib'
      archive_expsym_cmds="\$CC"' -o $output_objdir/$soname $libobjs $deplibs $compiler_flags `if test "x${allow_undefined_flag}" != "x"; then echo "${wl}${allow_undefined_flag}"; else :; fi` '"\${wl}$no_entry_flag \${wl}$exp_sym_flag:\$export_symbols $shared_flag"
    else
      if test "$host_cpu" = ia64; then
	hardcode_libdir_flag_spec='${wl}-R $libdir:/usr/lib:/lib'
	allow_undefined_flag="-z nodefs"
	archive_expsym_cmds="\$CC $shared_flag"' -o $output_objdir/$soname ${wl}-h$soname $libobjs $deplibs $compiler_flags ${wl}${allow_undefined_flag} '"\${wl}$no_entry_flag \${wl}$exp_sym_flag:\$export_symbols"
      else
        #gkk I added -brtl here
	hardcode_libdir_flag_spec='${wl}-bnolibpath ${wl}-blibpath:$libdir:/usr/lib:/lib -brtl'
	# Warning - without using the other run time loading flags,
	# -berok will link without error, but may produce a broken library.
	allow_undefined_flag='${wl}-berok'
	# This is a bit strange, but is similar to how AIX traditionally builds
	# it's shared libraries.
	archive_expsym_cmds="\$CC $shared_flag"' -o $output_objdir/$soname $libobjs $deplibs $compiler_flags ${allow_undefined_flag} '"\${wl}$no_entry_flag \${wl}$exp_sym_flag:\$export_symbols"' ~$AR -crlo $objdir/$libname$release.a $objdir/$soname'
      fi
    fi
    ;;

  amigaos*)
    archive_cmds='$rm $output_objdir/a2ixlibrary.data~$echo "#define NAME $libname" > $output_objdir/a2ixlibrary.data~$echo "#define LIBRARY_ID 1" >> $output_objdir/a2ixlibrary.data~$echo "#define VERSION $major" >> $output_objdir/a2ixlibrary.data~$echo "#define REVISION $revision" >> $output_objdir/a2ixlibrary.data~$AR $AR_FLAGS $lib $libobjs~$RANLIB $lib~(cd $output_objdir && a2ixlibrary -32)'
    hardcode_libdir_flag_spec='-L$libdir'
    hardcode_minus_L=yes
    # see comment about different semantics on the GNU ld section
    ld_shlibs=no
    ;;

  cygwin* | mingw* | pw32*)
    # When not using gcc, we currently assume that we are using
    # Microsoft Visual C++.
    # hardcode_libdir_flag_spec is actually meaningless, as there is
    # no search path for DLLs.
    hardcode_libdir_flag_spec=' '
    allow_undefined_flag=unsupported
    # Tell ltmain to make .lib files, not .a files.
    libext=lib
    # FIXME: Setting linknames here is a bad hack.
    archive_cmds='$CC -o $lib $libobjs $compiler_flags `echo "$deplibs" | sed -e '\''s/ -lc$//'\''` -link -dll~linknames='
    # The linker will automatically build a .lib file if we build a DLL.
    old_archive_from_new_cmds='true'
    # FIXME: Should let the user specify the lib program.
    old_archive_cmds='lib /OUT:$oldlib$oldobjs$old_deplibs'
    fix_srcfile_path='`cygpath -w "$srcfile"`'
    ;;

  darwin* | rhapsody*)
    case "$host_os" in
    rhapsody* | darwin1.[[012]])
      allow_undefined_flag='-undefined suppress'
      ;;
    *) # Darwin 1.3 on
      allow_undefined_flag='-flat_namespace -undefined suppress'
      ;;
    esac
    # FIXME: Relying on posixy $() will cause problems for
    #        cross-compilation, but unfortunately the echo tests do not
    #        yet detect zsh echo's removal of \ escapes.  Also zsh mangles
    #	     `"' quotes if we put them in here... so don't!
    archive_cmds='$CC -r -keep_private_externs -nostdlib -o ${lib}-master.o $libobjs && $CC $(test .$module = .yes && echo -bundle || echo -dynamiclib) $allow_undefined_flag -o $lib ${lib}-master.o $deplibs$linker_flags $(test .$module != .yes && echo -install_name $rpath/$soname $verstring)'
    # We need to add '_' to the symbols in $export_symbols first
    #archive_expsym_cmds="$archive_cmds"' && strip -s $export_symbols'
    hardcode_direct=yes
    hardcode_shlibpath_var=no
    whole_archive_flag_spec='-all_load $convenience'
    ;;

  freebsd1*)
    ld_shlibs=no
    ;;

  # FreeBSD 2.2.[012] allows us to include c++rt0.o to get C++ constructor
  # support.  Future versions do this automatically, but an explicit c++rt0.o
  # does not break anything, and helps significantly (at the cost of a little
  # extra space).
  freebsd2.2*)
    archive_cmds='$LD -Bshareable -o $lib $libobjs $deplibs $linker_flags /usr/lib/c++rt0.o'
    hardcode_libdir_flag_spec='-R$libdir'
    hardcode_direct=yes
    hardcode_shlibpath_var=no
    ;;

  # Unfortunately, older versions of FreeBSD 2 do not have this feature.
  freebsd2*)
    archive_cmds='$LD -Bshareable -o $lib $libobjs $deplibs $linker_flags'
    hardcode_direct=yes
    hardcode_minus_L=yes
    hardcode_shlibpath_var=no
    ;;

  # FreeBSD 3 and greater uses gcc -shared to do shared libraries.
  freebsd*)
    archive_cmds='$CC -shared -o $lib $libobjs $deplibs $compiler_flags'
    hardcode_libdir_flag_spec='-R$libdir'
    hardcode_direct=yes
    hardcode_shlibpath_var=no
    ;;

  hpux9* | hpux10* | hpux11*)
    case $host_os in
    hpux9*) archive_cmds='$rm $output_objdir/$soname~$LD -b +b $install_libdir -o $output_objdir/$soname $libobjs $deplibs $linker_flags~test $output_objdir/$soname = $lib || mv $output_objdir/$soname $lib' ;;
    *) archive_cmds='$LD -b +h $soname +b $install_libdir -o $lib $libobjs $deplibs $linker_flags' ;;
    esac
    hardcode_libdir_flag_spec='${wl}+b ${wl}$libdir'
    hardcode_libdir_separator=:
    hardcode_direct=yes
    hardcode_minus_L=yes # Not in the search PATH, but as the default
			 # location of the library.
    export_dynamic_flag_spec='${wl}-E'
    ;;

  irix5* | irix6* | nonstopux*)
    if test "$GCC" = yes; then
      archive_cmds='$CC -shared $libobjs $deplibs $compiler_flags ${wl}-soname ${wl}$soname `test -n "$verstring" && echo ${wl}-set_version ${wl}$verstring` ${wl}-update_registry ${wl}${output_objdir}/so_locations -o $lib'
      hardcode_libdir_flag_spec='${wl}-rpath ${wl}$libdir'
    else
      archive_cmds='$LD -shared $libobjs $deplibs $linker_flags -soname $soname `test -n "$verstring" && echo -set_version $verstring` -update_registry ${output_objdir}/so_locations -o $lib'
      hardcode_libdir_flag_spec='-rpath $libdir'
    fi
    hardcode_libdir_separator=:
    link_all_deplibs=yes
    ;;

  netbsd*)
    if echo __ELF__ | $CC -E - | grep __ELF__ >/dev/null; then
      archive_cmds='$LD -Bshareable -o $lib $libobjs $deplibs $linker_flags'  # a.out
    else
      archive_cmds='$LD -shared -o $lib $libobjs $deplibs $linker_flags'      # ELF
    fi
    hardcode_libdir_flag_spec='-R$libdir'
    hardcode_direct=yes
    hardcode_shlibpath_var=no
    ;;

  newsos6)
    archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
    hardcode_direct=yes
    hardcode_libdir_flag_spec='${wl}-rpath ${wl}$libdir'
    hardcode_libdir_separator=:
    hardcode_shlibpath_var=no
    ;;

  openbsd*)
    hardcode_direct=yes
    hardcode_shlibpath_var=no
    if test -z "`echo __ELF__ | $CC -E - | grep __ELF__`" || test "$host_os-$host_cpu" = "openbsd2.8-powerpc"; then
      archive_cmds='$CC -shared $pic_flag -o $lib $libobjs $deplibs $compiler_flags'
      hardcode_libdir_flag_spec='${wl}-rpath,$libdir'
      export_dynamic_flag_spec='${wl}-E'
    else
      case "$host_os" in
      openbsd[[01]].* | openbsd2.[[0-7]] | openbsd2.[[0-7]].*)
	archive_cmds='$LD -Bshareable -o $lib $libobjs $deplibs $linker_flags'
	hardcode_libdir_flag_spec='-R$libdir'
        ;;
      *)
        archive_cmds='$CC -shared $pic_flag -o $lib $libobjs $deplibs $compiler_flags'
        hardcode_libdir_flag_spec='${wl}-rpath,$libdir'
        ;;
      esac
    fi
    ;;

  os2*)
    hardcode_libdir_flag_spec='-L$libdir'
    hardcode_minus_L=yes
    allow_undefined_flag=unsupported
    archive_cmds='$echo "LIBRARY $libname INITINSTANCE" > $output_objdir/$libname.def~$echo "DESCRIPTION \"$libname\"" >> $output_objdir/$libname.def~$echo DATA >> $output_objdir/$libname.def~$echo " SINGLE NONSHARED" >> $output_objdir/$libname.def~$echo EXPORTS >> $output_objdir/$libname.def~emxexp $libobjs >> $output_objdir/$libname.def~$CC -Zdll -Zcrtdll -o $lib $libobjs $deplibs $compiler_flags $output_objdir/$libname.def'
    old_archive_from_new_cmds='emximp -o $output_objdir/$libname.a $output_objdir/$libname.def'
    ;;

  osf3*)
    if test "$GCC" = yes; then
      allow_undefined_flag=' ${wl}-expect_unresolved ${wl}\*'
      archive_cmds='$CC -shared${allow_undefined_flag} $libobjs $deplibs $compiler_flags ${wl}-soname ${wl}$soname `test -n "$verstring" && echo ${wl}-set_version ${wl}$verstring` ${wl}-update_registry ${wl}${output_objdir}/so_locations -o $lib'
    else
      allow_undefined_flag=' -expect_unresolved \*'
      archive_cmds='$LD -shared${allow_undefined_flag} $libobjs $deplibs $linker_flags -soname $soname `test -n "$verstring" && echo -set_version $verstring` -update_registry ${output_objdir}/so_locations -o $lib'
    fi
    hardcode_libdir_flag_spec='${wl}-rpath ${wl}$libdir'
    hardcode_libdir_separator=:
    ;;

  osf4* | osf5*)	# as osf3* with the addition of -msym flag
    if test "$GCC" = yes; then
      allow_undefined_flag=' ${wl}-expect_unresolved ${wl}\*'
      archive_cmds='$CC -shared${allow_undefined_flag} $libobjs $deplibs $compiler_flags ${wl}-msym ${wl}-soname ${wl}$soname `test -n "$verstring" && echo ${wl}-set_version ${wl}$verstring` ${wl}-update_registry ${wl}${output_objdir}/so_locations -o $lib'
      hardcode_libdir_flag_spec='${wl}-rpath ${wl}$libdir'
    else
      allow_undefined_flag=' -expect_unresolved \*'
      archive_cmds='$LD -shared${allow_undefined_flag} $libobjs $deplibs $linker_flags -msym -soname $soname `test -n "$verstring" && echo -set_version $verstring` -update_registry ${output_objdir}/so_locations -o $lib'
      archive_expsym_cmds='for i in `cat $export_symbols`; do printf "-exported_symbol " >> $lib.exp; echo "\$i" >> $lib.exp; done; echo "-hidden">> $lib.exp~
      $LD -shared${allow_undefined_flag} -input $lib.exp $linker_flags $libobjs $deplibs -soname $soname `test -n "$verstring" && echo -set_version $verstring` -update_registry ${objdir}/so_locations -o $lib~$rm $lib.exp'

      #Both c and cxx compiler support -rpath directly
      hardcode_libdir_flag_spec='-rpath $libdir'
    fi
    hardcode_libdir_separator=:
    ;;

  sco3.2v5*)
    archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
    hardcode_shlibpath_var=no
    runpath_var=LD_RUN_PATH
    hardcode_runpath_var=yes
    export_dynamic_flag_spec='${wl}-Bexport'
    ;;

  solaris*)
    # gcc --version < 3.0 without binutils cannot create self contained
    # shared libraries reliably, requiring libgcc.a to resolve some of
    # the object symbols generated in some cases.  Libraries that use
    # assert need libgcc.a to resolve __eprintf, for example.  Linking
    # a copy of libgcc.a into every shared library to guarantee resolving
    # such symbols causes other problems:  According to Tim Van Holder
    # <tim.van.holder@pandora.be>, C++ libraries end up with a separate
    # (to the application) exception stack for one thing.
    no_undefined_flag=' -z defs'
    if test "$GCC" = yes; then
      case `$CC --version 2>/dev/null` in
      [[12]].*)
	cat <<EOF 1>&2

*** Warning: Releases of GCC earlier than version 3.0 cannot reliably
*** create self contained shared libraries on Solaris systems, without
*** introducing a dependency on libgcc.a.  Therefore, libtool is disabling
*** -no-undefined support, which will at least allow you to build shared
*** libraries.  However, you may find that when you link such libraries
*** into an application without using GCC, you have to manually add
*** \`gcc --print-libgcc-file-name\` to the link command.  We urge you to
*** upgrade to a newer version of GCC.  Another option is to rebuild your
*** current GCC to use the GNU linker from GNU binutils 2.9.1 or newer.

EOF
        no_undefined_flag=
	;;
      esac
    fi
    # $CC -shared without GNU ld will not create a library from C++
    # object files and a static libstdc++, better avoid it by now
    archive_cmds='$LD -G${allow_undefined_flag} -h $soname -o $lib $libobjs $deplibs $linker_flags'
    archive_expsym_cmds='$echo "{ global:" > $lib.exp~cat $export_symbols | sed -e "s/\(.*\)/\1;/" >> $lib.exp~$echo "local: *; };" >> $lib.exp~
		$LD -G${allow_undefined_flag} -M $lib.exp -h $soname -o $lib $libobjs $deplibs $linker_flags~$rm $lib.exp'
    hardcode_libdir_flag_spec='-R$libdir'
    hardcode_shlibpath_var=no
    case $host_os in
    solaris2.[[0-5]] | solaris2.[[0-5]].*) ;;
    *) # Supported since Solaris 2.6 (maybe 2.5.1?)
      whole_archive_flag_spec='-z allextract$convenience -z defaultextract' ;;
    esac
    link_all_deplibs=yes
    ;;

  sunos4*)
    if test "x$host_vendor" = xsequent; then
      # Use $CC to link under sequent, because it throws in some extra .o
      # files that make .init and .fini sections work.
      archive_cmds='$CC -G ${wl}-h $soname -o $lib $libobjs $deplibs $compiler_flags'
    else
      archive_cmds='$LD -assert pure-text -Bstatic -o $lib $libobjs $deplibs $linker_flags'
    fi
    hardcode_libdir_flag_spec='-L$libdir'
    hardcode_direct=yes
    hardcode_minus_L=yes
    hardcode_shlibpath_var=no
    ;;

  sysv4)
    case $host_vendor in
      sni)
        archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
        hardcode_direct=yes # is this really true???
        ;;
      siemens)
        ## LD is ld it makes a PLAMLIB
        ## CC just makes a GrossModule.
        archive_cmds='$LD -G -o $lib $libobjs $deplibs $linker_flags'
        reload_cmds='$CC -r -o $output$reload_objs'
        hardcode_direct=no
        ;;
      motorola)
        archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
        hardcode_direct=no #Motorola manual says yes, but my tests say they lie
        ;;
    esac
    runpath_var='LD_RUN_PATH'
    hardcode_shlibpath_var=no
    ;;

  sysv4.3*)
    archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
    hardcode_shlibpath_var=no
    export_dynamic_flag_spec='-Bexport'
    ;;

  sysv5*)
    no_undefined_flag=' -z text'
    # $CC -shared without GNU ld will not create a library from C++
    # object files and a static libstdc++, better avoid it by now
    archive_cmds='$LD -G${allow_undefined_flag} -h $soname -o $lib $libobjs $deplibs $linker_flags'
    archive_expsym_cmds='$echo "{ global:" > $lib.exp~cat $export_symbols | sed -e "s/\(.*\)/\1;/" >> $lib.exp~$echo "local: *; };" >> $lib.exp~
		$LD -G${allow_undefined_flag} -M $lib.exp -h $soname -o $lib $libobjs $deplibs $linker_flags~$rm $lib.exp'
    hardcode_libdir_flag_spec=
    hardcode_shlibpath_var=no
    runpath_var='LD_RUN_PATH'
    ;;

  uts4*)
    archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
    hardcode_libdir_flag_spec='-L$libdir'
    hardcode_shlibpath_var=no
    ;;

  dgux*)
    archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
    hardcode_libdir_flag_spec='-L$libdir'
    hardcode_shlibpath_var=no
    ;;

  sysv4*MP*)
    if test -d /usr/nec; then
      archive_cmds='$LD -G -h $soname -o $lib $libobjs $deplibs $linker_flags'
      hardcode_shlibpath_var=no
      runpath_var=LD_RUN_PATH
      hardcode_runpath_var=yes
      ld_shlibs=yes
    fi
    ;;

  sysv4.2uw2*)
    archive_cmds='$LD -G -o $lib $libobjs $deplibs $linker_flags'
    hardcode_direct=yes
    hardcode_minus_L=no
    hardcode_shlibpath_var=no
    hardcode_runpath_var=yes
    runpath_var=LD_RUN_PATH
    ;;

  sysv5uw7* | unixware7*)
    no_undefined_flag='${wl}-z ${wl}text'
    if test "$GCC" = yes; then
      archive_cmds='$CC -shared ${wl}-h ${wl}$soname -o $lib $libobjs $deplibs $compiler_flags'
    else
      archive_cmds='$CC -G ${wl}-h ${wl}$soname -o $lib $libobjs $deplibs $compiler_flags'
    fi
    runpath_var='LD_RUN_PATH'
    hardcode_shlibpath_var=no
    ;;

  *)
    ld_shlibs=no
    ;;
  esac
fi
AC_MSG_RESULT([$ld_shlibs])
test "$ld_shlibs" = no && can_build_shared=no
##
## END FIXME

## FIXME: this should be a separate macro
##
# Check hardcoding attributes.
AC_MSG_CHECKING([how to hardcode library paths into programs])
hardcode_action=
if test -n "$hardcode_libdir_flag_spec" || \
   test -n "$runpath_var"; then

  # We can hardcode non-existant directories.
  if test "$hardcode_direct" != no &&
     # If the only mechanism to avoid hardcoding is shlibpath_var, we
     # have to relink, otherwise we might link with an installed library
     # when we should be linking with a yet-to-be-installed one
     ## test "$hardcode_shlibpath_var" != no &&
     test "$hardcode_minus_L" != no; then
    # Linking always hardcodes the temporary library directory.
    hardcode_action=relink
  else
    # We can link without hardcoding, and we can hardcode nonexisting dirs.
    hardcode_action=immediate
  fi
else
  # We cannot hardcode anything, or else we can only hardcode existing
  # directories.
  hardcode_action=unsupported
fi
AC_MSG_RESULT([$hardcode_action])
##
## END FIXME

## FIXME: this should be a separate macro
##
striplib=
old_striplib=
AC_MSG_CHECKING([whether stripping libraries is possible])
if test -n "$STRIP" && $STRIP -V 2>&1 | grep "GNU strip" >/dev/null; then
  test -z "$old_striplib" && old_striplib="$STRIP --strip-debug"
  test -z "$striplib" && striplib="$STRIP --strip-unneeded"
  AC_MSG_RESULT([yes])
else
  AC_MSG_RESULT([no])
fi
##
## END FIXME

reload_cmds='$LD$reload_flag -o $output$reload_objs'
test -z "$deplibs_check_method" && deplibs_check_method=unknown

## FIXME: this should be a separate macro
##
# PORTME Fill in your ld.so characteristics
AC_MSG_CHECKING([dynamic linker characteristics])
library_names_spec=
libname_spec='lib$name'
soname_spec=
postinstall_cmds=
postuninstall_cmds=
finish_cmds=
finish_eval=
shlibpath_var=
shlibpath_overrides_runpath=unknown
version_type=none
dynamic_linker="$host_os ld.so"
sys_lib_dlsearch_path_spec="/lib /usr/lib"
sys_lib_search_path_spec="/lib /usr/lib /usr/local/lib"

case $host_os in
aix3*)
  version_type=linux
  library_names_spec='${libname}${release}.so$versuffix $libname.a'
  shlibpath_var=LIBPATH

  # AIX has no versioning support, so we append a major version to the name.
  soname_spec='${libname}${release}.so$major'
  ;;

aix4* | aix5*)
  version_type=linux
  need_lib_prefix=no
  need_version=no
  hardcode_into_libs=yes
  if test "$host_cpu" = ia64; then
    # AIX 5 supports IA64
    library_names_spec='${libname}${release}.so$major ${libname}${release}.so$versuffix $libname.so'
    shlibpath_var=LD_LIBRARY_PATH
  else
    # With GCC up to 2.95.x, collect2 would create an import file
    # for dependence libraries.  The import file would start with
    # the line `#! .'.  This would cause the generated library to
    # depend on `.', always an invalid library.  This was fixed in
    # development snapshots of GCC prior to 3.0.
    case $host_os in
      aix4 | aix4.[[01]] | aix4.[[01]].*)
	if { echo '#if __GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 97)'
	     echo ' yes '
	     echo '#endif'; } | ${CC} -E - | grep yes > /dev/null; then
	  :
	else
	  can_build_shared=no
	fi
	;;
    esac
    # AIX (on Power*) has no versioning support, so currently we can
    # not hardcode correct soname into executable. Probably we can
    # add versioning support to collect2, so additional links can
    # be useful in future.
    if test "$aix_use_runtimelinking" = yes; then
      # If using run time linking (on AIX 4.2 or later) use lib<name>.so
      # instead of lib<name>.a to let people know that these are not
      # typical AIX shared libraries.
      library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
    else
      # We preserve .a as extension for shared libraries through AIX4.2
      # and later when we are not doing run time linking.
      library_names_spec='${libname}${release}.a $libname.a'
      soname_spec='${libname}${release}.so$major'
    fi
    shlibpath_var=LIBPATH
  fi
  hardcode_into_libs=yes
  ;;

amigaos*)
  library_names_spec='$libname.ixlibrary $libname.a'
  # Create ${libname}_ixlibrary.a entries in /sys/libs.
  finish_eval='for lib in `ls $libdir/*.ixlibrary 2>/dev/null`; do libname=`$echo "X$lib" | $Xsed -e '\''s%^.*/\([[^/]]*\)\.ixlibrary$%\1%'\''`; test $rm /sys/libs/${libname}_ixlibrary.a; $show "(cd /sys/libs && $LN_S $lib ${libname}_ixlibrary.a)"; (cd /sys/libs && $LN_S $lib ${libname}_ixlibrary.a) || exit 1; done'
  ;;

beos*)
  library_names_spec='${libname}.so'
  dynamic_linker="$host_os ld.so"
  shlibpath_var=LIBRARY_PATH
  ;;

bsdi4*)
  version_type=linux
  need_version=no
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  soname_spec='${libname}${release}.so$major'
  finish_cmds='PATH="\$PATH:/sbin" ldconfig $libdir'
  shlibpath_var=LD_LIBRARY_PATH
  sys_lib_search_path_spec="/shlib /usr/lib /usr/X11/lib /usr/contrib/lib /lib /usr/local/lib"
  sys_lib_dlsearch_path_spec="/shlib /usr/lib /usr/local/lib"
  export_dynamic_flag_spec=-rdynamic
  # the default ld.so.conf also contains /usr/contrib/lib and
  # /usr/X11R6/lib (/usr/X11 is a link to /usr/X11R6), but let us allow
  # libtool to hard-code these into programs
  ;;

cygwin* | mingw* | pw32*)
  version_type=windows
  need_version=no
  need_lib_prefix=no
  case $GCC,$host_os in
  yes,cygwin*)
    library_names_spec='$libname.dll.a'
    soname_spec='`echo ${libname} | sed -e 's/^lib/cyg/'``echo ${release} | sed -e 's/[[.]]/-/g'`${versuffix}.dll'
    postinstall_cmds='dlpath=`bash 2>&1 -c '\''. $dir/${file}i;echo \$dlname'\''`~
      dldir=$destdir/`dirname \$dlpath`~
      test -d \$dldir || mkdir -p \$dldir~
      $install_prog .libs/$dlname \$dldir/$dlname'
    postuninstall_cmds='dldll=`bash 2>&1 -c '\''. $file; echo \$dlname'\''`~
      dlpath=$dir/\$dldll~
       $rm \$dlpath'
    ;;
  yes,mingw*)
    library_names_spec='${libname}`echo ${release} | sed -e 's/[[.]]/-/g'`${versuffix}.dll'
    sys_lib_search_path_spec=`$CC -print-search-dirs | grep "^libraries:" | sed -e "s/^libraries://" -e "s/;/ /g" -e "s,=/,/,g"`
    ;;
  yes,pw32*)
    library_names_spec='`echo ${libname} | sed -e 's/^lib/pw/'``echo ${release} | sed -e 's/[.]/-/g'`${versuffix}.dll'
    ;;
  *)
    library_names_spec='${libname}`echo ${release} | sed -e 's/[[.]]/-/g'`${versuffix}.dll $libname.lib'
    ;;
  esac
  dynamic_linker='Win32 ld.exe'
  # FIXME: first we should search . and the directory the executable is in
  shlibpath_var=PATH
  ;;

darwin* | rhapsody*)
  dynamic_linker="$host_os dyld"
  version_type=darwin
  need_lib_prefix=no
  need_version=no
  # FIXME: Relying on posixy $() will cause problems for
  #        cross-compilation, but unfortunately the echo tests do not
  #        yet detect zsh echo's removal of \ escapes.
  library_names_spec='${libname}${release}${versuffix}.$(test .$module = .yes && echo so || echo dylib) ${libname}${release}${major}.$(test .$module = .yes && echo so || echo dylib) ${libname}.$(test .$module = .yes && echo so || echo dylib)'
  soname_spec='${libname}${release}${major}.$(test .$module = .yes && echo so || echo dylib)'
  shlibpath_overrides_runpath=yes
  shlibpath_var=DYLD_LIBRARY_PATH
  ;;

freebsd1*)
  dynamic_linker=no
  ;;

freebsd*)
  objformat=`test -x /usr/bin/objformat && /usr/bin/objformat || echo aout`
  version_type=freebsd-$objformat
  case $version_type in
    freebsd-elf*)
      library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so $libname.so'
      need_version=no
      need_lib_prefix=no
      ;;
    freebsd-*)
      library_names_spec='${libname}${release}.so$versuffix $libname.so$versuffix'
      need_version=yes
      ;;
  esac
  shlibpath_var=LD_LIBRARY_PATH
  case $host_os in
  freebsd2*)
    shlibpath_overrides_runpath=yes
    ;;
  *)
    shlibpath_overrides_runpath=no
    hardcode_into_libs=yes
    ;;
  esac
  ;;

gnu*)
  version_type=linux
  need_lib_prefix=no
  need_version=no
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so${major} ${libname}.so'
  soname_spec='${libname}${release}.so$major'
  shlibpath_var=LD_LIBRARY_PATH
  hardcode_into_libs=yes
  ;;

hpux9* | hpux10* | hpux11*)
  # Give a soname corresponding to the major version so that dld.sl refuses to
  # link against other versions.
  dynamic_linker="$host_os dld.sl"
  version_type=sunos
  need_lib_prefix=no
  need_version=no
  shlibpath_var=SHLIB_PATH
  shlibpath_overrides_runpath=no # +s is required to enable SHLIB_PATH
  library_names_spec='${libname}${release}.sl$versuffix ${libname}${release}.sl$major $libname.sl'
  soname_spec='${libname}${release}.sl$major'
  # HP-UX runs *really* slowly unless shared libraries are mode 555.
  postinstall_cmds='chmod 555 $lib'
  ;;

irix5* | irix6* | nonstopux*)
  case $host_os in
    nonstopux*) version_type=nonstopux ;;
    *)          version_type=irix ;;
  esac
  need_lib_prefix=no
  need_version=no
  soname_spec='${libname}${release}.so$major'
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major ${libname}${release}.so $libname.so'
  case $host_os in
  irix5* | nonstopux*)
    libsuff= shlibsuff=
    ;;
  *)
    case $LD in # libtool.m4 will add one of these switches to LD
    *-32|*"-32 ") libsuff= shlibsuff= libmagic=32-bit;;
    *-n32|*"-n32 ") libsuff=32 shlibsuff=N32 libmagic=N32;;
    *-64|*"-64 ") libsuff=64 shlibsuff=64 libmagic=64-bit;;
    *) libsuff= shlibsuff= libmagic=never-match;;
    esac
    ;;
  esac
  shlibpath_var=LD_LIBRARY${shlibsuff}_PATH
  shlibpath_overrides_runpath=no
  sys_lib_search_path_spec="/usr/lib${libsuff} /lib${libsuff} /usr/local/lib${libsuff}"
  sys_lib_dlsearch_path_spec="/usr/lib${libsuff} /lib${libsuff}"
  ;;

# No shared lib support for Linux oldld, aout, or coff.
linux-gnuoldld* | linux-gnuaout* | linux-gnucoff*)
  dynamic_linker=no
  ;;

# This must be Linux ELF.
linux-gnu*)
  version_type=linux
  need_lib_prefix=no
  need_version=no
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  soname_spec='${libname}${release}.so$major'
  finish_cmds='PATH="\$PATH:/sbin" ldconfig -n $libdir'
  shlibpath_var=LD_LIBRARY_PATH
  shlibpath_overrides_runpath=no
  # This implies no fast_install, which is unacceptable.
  # Some rework will be needed to allow for fast_install
  # before this can be enabled.
  hardcode_into_libs=yes

  # We used to test for /lib/ld.so.1 and disable shared libraries on
  # powerpc, because MkLinux only supported shared libraries with the
  # GNU dynamic linker.  Since this was broken with cross compilers,
  # most powerpc-linux boxes support dynamic linking these days and
  # people can always --disable-shared, the test was removed, and we
  # assume the GNU/Linux dynamic linker is in use.
  dynamic_linker='GNU/Linux ld.so'
  ;;

netbsd*)
  version_type=sunos
  need_lib_prefix=no
  need_version=no
  if echo __ELF__ | $CC -E - | grep __ELF__ >/dev/null; then
    library_names_spec='${libname}${release}.so$versuffix ${libname}.so$versuffix'
    finish_cmds='PATH="\$PATH:/sbin" ldconfig -m $libdir'
    dynamic_linker='NetBSD (a.out) ld.so'
  else
    library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major ${libname}${release}.so ${libname}.so'
    soname_spec='${libname}${release}.so$major'
    dynamic_linker='NetBSD ld.elf_so'
  fi
  shlibpath_var=LD_LIBRARY_PATH
  shlibpath_overrides_runpath=yes
  hardcode_into_libs=yes
  ;;

newsos6)
  version_type=linux
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  shlibpath_var=LD_LIBRARY_PATH
  shlibpath_overrides_runpath=yes
  ;;

openbsd*)
  version_type=sunos
  need_lib_prefix=no
  need_version=no
  if test -z "`echo __ELF__ | $CC -E - | grep __ELF__`" || test "$host_os-$host_cpu" = "openbsd2.8-powerpc"; then
    case "$host_os" in
    openbsd2.[[89]] | openbsd2.[[89]].*)
      shlibpath_overrides_runpath=no
      ;;
    *)
      shlibpath_overrides_runpath=yes
      ;;
    esac
  else
    shlibpath_overrides_runpath=yes
  fi
  library_names_spec='${libname}${release}.so$versuffix ${libname}.so$versuffix'
  finish_cmds='PATH="\$PATH:/sbin" ldconfig -m $libdir'
  shlibpath_var=LD_LIBRARY_PATH
  ;;

os2*)
  libname_spec='$name'
  need_lib_prefix=no
  library_names_spec='$libname.dll $libname.a'
  dynamic_linker='OS/2 ld.exe'
  shlibpath_var=LIBPATH
  ;;

osf3* | osf4* | osf5*)
  version_type=osf
  need_version=no
  soname_spec='${libname}${release}.so$major'
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  shlibpath_var=LD_LIBRARY_PATH
  sys_lib_search_path_spec="/usr/shlib /usr/ccs/lib /usr/lib/cmplrs/cc /usr/lib /usr/local/lib /var/shlib"
  sys_lib_dlsearch_path_spec="$sys_lib_search_path_spec"
  hardcode_into_libs=yes
  ;;

sco3.2v5*)
  version_type=osf
  soname_spec='${libname}${release}.so$major'
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  shlibpath_var=LD_LIBRARY_PATH
  ;;

solaris*)
  version_type=linux
  need_lib_prefix=no
  need_version=no
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  soname_spec='${libname}${release}.so$major'
  shlibpath_var=LD_LIBRARY_PATH
  shlibpath_overrides_runpath=yes
  hardcode_into_libs=yes
  # ldd complains unless libraries are executable
  postinstall_cmds='chmod +x $lib'
  ;;

sunos4*)
  version_type=sunos
  library_names_spec='${libname}${release}.so$versuffix ${libname}.so$versuffix'
  finish_cmds='PATH="\$PATH:/usr/etc" ldconfig $libdir'
  shlibpath_var=LD_LIBRARY_PATH
  shlibpath_overrides_runpath=yes
  if test "$with_gnu_ld" = yes; then
    need_lib_prefix=no
  fi
  need_version=yes
  ;;

sysv4 | sysv4.2uw2* | sysv4.3* | sysv5*)
  version_type=linux
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  soname_spec='${libname}${release}.so$major'
  shlibpath_var=LD_LIBRARY_PATH
  case $host_vendor in
    sni)
      shlibpath_overrides_runpath=no
      need_lib_prefix=no
      export_dynamic_flag_spec='${wl}-Blargedynsym'
      runpath_var=LD_RUN_PATH
      ;;
    siemens)
      need_lib_prefix=no
      ;;
    motorola)
      need_lib_prefix=no
      need_version=no
      shlibpath_overrides_runpath=no
      sys_lib_search_path_spec='/lib /usr/lib /usr/ccs/lib'
      ;;
  esac
  ;;

uts4*)
  version_type=linux
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  soname_spec='${libname}${release}.so$major'
  shlibpath_var=LD_LIBRARY_PATH
  ;;

dgux*)
  version_type=linux
  need_lib_prefix=no
  need_version=no
  library_names_spec='${libname}${release}.so$versuffix ${libname}${release}.so$major $libname.so'
  soname_spec='${libname}${release}.so$major'
  shlibpath_var=LD_LIBRARY_PATH
  ;;

sysv4*MP*)
  if test -d /usr/nec ;then
    version_type=linux
    library_names_spec='$libname.so.$versuffix $libname.so.$major $libname.so'
    soname_spec='$libname.so.$major'
    shlibpath_var=LD_LIBRARY_PATH
  fi
  ;;

*)
  dynamic_linker=no
  ;;
esac
AC_MSG_RESULT([$dynamic_linker])
test "$dynamic_linker" = no && can_build_shared=no
##
## END FIXME

## FIXME: this should be a separate macro
##
# Report the final consequences.
AC_MSG_CHECKING([if libtool supports shared libraries])
AC_MSG_RESULT([$can_build_shared])
##
## END FIXME

## FIXME: this should be a separate macro
##
AC_MSG_CHECKING([whether to build shared libraries])
test "$can_build_shared" = "no" && enable_shared=no

# On AIX, shared libraries and static libraries use the same namespace, and
# are all built from PIC.
case "$host_os" in
aix3*)
  test "$enable_shared" = yes && enable_static=no
  if test -n "$RANLIB"; then
    archive_cmds="$archive_cmds~\$RANLIB \$lib"
    postinstall_cmds='$RANLIB $lib'
  fi
  ;;

aix4*)
  if test "$host_cpu" != ia64 && test "$aix_use_runtimelinking" = no ; then
    test "$enable_shared" = yes && enable_static=no
  fi
  ;;
esac
AC_MSG_RESULT([$enable_shared])
##
## END FIXME

## FIXME: this should be a separate macro
##
AC_MSG_CHECKING([whether to build static libraries])
# Make sure either enable_shared or enable_static is yes.
test "$enable_shared" = yes || enable_static=yes
AC_MSG_RESULT([$enable_static])
##
## END FIXME

if test "$hardcode_action" = relink; then
  # Fast installation is not supported
  enable_fast_install=no
elif test "$shlibpath_overrides_runpath" = yes ||
     test "$enable_shared" = no; then
  # Fast installation is not necessary
  enable_fast_install=needless
fi

variables_saved_for_relink="PATH $shlibpath_var $runpath_var"
if test "$GCC" = yes; then
  variables_saved_for_relink="$variables_saved_for_relink GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH"
fi

AC_LIBTOOL_DLOPEN_SELF

## FIXME: this should be a separate macro
##
if test "$enable_shared" = yes && test "$GCC" = yes; then
  case $archive_cmds in
  *'~'*)
    # FIXME: we may have to deal with multi-command sequences.
    ;;
  '$CC '*)
    # Test whether the compiler implicitly links with -lc since on some
    # systems, -lgcc has to come before -lc. If gcc already passes -lc
    # to ld, don't add -lc before -lgcc.
    AC_MSG_CHECKING([whether -lc should be explicitly linked in])
    AC_CACHE_VAL([lt_cv_archive_cmds_need_lc],
    [$rm conftest*
    echo 'static int dummy;' > conftest.$ac_ext

    if AC_TRY_EVAL(ac_compile); then
      soname=conftest
      lib=conftest
      libobjs=conftest.$ac_objext
      deplibs=
      wl=$lt_cv_prog_cc_wl
      compiler_flags=-v
      linker_flags=-v
      verstring=
      output_objdir=.
      libname=conftest
      save_allow_undefined_flag=$allow_undefined_flag
      allow_undefined_flag=
      if AC_TRY_EVAL(archive_cmds 2\>\&1 \| grep \" -lc \" \>/dev/null 2\>\&1)
      then
	lt_cv_archive_cmds_need_lc=no
      else
	lt_cv_archive_cmds_need_lc=yes
      fi
      allow_undefined_flag=$save_allow_undefined_flag
    else
      cat conftest.err 1>&5
    fi])
    AC_MSG_RESULT([$lt_cv_archive_cmds_need_lc])
    ;;
  esac
fi
need_lc=${lt_cv_archive_cmds_need_lc-yes}
##
## END FIXME

## FIXME: this should be a separate macro
##
# The second clause should only fire when bootstrapping the
# libtool distribution, otherwise you forgot to ship ltmain.sh
# with your package, and you will get complaints that there are
# no rules to generate ltmain.sh.
if test -f "$ltmain"; then
  :
else
  # If there is no Makefile yet, we rely on a make rule to execute
  # `config.status --recheck' to rerun these tests and create the
  # libtool script then.
  test -f Makefile && make "$ltmain"
fi

if test -f "$ltmain"; then
  trap "$rm \"${ofile}T\"; exit 1" 1 2 15
  $rm -f "${ofile}T"

  echo creating $ofile

  # Now quote all the things that may contain metacharacters while being
  # careful not to overquote the AC_SUBSTed values.  We take copies of the
  # variables and quote the copies for generation of the libtool script.
  for var in echo old_CC old_CFLAGS SED \
    AR AR_FLAGS CC LD LN_S NM SHELL \
    reload_flag reload_cmds wl \
    pic_flag link_static_flag no_builtin_flag export_dynamic_flag_spec \
    thread_safe_flag_spec whole_archive_flag_spec libname_spec \
    library_names_spec soname_spec \
    RANLIB old_archive_cmds old_archive_from_new_cmds old_postinstall_cmds \
    old_postuninstall_cmds archive_cmds archive_expsym_cmds postinstall_cmds \
    postuninstall_cmds extract_expsyms_cmds old_archive_from_expsyms_cmds \
    old_striplib striplib file_magic_cmd export_symbols_cmds \
    deplibs_check_method allow_undefined_flag no_undefined_flag \
    finish_cmds finish_eval global_symbol_pipe global_symbol_to_cdecl \
    global_symbol_to_c_name_address \
    hardcode_libdir_flag_spec hardcode_libdir_separator  \
    sys_lib_search_path_spec sys_lib_dlsearch_path_spec \
    compiler_c_o compiler_o_lo need_locks exclude_expsyms include_expsyms; do

    case $var in
    reload_cmds | old_archive_cmds | old_archive_from_new_cmds | \
    old_postinstall_cmds | old_postuninstall_cmds | \
    export_symbols_cmds | archive_cmds | archive_expsym_cmds | \
    extract_expsyms_cmds | old_archive_from_expsyms_cmds | \
    postinstall_cmds | postuninstall_cmds | \
    finish_cmds | sys_lib_search_path_spec | sys_lib_dlsearch_path_spec)
      # Double-quote double-evaled strings.
      eval "lt_$var=\\\"\`\$echo \"X\$$var\" | \$Xsed -e \"\$double_quote_subst\" -e \"\$sed_quote_subst\" -e \"\$delay_variable_subst\"\`\\\""
      ;;
    *)
      eval "lt_$var=\\\"\`\$echo \"X\$$var\" | \$Xsed -e \"\$sed_quote_subst\"\`\\\""
      ;;
    esac
  done

  cat <<__EOF__ > "${ofile}T"
#! $SHELL

# `$echo "$ofile" | sed 's%^.*/%%'` - Provide generalized library-building support services.
# Generated automatically by $PROGRAM (GNU $PACKAGE $VERSION$TIMESTAMP)
# NOTE: Changes made to this file will be lost: look at ltmain.sh.
#
# Copyright (C) 1996-2000 Free Software Foundation, Inc.
# Originally by Gordon Matzigkeit <gord@gnu.ai.mit.edu>, 1996
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# As a special exception to the GNU General Public License, if you
# distribute this file as part of a program that contains a
# configuration script generated by Autoconf, you may include it under
# the same distribution terms that you use for the rest of that program.

# A sed that does not truncate output.
SED=$lt_SED

# Sed that helps us avoid accidentally triggering echo(1) options like -n.
Xsed="${SED} -e s/^X//"

# The HP-UX ksh and POSIX shell print the target directory to stdout
# if CDPATH is set.
if test "X\${CDPATH+set}" = Xset; then CDPATH=:; export CDPATH; fi

# ### BEGIN LIBTOOL CONFIG

# Libtool was configured on host `(hostname || uname -n) 2>/dev/null | sed 1q`:

# Shell to use when invoking shell scripts.
SHELL=$lt_SHELL

# Whether or not to build shared libraries.
build_libtool_libs=$enable_shared

# Whether or not to build static libraries.
build_old_libs=$enable_static

# Whether or not to add -lc for building shared libraries.
build_libtool_need_lc=$need_lc

# Whether or not to optimize for fast installation.
fast_install=$enable_fast_install

# The host system.
host_alias=$host_alias
host=$host

# An echo program that does not interpret backslashes.
echo=$lt_echo

# The archiver.
AR=$lt_AR
AR_FLAGS=$lt_AR_FLAGS

# The default C compiler.
CC=$lt_CC

# Is the compiler the GNU C compiler?
with_gcc=$GCC

# The linker used to build libraries.
LD=$lt_LD

# Whether we need hard or soft links.
LN_S=$lt_LN_S

# A BSD-compatible nm program.
NM=$lt_NM

# A symbol stripping program
STRIP=$STRIP

# Used to examine libraries when file_magic_cmd begins "file"
MAGIC_CMD=$MAGIC_CMD

# Used on cygwin: DLL creation program.
DLLTOOL="$DLLTOOL"

# Used on cygwin: object dumper.
OBJDUMP="$OBJDUMP"

# Used on cygwin: assembler.
AS="$AS"

# The name of the directory that contains temporary libtool files.
objdir=$objdir

# How to create reloadable object files.
reload_flag=$lt_reload_flag
reload_cmds=$lt_reload_cmds

# How to pass a linker flag through the compiler.
wl=$lt_wl

# Object file suffix (normally "o").
objext="$ac_objext"

# Old archive suffix (normally "a").
libext="$libext"

# Executable file suffix (normally "").
exeext="$exeext"

# Additional compiler flags for building library objects.
pic_flag=$lt_pic_flag
pic_mode=$pic_mode

# Does compiler simultaneously support -c and -o options?
compiler_c_o=$lt_compiler_c_o

# Can we write directly to a .lo ?
compiler_o_lo=$lt_compiler_o_lo

# Must we lock files when doing compilation ?
need_locks=$lt_need_locks

# Do we need the lib prefix for modules?
need_lib_prefix=$need_lib_prefix

# Do we need a version for libraries?
need_version=$need_version

# Whether dlopen is supported.
dlopen_support=$enable_dlopen

# Whether dlopen of programs is supported.
dlopen_self=$enable_dlopen_self

# Whether dlopen of statically linked programs is supported.
dlopen_self_static=$enable_dlopen_self_static

# Compiler flag to prevent dynamic linking.
link_static_flag=$lt_link_static_flag

# Compiler flag to turn off builtin functions.
no_builtin_flag=$lt_no_builtin_flag

# Compiler flag to allow reflexive dlopens.
export_dynamic_flag_spec=$lt_export_dynamic_flag_spec

# Compiler flag to generate shared objects directly from archives.
whole_archive_flag_spec=$lt_whole_archive_flag_spec

# Compiler flag to generate thread-safe objects.
thread_safe_flag_spec=$lt_thread_safe_flag_spec

# Library versioning type.
version_type=$version_type

# Format of library name prefix.
libname_spec=$lt_libname_spec

# List of archive names.  First name is the real one, the rest are links.
# The last name is the one that the linker finds with -lNAME.
library_names_spec=$lt_library_names_spec

# The coded name of the library, if different from the real name.
soname_spec=$lt_soname_spec

# Commands used to build and install an old-style archive.
RANLIB=$lt_RANLIB
old_archive_cmds=$lt_old_archive_cmds
old_postinstall_cmds=$lt_old_postinstall_cmds
old_postuninstall_cmds=$lt_old_postuninstall_cmds

# Create an old-style archive from a shared archive.
old_archive_from_new_cmds=$lt_old_archive_from_new_cmds

# Create a temporary old-style archive to link instead of a shared archive.
old_archive_from_expsyms_cmds=$lt_old_archive_from_expsyms_cmds

# Commands used to build and install a shared archive.
archive_cmds=$lt_archive_cmds
archive_expsym_cmds=$lt_archive_expsym_cmds
postinstall_cmds=$lt_postinstall_cmds
postuninstall_cmds=$lt_postuninstall_cmds

# Commands to strip libraries.
old_striplib=$lt_old_striplib
striplib=$lt_striplib

# Method to check whether dependent libraries are shared objects.
deplibs_check_method=$lt_deplibs_check_method

# Command to use when deplibs_check_method == file_magic.
file_magic_cmd=$lt_file_magic_cmd

# Flag that allows shared libraries with undefined symbols to be built.
allow_undefined_flag=$lt_allow_undefined_flag

# Flag that forces no undefined symbols.
no_undefined_flag=$lt_no_undefined_flag

# Commands used to finish a libtool library installation in a directory.
finish_cmds=$lt_finish_cmds

# Same as above, but a single script fragment to be evaled but not shown.
finish_eval=$lt_finish_eval

# Take the output of nm and produce a listing of raw symbols and C names.
global_symbol_pipe=$lt_global_symbol_pipe

# Transform the output of nm in a proper C declaration
global_symbol_to_cdecl=$lt_global_symbol_to_cdecl

# Transform the output of nm in a C name address pair
global_symbol_to_c_name_address=$lt_global_symbol_to_c_name_address

# This is the shared library runtime path variable.
runpath_var=$runpath_var

# This is the shared library path variable.
shlibpath_var=$shlibpath_var

# Is shlibpath searched before the hard-coded library search path?
shlibpath_overrides_runpath=$shlibpath_overrides_runpath

# How to hardcode a shared library path into an executable.
hardcode_action=$hardcode_action

# Whether we should hardcode library paths into libraries.
hardcode_into_libs=$hardcode_into_libs

# Flag to hardcode \$libdir into a binary during linking.
# This must work even if \$libdir does not exist.
hardcode_libdir_flag_spec=$lt_hardcode_libdir_flag_spec

# Whether we need a single -rpath flag with a separated argument.
hardcode_libdir_separator=$lt_hardcode_libdir_separator

# Set to yes if using DIR/libNAME.so during linking hardcodes DIR into the
# resulting binary.
hardcode_direct=$hardcode_direct

# Set to yes if using the -LDIR flag during linking hardcodes DIR into the
# resulting binary.
hardcode_minus_L=$hardcode_minus_L

# Set to yes if using SHLIBPATH_VAR=DIR during linking hardcodes DIR into
# the resulting binary.
hardcode_shlibpath_var=$hardcode_shlibpath_var

# Variables whose values should be saved in libtool wrapper scripts and
# restored at relink time.
variables_saved_for_relink="$variables_saved_for_relink"

# Whether libtool must link a program against all its dependency libraries.
link_all_deplibs=$link_all_deplibs

# Compile-time system search path for libraries
sys_lib_search_path_spec=$lt_sys_lib_search_path_spec

# Run-time system search path for libraries
sys_lib_dlsearch_path_spec=$lt_sys_lib_dlsearch_path_spec

# Fix the shell variable \$srcfile for the compiler.
fix_srcfile_path="$fix_srcfile_path"

# Set to yes if exported symbols are required.
always_export_symbols=$always_export_symbols

# The commands to list exported symbols.
export_symbols_cmds=$lt_export_symbols_cmds

# The commands to extract the exported symbol list from a shared archive.
extract_expsyms_cmds=$lt_extract_expsyms_cmds

# Symbols that should not be listed in the preloaded symbols.
exclude_expsyms=$lt_exclude_expsyms

# Symbols that must always be exported.
include_expsyms=$lt_include_expsyms

# ### END LIBTOOL CONFIG

__EOF__

  case $host_os in
  aix3*)
    cat <<\EOF >> "${ofile}T"

# AIX sometimes has problems with the GCC collect2 program.  For some
# reason, if we set the COLLECT_NAMES environment variable, the problems
# vanish in a puff of smoke.
if test "X${COLLECT_NAMES+set}" != Xset; then
  COLLECT_NAMES=
  export COLLECT_NAMES
fi
EOF
    ;;
  esac

  case $host_os in
  cygwin* | mingw* | pw32* | os2*)
    cat <<'EOF' >> "${ofile}T"
      # This is a source program that is used to create dlls on Windows
      # Don't remove nor modify the starting and closing comments
# /* ltdll.c starts here */
# #define WIN32_LEAN_AND_MEAN
# #include <windows.h>
# #undef WIN32_LEAN_AND_MEAN
# #include <stdio.h>
#
# #ifndef __CYGWIN__
# #  ifdef __CYGWIN32__
# #    define __CYGWIN__ __CYGWIN32__
# #  endif
# #endif
#
# #ifdef __cplusplus
# extern "C" {
# #endif
# BOOL APIENTRY DllMain (HINSTANCE hInst, DWORD reason, LPVOID reserved);
# #ifdef __cplusplus
# }
# #endif
#
# #ifdef __CYGWIN__
# #include <cygwin/cygwin_dll.h>
# DECLARE_CYGWIN_DLL( DllMain );
# #endif
# HINSTANCE __hDllInstance_base;
#
# BOOL APIENTRY
# DllMain (HINSTANCE hInst, DWORD reason, LPVOID reserved)
# {
#   __hDllInstance_base = hInst;
#   return TRUE;
# }
# /* ltdll.c ends here */
	# This is a source program that is used to create import libraries
	# on Windows for dlls which lack them. Don't remove nor modify the
	# starting and closing comments
# /* impgen.c starts here */
# /*   Copyright (C) 1999-2000 Free Software Foundation, Inc.
#
#  This file is part of GNU libtool.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#  */
#
# #include <stdio.h>		/* for printf() */
# #include <unistd.h>		/* for open(), lseek(), read() */
# #include <fcntl.h>		/* for O_RDONLY, O_BINARY */
# #include <string.h>		/* for strdup() */
#
# /* O_BINARY isn't required (or even defined sometimes) under Unix */
# #ifndef O_BINARY
# #define O_BINARY 0
# #endif
#
# static unsigned int
# pe_get16 (fd, offset)
#      int fd;
#      int offset;
# {
#   unsigned char b[2];
#   lseek (fd, offset, SEEK_SET);
#   read (fd, b, 2);
#   return b[0] + (b[1]<<8);
# }
#
# static unsigned int
# pe_get32 (fd, offset)
#     int fd;
#     int offset;
# {
#   unsigned char b[4];
#   lseek (fd, offset, SEEK_SET);
#   read (fd, b, 4);
#   return b[0] + (b[1]<<8) + (b[2]<<16) + (b[3]<<24);
# }
#
# static unsigned int
# pe_as32 (ptr)
#      void *ptr;
# {
#   unsigned char *b = ptr;
#   return b[0] + (b[1]<<8) + (b[2]<<16) + (b[3]<<24);
# }
#
# int
# main (argc, argv)
#     int argc;
#     char *argv[];
# {
#     int dll;
#     unsigned long pe_header_offset, opthdr_ofs, num_entries, i;
#     unsigned long export_rva, export_size, nsections, secptr, expptr;
#     unsigned long name_rvas, nexp;
#     unsigned char *expdata, *erva;
#     char *filename, *dll_name;
#
#     filename = argv[1];
#
#     dll = open(filename, O_RDONLY|O_BINARY);
#     if (dll < 1)
# 	return 1;
#
#     dll_name = filename;
#
#     for (i=0; filename[i]; i++)
# 	if (filename[i] == '/' || filename[i] == '\\'  || filename[i] == ':')
# 	    dll_name = filename + i +1;
#
#     pe_header_offset = pe_get32 (dll, 0x3c);
#     opthdr_ofs = pe_header_offset + 4 + 20;
#     num_entries = pe_get32 (dll, opthdr_ofs + 92);
#
#     if (num_entries < 1) /* no exports */
# 	return 1;
#
#     export_rva = pe_get32 (dll, opthdr_ofs + 96);
#     export_size = pe_get32 (dll, opthdr_ofs + 100);
#     nsections = pe_get16 (dll, pe_header_offset + 4 +2);
#     secptr = (pe_header_offset + 4 + 20 +
# 	      pe_get16 (dll, pe_header_offset + 4 + 16));
#
#     expptr = 0;
#     for (i = 0; i < nsections; i++)
#     {
# 	char sname[8];
# 	unsigned long secptr1 = secptr + 40 * i;
# 	unsigned long vaddr = pe_get32 (dll, secptr1 + 12);
# 	unsigned long vsize = pe_get32 (dll, secptr1 + 16);
# 	unsigned long fptr = pe_get32 (dll, secptr1 + 20);
# 	lseek(dll, secptr1, SEEK_SET);
# 	read(dll, sname, 8);
# 	if (vaddr <= export_rva && vaddr+vsize > export_rva)
# 	{
# 	    expptr = fptr + (export_rva - vaddr);
# 	    if (export_rva + export_size > vaddr + vsize)
# 		export_size = vsize - (export_rva - vaddr);
# 	    break;
# 	}
#     }
#
#     expdata = (unsigned char*)malloc(export_size);
#     lseek (dll, expptr, SEEK_SET);
#     read (dll, expdata, export_size);
#     erva = expdata - export_rva;
#
#     nexp = pe_as32 (expdata+24);
#     name_rvas = pe_as32 (expdata+32);
#
#     printf ("EXPORTS\n");
#     for (i = 0; i<nexp; i++)
#     {
# 	unsigned long name_rva = pe_as32 (erva+name_rvas+i*4);
# 	printf ("\t%s @ %ld ;\n", erva+name_rva, 1+ i);
#     }
#
#     return 0;
# }
# /* impgen.c ends here */

EOF
    ;;
  esac

  # We use sed instead of cat because bash on DJGPP gets confused if
  # if finds mixed CR/LF and LF-only lines.  Since sed operates in
  # text mode, it properly converts lines to CR/LF.  This bash problem
  # is reportedly fixed, but why not run on old versions too?
  sed '$q' "$ltmain" >> "${ofile}T" || (rm -f "${ofile}T"; exit 1)

  mv -f "${ofile}T" "$ofile" || \
    (rm -f "$ofile" && cp "${ofile}T" "$ofile" && rm -f "${ofile}T")
  chmod +x "$ofile"
fi
##
## END FIXME

])# _LT_AC_LTCONFIG_HACK

# AC_LIBTOOL_DLOPEN - enable checks for dlopen support
AC_DEFUN([AC_LIBTOOL_DLOPEN], [AC_BEFORE([$0],[AC_LIBTOOL_SETUP])])

# AC_LIBTOOL_WIN32_DLL - declare package support for building win32 dll's
AC_DEFUN([AC_LIBTOOL_WIN32_DLL], [AC_BEFORE([$0], [AC_LIBTOOL_SETUP])])

# AC_ENABLE_SHARED - implement the --enable-shared flag
# Usage: AC_ENABLE_SHARED[(DEFAULT)]
#   Where DEFAULT is either `yes' or `no'.  If omitted, it defaults to
#   `yes'.
AC_DEFUN([AC_ENABLE_SHARED],
[define([AC_ENABLE_SHARED_DEFAULT], ifelse($1, no, no, yes))dnl
AC_ARG_ENABLE(shared,
changequote(<<, >>)dnl
<<  --enable-shared[=PKGS]  build shared libraries [default=>>AC_ENABLE_SHARED_DEFAULT],
changequote([, ])dnl
[p=${PACKAGE-default}
case $enableval in
yes) enable_shared=yes ;;
no) enable_shared=no ;;
*)
  enable_shared=no
  # Look at the argument we got.  We use all the common list separators.
  IFS="${IFS= 	}"; ac_save_ifs="$IFS"; IFS="${IFS}:,"
  for pkg in $enableval; do
    if test "X$pkg" = "X$p"; then
      enable_shared=yes
    fi
  done
  IFS="$ac_save_ifs"
  ;;
esac],
enable_shared=AC_ENABLE_SHARED_DEFAULT)dnl
])

# AC_DISABLE_SHARED - set the default shared flag to --disable-shared
AC_DEFUN([AC_DISABLE_SHARED],
[AC_BEFORE([$0],[AC_LIBTOOL_SETUP])dnl
AC_ENABLE_SHARED(no)])

# AC_ENABLE_STATIC - implement the --enable-static flag
# Usage: AC_ENABLE_STATIC[(DEFAULT)]
#   Where DEFAULT is either `yes' or `no'.  If omitted, it defaults to
#   `yes'.
AC_DEFUN([AC_ENABLE_STATIC],
[define([AC_ENABLE_STATIC_DEFAULT], ifelse($1, no, no, yes))dnl
AC_ARG_ENABLE(static,
changequote(<<, >>)dnl
<<  --enable-static[=PKGS]  build static libraries [default=>>AC_ENABLE_STATIC_DEFAULT],
changequote([, ])dnl
[p=${PACKAGE-default}
case $enableval in
yes) enable_static=yes ;;
no) enable_static=no ;;
*)
  enable_static=no
  # Look at the argument we got.  We use all the common list separators.
  IFS="${IFS= 	}"; ac_save_ifs="$IFS"; IFS="${IFS}:,"
  for pkg in $enableval; do
    if test "X$pkg" = "X$p"; then
      enable_static=yes
    fi
  done
  IFS="$ac_save_ifs"
  ;;
esac],
enable_static=AC_ENABLE_STATIC_DEFAULT)dnl
])

# AC_DISABLE_STATIC - set the default static flag to --disable-static
AC_DEFUN([AC_DISABLE_STATIC],
[AC_BEFORE([$0],[AC_LIBTOOL_SETUP])dnl
AC_ENABLE_STATIC(no)])


# AC_ENABLE_FAST_INSTALL - implement the --enable-fast-install flag
# Usage: AC_ENABLE_FAST_INSTALL[(DEFAULT)]
#   Where DEFAULT is either `yes' or `no'.  If omitted, it defaults to
#   `yes'.
AC_DEFUN([AC_ENABLE_FAST_INSTALL],
[define([AC_ENABLE_FAST_INSTALL_DEFAULT], ifelse($1, no, no, yes))dnl
AC_ARG_ENABLE(fast-install,
changequote(<<, >>)dnl
<<  --enable-fast-install[=PKGS]  optimize for fast installation [default=>>AC_ENABLE_FAST_INSTALL_DEFAULT],
changequote([, ])dnl
[p=${PACKAGE-default}
case $enableval in
yes) enable_fast_install=yes ;;
no) enable_fast_install=no ;;
*)
  enable_fast_install=no
  # Look at the argument we got.  We use all the common list separators.
  IFS="${IFS= 	}"; ac_save_ifs="$IFS"; IFS="${IFS}:,"
  for pkg in $enableval; do
    if test "X$pkg" = "X$p"; then
      enable_fast_install=yes
    fi
  done
  IFS="$ac_save_ifs"
  ;;
esac],
enable_fast_install=AC_ENABLE_FAST_INSTALL_DEFAULT)dnl
])

# AC_DISABLE_FAST_INSTALL - set the default to --disable-fast-install
AC_DEFUN([AC_DISABLE_FAST_INSTALL],
[AC_BEFORE([$0],[AC_LIBTOOL_SETUP])dnl
AC_ENABLE_FAST_INSTALL(no)])

# AC_LIBTOOL_PICMODE - implement the --with-pic flag
# Usage: AC_LIBTOOL_PICMODE[(MODE)]
#   Where MODE is either `yes' or `no'.  If omitted, it defaults to
#   `both'.
AC_DEFUN([AC_LIBTOOL_PICMODE],
[AC_BEFORE([$0],[AC_LIBTOOL_SETUP])dnl
pic_mode=ifelse($#,1,$1,default)])


# AC_PATH_TOOL_PREFIX - find a file program which can recognise shared library
AC_DEFUN([AC_PATH_TOOL_PREFIX],
[AC_MSG_CHECKING([for $1])
AC_CACHE_VAL(lt_cv_path_MAGIC_CMD,
[case $MAGIC_CMD in
  /*)
  lt_cv_path_MAGIC_CMD="$MAGIC_CMD" # Let the user override the test with a path.
  ;;
  ?:/*)
  lt_cv_path_MAGIC_CMD="$MAGIC_CMD" # Let the user override the test with a dos path.
  ;;
  *)
  ac_save_MAGIC_CMD="$MAGIC_CMD"
  IFS="${IFS=   }"; ac_save_ifs="$IFS"; IFS=":"
dnl $ac_dummy forces splitting on constant user-supplied paths.
dnl POSIX.2 word splitting is done only on the output of word expansions,
dnl not every word.  This closes a longstanding sh security hole.
  ac_dummy="ifelse([$2], , $PATH, [$2])"
  for ac_dir in $ac_dummy; do
    test -z "$ac_dir" && ac_dir=.
    if test -f $ac_dir/$1; then
      lt_cv_path_MAGIC_CMD="$ac_dir/$1"
      if test -n "$file_magic_test_file"; then
	case $deplibs_check_method in
	"file_magic "*)
	  file_magic_regex="`expr \"$deplibs_check_method\" : \"file_magic \(.*\)\"`"
	  MAGIC_CMD="$lt_cv_path_MAGIC_CMD"
	  if eval $file_magic_cmd \$file_magic_test_file 2> /dev/null |
	    egrep "$file_magic_regex" > /dev/null; then
	    :
	  else
	    cat <<EOF 1>&2

*** Warning: the command libtool uses to detect shared libraries,
*** $file_magic_cmd, produces output that libtool cannot recognize.
*** The result is that libtool may fail to recognize shared libraries
*** as such.  This will affect the creation of libtool libraries that
*** depend on shared libraries, but programs linked with such libtool
*** libraries will work regardless of this problem.  Nevertheless, you
*** may want to report the problem to your system manager and/or to
*** bug-libtool@gnu.org

EOF
	  fi ;;
	esac
      fi
      break
    fi
  done
  IFS="$ac_save_ifs"
  MAGIC_CMD="$ac_save_MAGIC_CMD"
  ;;
esac])
MAGIC_CMD="$lt_cv_path_MAGIC_CMD"
if test -n "$MAGIC_CMD"; then
  AC_MSG_RESULT($MAGIC_CMD)
else
  AC_MSG_RESULT(no)
fi
])


# AC_PATH_MAGIC - find a file program which can recognise a shared library
AC_DEFUN([AC_PATH_MAGIC],
[AC_REQUIRE([AC_CHECK_TOOL_PREFIX])dnl
AC_PATH_TOOL_PREFIX(${ac_tool_prefix}file, /usr/bin:$PATH)
if test -z "$lt_cv_path_MAGIC_CMD"; then
  if test -n "$ac_tool_prefix"; then
    AC_PATH_TOOL_PREFIX(file, /usr/bin:$PATH)
  else
    MAGIC_CMD=:
  fi
fi
])


# AC_PROG_LD - find the path to the GNU or non-GNU linker
AC_DEFUN([AC_PROG_LD],
[AC_ARG_WITH(gnu-ld,
[  --with-gnu-ld           assume the C compiler uses GNU ld [default=no]],
test "$withval" = no || with_gnu_ld=yes, with_gnu_ld=no)
AC_REQUIRE([AC_PROG_CC])dnl
AC_REQUIRE([AC_CANONICAL_HOST])dnl
AC_REQUIRE([AC_CANONICAL_BUILD])dnl
AC_REQUIRE([_LT_AC_LIBTOOL_SYS_PATH_SEPARATOR])dnl
ac_prog=ld
if test "$GCC" = yes; then
  # Check if gcc -print-prog-name=ld gives a path.
  AC_MSG_CHECKING([for ld used by GCC])
  case $host in
  *-*-mingw*)
    # gcc leaves a trailing carriage return which upsets mingw
    ac_prog=`($CC -print-prog-name=ld) 2>&5 | tr -d '\015'` ;;
  *)
    ac_prog=`($CC -print-prog-name=ld) 2>&5` ;;
  esac
  case $ac_prog in
    # Accept absolute paths.
    [[\\/]]* | [[A-Za-z]]:[[\\/]]*)
      re_direlt='/[[^/]][[^/]]*/\.\./'
      # Canonicalize the path of ld
      ac_prog=`echo $ac_prog| sed 's%\\\\%/%g'`
      while echo $ac_prog | grep "$re_direlt" > /dev/null 2>&1; do
	ac_prog=`echo $ac_prog| sed "s%$re_direlt%/%"`
      done
      test -z "$LD" && LD="$ac_prog"
      ;;
  "")
    # If it fails, then pretend we aren't using GCC.
    ac_prog=ld
    ;;
  *)
    # If it is relative, then search for the first ld in PATH.
    with_gnu_ld=unknown
    ;;
  esac
elif test "$with_gnu_ld" = yes; then
  AC_MSG_CHECKING([for GNU ld])
else
  AC_MSG_CHECKING([for non-GNU ld])
fi
AC_CACHE_VAL(lt_cv_path_LD,
[if test -z "$LD"; then
  IFS="${IFS= 	}"; ac_save_ifs="$IFS"; IFS=$PATH_SEPARATOR
  for ac_dir in $PATH; do
    test -z "$ac_dir" && ac_dir=.
    if test -f "$ac_dir/$ac_prog" || test -f "$ac_dir/$ac_prog$ac_exeext"; then
      lt_cv_path_LD="$ac_dir/$ac_prog"
      # Check to see if the program is GNU ld.  I'd rather use --version,
      # but apparently some GNU ld's only accept -v.
      # Break only if it was the GNU/non-GNU ld that we prefer.
      if "$lt_cv_path_LD" -v 2>&1 < /dev/null | egrep '(GNU|with BFD)' > /dev/null; then
	test "$with_gnu_ld" != no && break
      else
	test "$with_gnu_ld" != yes && break
      fi
    fi
  done
  IFS="$ac_save_ifs"
else
  lt_cv_path_LD="$LD" # Let the user override the test with a path.
fi])
LD="$lt_cv_path_LD"
if test -n "$LD"; then
  AC_MSG_RESULT($LD)
else
  AC_MSG_RESULT(no)
fi
test -z "$LD" && AC_MSG_ERROR([no acceptable ld found in \$PATH])
AC_PROG_LD_GNU
])

# AC_PROG_LD_GNU -
AC_DEFUN([AC_PROG_LD_GNU],
[AC_CACHE_CHECK([if the linker ($LD) is GNU ld], lt_cv_prog_gnu_ld,
[# I'd rather use --version here, but apparently some GNU ld's only accept -v.
if $LD -v 2>&1 </dev/null | egrep '(GNU|with BFD)' 1>&5; then
  lt_cv_prog_gnu_ld=yes
else
  lt_cv_prog_gnu_ld=no
fi])
with_gnu_ld=$lt_cv_prog_gnu_ld
])

# AC_PROG_LD_RELOAD_FLAG - find reload flag for linker
#   -- PORTME Some linkers may need a different reload flag.
AC_DEFUN([AC_PROG_LD_RELOAD_FLAG],
[AC_CACHE_CHECK([for $LD option to reload object files], lt_cv_ld_reload_flag,
[lt_cv_ld_reload_flag='-r'])
reload_flag=$lt_cv_ld_reload_flag
test -n "$reload_flag" && reload_flag=" $reload_flag"
])

# AC_DEPLIBS_CHECK_METHOD - how to check for library dependencies
#  -- PORTME fill in with the dynamic library characteristics
AC_DEFUN([AC_DEPLIBS_CHECK_METHOD],
[AC_CACHE_CHECK([how to recognise dependent libraries],
lt_cv_deplibs_check_method,
[lt_cv_file_magic_cmd='$MAGIC_CMD'
lt_cv_file_magic_test_file=
lt_cv_deplibs_check_method='unknown'
# Need to set the preceding variable on all platforms that support
# interlibrary dependencies.
# 'none' -- dependencies not supported.
# `unknown' -- same as none, but documents that we really don't know.
# 'pass_all' -- all dependencies passed with no checks.
# 'test_compile' -- check by making test program.
# 'file_magic [[regex]]' -- check by looking for files in library path
# which responds to the $file_magic_cmd with a given egrep regex.
# If you have `file' or equivalent on your system and you're not sure
# whether `pass_all' will *always* work, you probably want this one.

case $host_os in
aix4* | aix5*)
  lt_cv_deplibs_check_method=pass_all
  ;;

beos*)
  lt_cv_deplibs_check_method=pass_all
  ;;

bsdi4*)
  lt_cv_deplibs_check_method='file_magic ELF [[0-9]][[0-9]]*-bit [[ML]]SB (shared object|dynamic lib)'
  lt_cv_file_magic_cmd='/usr/bin/file -L'
  lt_cv_file_magic_test_file=/shlib/libc.so
  ;;

cygwin* | mingw* | pw32*)
  lt_cv_deplibs_check_method='file_magic file format pei*-i386(.*architecture: i386)?'
  lt_cv_file_magic_cmd='$OBJDUMP -f'
  ;;

darwin* | rhapsody*)
  lt_cv_deplibs_check_method='file_magic Mach-O dynamically linked shared library'
  lt_cv_file_magic_cmd='/usr/bin/file -L'
  case "$host_os" in
  rhapsody* | darwin1.[[012]])
    lt_cv_file_magic_test_file=`echo /System/Library/Frameworks/System.framework/Versions/*/System | head -1`
    ;;
  *) # Darwin 1.3 on
    lt_cv_file_magic_test_file='/usr/lib/libSystem.dylib'
    ;;
  esac
  ;;

freebsd*)
  if echo __ELF__ | $CC -E - | grep __ELF__ > /dev/null; then
    case $host_cpu in
    i*86 )
      # Not sure whether the presence of OpenBSD here was a mistake.
      # Let's accept both of them until this is cleared up.
      lt_cv_deplibs_check_method='file_magic (FreeBSD|OpenBSD)/i[[3-9]]86 (compact )?demand paged shared library'
      lt_cv_file_magic_cmd=/usr/bin/file
      lt_cv_file_magic_test_file=`echo /usr/lib/libc.so.*`
      ;;
    esac
  else
    lt_cv_deplibs_check_method=pass_all
  fi
  ;;

gnu*)
  lt_cv_deplibs_check_method=pass_all
  ;;

hpux10.20*|hpux11*)
  lt_cv_deplibs_check_method='file_magic (s[[0-9]][[0-9]][[0-9]]|PA-RISC[[0-9]].[[0-9]]) shared library'
  lt_cv_file_magic_cmd=/usr/bin/file
  lt_cv_file_magic_test_file=/usr/lib/libc.sl
  ;;

irix5* | irix6* | nonstopux*)
  case $host_os in
  irix5* | nonstopux*)
    # this will be overridden with pass_all, but let us keep it just in case
    lt_cv_deplibs_check_method="file_magic ELF 32-bit MSB dynamic lib MIPS - version 1"
    ;;
  *)
    case $LD in
    *-32|*"-32 ") libmagic=32-bit;;
    *-n32|*"-n32 ") libmagic=N32;;
    *-64|*"-64 ") libmagic=64-bit;;
    *) libmagic=never-match;;
    esac
    # this will be overridden with pass_all, but let us keep it just in case
    lt_cv_deplibs_check_method="file_magic ELF ${libmagic} MSB mips-[[1234]] dynamic lib MIPS - version 1"
    ;;
  esac
  lt_cv_file_magic_test_file=`echo /lib${libsuff}/libc.so*`
  lt_cv_deplibs_check_method=pass_all
  ;;

# This must be Linux ELF.
linux-gnu*)
  case $host_cpu in
  alpha* | hppa* | i*86 | mips | mipsel | powerpc* | sparc* | ia64*)
    lt_cv_deplibs_check_method=pass_all ;;
  *)
    # glibc up to 2.1.1 does not perform some relocations on ARM
    lt_cv_deplibs_check_method='file_magic ELF [[0-9]][[0-9]]*-bit [[LM]]SB (shared object|dynamic lib )' ;;
  esac
  lt_cv_file_magic_test_file=`echo /lib/libc.so* /lib/libc-*.so`
  ;;

netbsd*)
  if echo __ELF__ | $CC -E - | grep __ELF__ > /dev/null; then
    lt_cv_deplibs_check_method='match_pattern /lib[[^/\.]]+\.so\.[[0-9]]+\.[[0-9]]+$'
  else
    lt_cv_deplibs_check_method='match_pattern /lib[[^/\.]]+\.so$'
  fi
  ;;

newos6*)
  lt_cv_deplibs_check_method='file_magic ELF [[0-9]][[0-9]]*-bit [[ML]]SB (executable|dynamic lib)'
  lt_cv_file_magic_cmd=/usr/bin/file
  lt_cv_file_magic_test_file=/usr/lib/libnls.so
  ;;

openbsd*)
  lt_cv_file_magic_cmd=/usr/bin/file
  lt_cv_file_magic_test_file=`echo /usr/lib/libc.so.*`
  if test -z "`echo __ELF__ | $CC -E - | grep __ELF__`" || test "$host_os-$host_cpu" = "openbsd2.8-powerpc"; then
    lt_cv_deplibs_check_method='file_magic ELF [[0-9]][[0-9]]*-bit [[LM]]SB shared object'
  else
    lt_cv_deplibs_check_method='file_magic OpenBSD.* shared library'
  fi
  ;;

osf3* | osf4* | osf5*)
  # this will be overridden with pass_all, but let us keep it just in case
  lt_cv_deplibs_check_method='file_magic COFF format alpha shared library'
  lt_cv_file_magic_test_file=/shlib/libc.so
  lt_cv_deplibs_check_method=pass_all
  ;;

sco3.2v5*)
  lt_cv_deplibs_check_method=pass_all
  ;;

solaris*)
  lt_cv_deplibs_check_method=pass_all
  lt_cv_file_magic_test_file=/lib/libc.so
  ;;

sysv5uw[[78]]* | sysv4*uw2*)
  lt_cv_deplibs_check_method=pass_all
  ;;

sysv4 | sysv4.2uw2* | sysv4.3* | sysv5*)
  case $host_vendor in
  motorola)
    lt_cv_deplibs_check_method='file_magic ELF [[0-9]][[0-9]]*-bit [[ML]]SB (shared object|dynamic lib) M[[0-9]][[0-9]]* Version [[0-9]]'
    lt_cv_file_magic_test_file=`echo /usr/lib/libc.so*`
    ;;
  ncr)
    lt_cv_deplibs_check_method=pass_all
    ;;
  sequent)
    lt_cv_file_magic_cmd='/bin/file'
    lt_cv_deplibs_check_method='file_magic ELF [[0-9]][[0-9]]*-bit [[LM]]SB (shared object|dynamic lib )'
    ;;
  sni)
    lt_cv_file_magic_cmd='/bin/file'
    lt_cv_deplibs_check_method="file_magic ELF [[0-9]][[0-9]]*-bit [[LM]]SB dynamic lib"
    lt_cv_file_magic_test_file=/lib/libc.so
    ;;
  siemens)
    lt_cv_deplibs_check_method=pass_all
    ;;
  esac
  ;;
esac
])
file_magic_cmd=$lt_cv_file_magic_cmd
deplibs_check_method=$lt_cv_deplibs_check_method
])


# AC_PROG_NM - find the path to a BSD-compatible name lister
AC_DEFUN([AC_PROG_NM],
[AC_REQUIRE([_LT_AC_LIBTOOL_SYS_PATH_SEPARATOR])dnl
AC_MSG_CHECKING([for BSD-compatible nm])
AC_CACHE_VAL(lt_cv_path_NM,
[if test -n "$NM"; then
  # Let the user override the test.
  lt_cv_path_NM="$NM"
else
  IFS="${IFS= 	}"; ac_save_ifs="$IFS"; IFS=$PATH_SEPARATOR
  for ac_dir in $PATH /usr/ccs/bin /usr/ucb /bin; do
    test -z "$ac_dir" && ac_dir=.
    tmp_nm=$ac_dir/${ac_tool_prefix}nm
    if test -f $tmp_nm || test -f $tmp_nm$ac_exeext ; then
      # Check to see if the nm accepts a BSD-compat flag.
      # Adding the `sed 1q' prevents false positives on HP-UX, which says:
      #   nm: unknown option "B" ignored
      # Tru64's nm complains that /dev/null is an invalid object file
      if ($tmp_nm -B /dev/null 2>&1 | sed '1q'; exit 0) | egrep '(/dev/null|Invalid file or object type)' >/dev/null; then
	lt_cv_path_NM="$tmp_nm -B"
	break
      elif ($tmp_nm -p /dev/null 2>&1 | sed '1q'; exit 0) | egrep /dev/null >/dev/null; then
	lt_cv_path_NM="$tmp_nm -p"
	break
      else
	lt_cv_path_NM=${lt_cv_path_NM="$tmp_nm"} # keep the first match, but
	continue # so that we can try to find one that supports BSD flags
      fi
    fi
  done
  IFS="$ac_save_ifs"
  test -z "$lt_cv_path_NM" && lt_cv_path_NM=nm
fi])
NM="$lt_cv_path_NM"
AC_MSG_RESULT([$NM])
])

# AC_CHECK_LIBM - check for math library
AC_DEFUN([AC_CHECK_LIBM],
[AC_REQUIRE([AC_CANONICAL_HOST])dnl
LIBM=
case $host in
*-*-beos* | *-*-cygwin* | *-*-pw32*)
  # These system don't have libm
  ;;
*-ncr-sysv4.3*)
  AC_CHECK_LIB(mw, _mwvalidcheckl, LIBM="-lmw")
  AC_CHECK_LIB(m, main, LIBM="$LIBM -lm")
  ;;
*)
  AC_CHECK_LIB(m, main, LIBM="-lm")
  ;;
esac
])

# AC_LIBLTDL_CONVENIENCE[(dir)] - sets LIBLTDL to the link flags for
# the libltdl convenience library and LTDLINCL to the include flags for
# the libltdl header and adds --enable-ltdl-convenience to the
# configure arguments.  Note that LIBLTDL and LTDLINCL are not
# AC_SUBSTed, nor is AC_CONFIG_SUBDIRS called.  If DIR is not
# provided, it is assumed to be `libltdl'.  LIBLTDL will be prefixed
# with '${top_builddir}/' and LTDLINCL will be prefixed with
# '${top_srcdir}/' (note the single quotes!).  If your package is not
# flat and you're not using automake, define top_builddir and
# top_srcdir appropriately in the Makefiles.
AC_DEFUN([AC_LIBLTDL_CONVENIENCE],
[AC_BEFORE([$0],[AC_LIBTOOL_SETUP])dnl
  case $enable_ltdl_convenience in
  no) AC_MSG_ERROR([this package needs a convenience libltdl]) ;;
  "") enable_ltdl_convenience=yes
      ac_configure_args="$ac_configure_args --enable-ltdl-convenience" ;;
  esac
  LIBLTDL='${top_builddir}/'ifelse($#,1,[$1],['libltdl'])/libltdlc.la
  LTDLINCL='-I${top_srcdir}/'ifelse($#,1,[$1],['libltdl'])
  # For backwards non-gettext consistent compatibility...
  INCLTDL="$LTDLINCL"
])

# AC_LIBLTDL_INSTALLABLE[(dir)] - sets LIBLTDL to the link flags for
# the libltdl installable library and LTDLINCL to the include flags for
# the libltdl header and adds --enable-ltdl-install to the configure
# arguments.  Note that LIBLTDL and LTDLINCL are not AC_SUBSTed, nor is
# AC_CONFIG_SUBDIRS called.  If DIR is not provided and an installed
# libltdl is not found, it is assumed to be `libltdl'.  LIBLTDL will
# be prefixed with '${top_builddir}/' and LTDLINCL will be prefixed
# with '${top_srcdir}/' (note the single quotes!).  If your package is
# not flat and you're not using automake, define top_builddir and
# top_srcdir appropriately in the Makefiles.
# In the future, this macro may have to be called after AC_PROG_LIBTOOL.
AC_DEFUN([AC_LIBLTDL_INSTALLABLE],
[AC_BEFORE([$0],[AC_LIBTOOL_SETUP])dnl
  AC_CHECK_LIB(ltdl, main,
  [test x"$enable_ltdl_install" != xyes && enable_ltdl_install=no],
  [if test x"$enable_ltdl_install" = xno; then
     AC_MSG_WARN([libltdl not installed, but installation disabled])
   else
     enable_ltdl_install=yes
   fi
  ])
  if test x"$enable_ltdl_install" = x"yes"; then
    ac_configure_args="$ac_configure_args --enable-ltdl-install"
    LIBLTDL='${top_builddir}/'ifelse($#,1,[$1],['libltdl'])/libltdl.la
    LTDLINCL='-I${top_srcdir}/'ifelse($#,1,[$1],['libltdl'])
  else
    ac_configure_args="$ac_configure_args --enable-ltdl-install=no"
    LIBLTDL="-lltdl"
    LTDLINCL=
  fi
  # For backwards non-gettext consistent compatibility...
  INCLTDL="$LTDLINCL"
])

# old names
AC_DEFUN([AM_PROG_LIBTOOL],   [AC_PROG_LIBTOOL])
AC_DEFUN([AM_ENABLE_SHARED],  [AC_ENABLE_SHARED($@)])
AC_DEFUN([AM_ENABLE_STATIC],  [AC_ENABLE_STATIC($@)])
AC_DEFUN([AM_DISABLE_SHARED], [AC_DISABLE_SHARED($@)])
AC_DEFUN([AM_DISABLE_STATIC], [AC_DISABLE_STATIC($@)])
AC_DEFUN([AM_PROG_LD],        [AC_PROG_LD])
AC_DEFUN([AM_PROG_NM],        [AC_PROG_NM])

# This is just to silence aclocal about the macro not being used
ifelse([AC_DISABLE_FAST_INSTALL])

############################################################
# NOTE: This macro has been submitted for inclusion into   #
#  GNU Autoconf as AC_PROG_SED.  When it is available in   #
#  a released version of Autoconf we should remove this    #
#  macro and use it instead.                               #
############################################################
# LT_AC_PROG_SED
# --------------
# Check for a fully-functional sed program, that truncates
# as few characters as possible.  Prefer GNU sed if found.
AC_DEFUN([LT_AC_PROG_SED],
[AC_MSG_CHECKING([for a sed that does not truncate output])
AC_CACHE_VAL(lt_cv_path_SED,
[# Loop through the user's path and test for sed and gsed.
# Then use that list of sed's as ones to test for truncation.
as_executable_p="test -f"
as_save_IFS=$IFS; IFS=$PATH_SEPARATOR
for as_dir in $PATH
do
  IFS=$as_save_IFS
  test -z "$as_dir" && as_dir=.
  for ac_prog in sed gsed; do
    for ac_exec_ext in '' $ac_executable_extensions; do
      if $as_executable_p "$as_dir/$ac_prog$ac_exec_ext"; then
        _sed_list="$_sed_list $as_dir/$ac_prog$ac_exec_ext"
      fi
    done
  done
done

  # Create a temporary directory, and hook for its removal unless debugging.
$debug ||
{
  trap 'exit_status=$?; rm -rf $tmp && exit $exit_status' 0
  trap '{ (exit 1); exit 1; }' 1 2 13 15
}

# Create a (secure) tmp directory for tmp files.
: ${TMPDIR=/tmp}
{
  tmp=`(umask 077 && mktemp -d -q "$TMPDIR/sedXXXXXX") 2>/dev/null` &&
  test -n "$tmp" && test -d "$tmp"
}  ||
{
  tmp=$TMPDIR/sed$$-$RANDOM
  (umask 077 && mkdir $tmp)
} ||
{
   echo "$me: cannot create a temporary directory in $TMPDIR" >&2
   { (exit 1); exit 1; }
}
  _max=0
  _count=0
  # Add /usr/xpg4/bin/sed as it is typically found on Solaris
  # along with /bin/sed that truncates output.
  for _sed in $_sed_list /usr/xpg4/bin/sed; do
    test ! -f ${_sed} && break
    cat /dev/null > "$tmp/sed.in"
    _count=0
    echo ${ECHO_N-$ac_n} "0123456789${ECHO_C-$ac_c}" >"$tmp/sed.in"
    # Check for GNU sed and select it if it is found.
    if "${_sed}" --version 2>&1 < /dev/null | egrep '(GNU)' > /dev/null; then
      lt_cv_path_SED=${_sed}
      break
    fi
    while true; do
      cat "$tmp/sed.in" "$tmp/sed.in" >"$tmp/sed.tmp"
      mv "$tmp/sed.tmp" "$tmp/sed.in"
      cp "$tmp/sed.in" "$tmp/sed.nl"
      echo >>"$tmp/sed.nl"
      ${_sed} -e 's/a$//' < "$tmp/sed.nl" >"$tmp/sed.out" || break
      cmp -s "$tmp/sed.out" "$tmp/sed.nl" || break
      # 40000 chars as input seems more than enough
      test $_count -gt 10 && break
      _count=`expr $_count + 1`
      if test $_count -gt $_max; then
        _max=$_count
        lt_cv_path_SED=$_sed
      fi
    done
  done
  rm -rf "$tmp"
])
if test "X$SED" != "X"; then
  lt_cv_path_SED=$SED
else
  SED=$lt_cv_path_SED
fi
AC_MSG_RESULT([$SED])
])


dnl *** file: config/llnl-ac-macros/ltdl.m4
## ltdl.m4 - Configure ltdl for the target system. -*-Shell-script-*-
## Copyright (C) 1999-2000 Free Software Foundation, Inc.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
##
## As a special exception to the GNU General Public License, if you
## distribute this file as part of a program that contains a
## configuration script generated by Autoconf, you may include it under
## the same distribution terms that you use for the rest of that program.

# serial 5 AC_LIB_LTDL

# AC_WITH_LTDL
# ------------
# Clients of libltdl can use this macro to allow the installer to
# choose between a shipped copy of the ltdl sources or a preinstalled
# version of the library.
AC_DEFUN([AC_WITH_LTDL],
[AC_REQUIRE([AC_LIB_LTDL])
AC_SUBST([LIBLTDL])
AC_SUBST([INCLTDL])

# Unless the user asks us to check, assume no installed ltdl exists.
use_installed_libltdl=no

AC_ARG_WITH([included_ltdl],
    [  --with-included-ltdl    use the GNU ltdl sources included here])

if test "x$with_included_ltdl" != xyes; then
  # We are not being forced to use the included libltdl sources, so
  # decide whether there is a useful installed version we can use.
  AC_CHECK_HEADER([ltdl.h],
      [AC_CHECK_LIB([ltdl], [lt_dlcaller_register],
          [with_included_ltdl=no],
          [with_included_ltdl=yes])
  ])
fi

if test "x$enable_ltdl_install" != xyes; then
  # If the user did not specify an installable libltdl, then default
  # to a convenience lib.
  AC_LIBLTDL_CONVENIENCE
fi

if test "x$with_included_ltdl" = xno; then
  # If the included ltdl is not to be used. then Use the
  # preinstalled libltdl we found.
  AC_DEFINE([HAVE_LTDL], 1,
    [Define this if a modern libltdl is already installed])
  LIBLTDL=-lltdl
fi

# Report our decision...
AC_MSG_CHECKING([whether to use included libltdl])
AC_MSG_RESULT([$with_included_ltdl])

AC_CONFIG_SUBDIRS([libltdl])
])# AC_WITH_LTDL


# AC_LIB_LTDL
# -----------
# Perform all the checks necessary for compilation of the ltdl objects
#  -- including compiler checks and header checks.
AC_DEFUN([AC_LIB_LTDL],
[AC_PREREQ(2.13)
AC_REQUIRE([AC_PROG_CC])
AC_REQUIRE([AC_C_CONST])
AC_REQUIRE([AC_HEADER_STDC])
AC_REQUIRE([AC_HEADER_DIRENT])
AC_REQUIRE([AC_LIBTOOL_HEADER_ASSERT])
AC_REQUIRE([_LT_AC_CHECK_DLFCN])
AC_REQUIRE([AC_LTDL_ENABLE_INSTALL])
AC_REQUIRE([AC_LTDL_SHLIBEXT])
AC_REQUIRE([AC_LTDL_SHLIBPATH])
AC_REQUIRE([AC_LTDL_SYSSEARCHPATH])
AC_REQUIRE([AC_LTDL_OBJDIR])
AC_REQUIRE([AC_LTDL_DLPREOPEN])
AC_REQUIRE([AC_LTDL_DLLIB])
AC_REQUIRE([AC_LTDL_SYMBOL_USCORE])
AC_REQUIRE([AC_LTDL_DLSYM_USCORE])
AC_REQUIRE([AC_LTDL_SYS_DLOPEN_DEPLIBS])
AC_REQUIRE([AC_LTDL_FUNC_ARGZ])

AC_CHECK_HEADERS([errno.h malloc.h memory.h stdlib.h stdio.h ctype.h unistd.h])
AC_CHECK_HEADERS([dl.h sys/dl.h dld.h])
AC_CHECK_HEADERS([string.h strings.h], break)

AC_CHECK_FUNCS([strchr index], break)
AC_CHECK_FUNCS([strrchr rindex], break)
AC_CHECK_FUNCS([memcpy bcopy], break)
AC_CHECK_FUNCS([memmove strcmp])

])# AC_LIB_LTDL

# AC_LTDL_ENABLE_INSTALL
# ----------------------
AC_DEFUN([AC_LTDL_ENABLE_INSTALL],
[AC_ARG_ENABLE(ltdl-install,
[  --enable-ltdl-install   install libltdl])

AM_CONDITIONAL(INSTALL_LTDL, test x"${enable_ltdl_install-no}" != xno)
AM_CONDITIONAL(CONVENIENCE_LTDL, test x"${enable_ltdl_convenience-no}" != xno)
])])# AC_LTDL_ENABLE_INSTALL

# AC_LTDL_SYS_DLOPEN_DEPLIBS
# --------------------------
AC_DEFUN([AC_LTDL_SYS_DLOPEN_DEPLIBS],
[AC_REQUIRE([AC_CANONICAL_HOST])
AC_CACHE_CHECK([whether deplibs are loaded by dlopen],
	libltdl_cv_sys_dlopen_deplibs, [dnl
	# PORTME does your system automatically load deplibs for dlopen()?
	libltdl_cv_sys_dlopen_deplibs=unknown
	case "$host_os" in
        hpux10*|hpux11*)
          libltdl_cv_sys_dlopen_deplibs=yes
          ;;
	linux*)
	  libltdl_cv_sys_dlopen_deplibs=yes
	  ;;
	netbsd*)
	  libltdl_cv_sys_dlopen_deplibs=yes
	  ;;
	openbsd*)
	  libltdl_cv_sys_dlopen_deplibs=yes
	  ;;
	solaris*)
	  libltdl_cv_sys_dlopen_deplibs=yes
	  ;;
	esac
])
if test "$libltdl_cv_sys_dlopen_deplibs" != yes; then
 AC_DEFINE(LTDL_DLOPEN_DEPLIBS, 1,
    [Define if the OS needs help to load dependent libraries for dlopen(). ])
fi
])# AC_LTDL_SYS_DLOPEN_DEPLIBS

# AC_LTDL_SHLIBEXT
# ----------------
AC_DEFUN([AC_LTDL_SHLIBEXT],
[AC_REQUIRE([_LT_AC_LTCONFIG_HACK])
AC_CACHE_CHECK([which extension is used for shared libraries],
  libltdl_cv_shlibext,
[ac_last=
  for ac_spec in $library_names_spec; do
    ac_last="$ac_spec"
  done
  echo "$ac_last" | [sed 's/\[.*\]//;s/^[^.]*//;s/\$.*$//;s/\.$//'] > conftest
libltdl_cv_shlibext=`cat conftest`
rm -f conftest
])
if test -n "$libltdl_cv_shlibext"; then
  AC_DEFINE_UNQUOTED(LTDL_SHLIB_EXT, "$libltdl_cv_shlibext",
    [Define to the extension used for shared libraries, say, ".so". ])
fi
])# AC_LTDL_SHLIBEXT

# AC_LTDL_SHLIBPATH
# -----------------
AC_DEFUN([AC_LTDL_SHLIBPATH],
[AC_REQUIRE([_LT_AC_LTCONFIG_HACK])
AC_CACHE_CHECK([which variable specifies run-time library path],
  libltdl_cv_shlibpath_var, [libltdl_cv_shlibpath_var="$shlibpath_var"])
if test -n "$libltdl_cv_shlibpath_var"; then
  AC_DEFINE_UNQUOTED(LTDL_SHLIBPATH_VAR, "$libltdl_cv_shlibpath_var",
    [Define to the name of the environment variable that determines the dynamic library search path. ])
fi
])# AC_LTDL_SHLIBPATH

# AC_LTDL_SYSSEARCHPATH
# ---------------------
AC_DEFUN([AC_LTDL_SYSSEARCHPATH],
[AC_REQUIRE([_LT_AC_LTCONFIG_HACK])
AC_CACHE_CHECK([for the default library search path],
  libltdl_cv_sys_search_path, [libltdl_cv_sys_search_path="$sys_lib_dlsearch_path_spec"])
if test -n "$libltdl_cv_sys_search_path"; then
  case "$host" in
  *-*-mingw*) pathsep=";" ;;
  *) pathsep=":" ;;
  esac
  sys_search_path=
  for dir in $libltdl_cv_sys_search_path; do
    if test -z "$sys_search_path"; then
      sys_search_path="$dir"
    else
      sys_search_path="$sys_search_path$pathsep$dir"
    fi
  done
  AC_DEFINE_UNQUOTED(LTDL_SYSSEARCHPATH, "$sys_search_path",
    [Define to the system default library search path. ])
fi
])# AC_LTDL_SYSSEARCHPATH

# AC_LTDL_OBJDIR
# --------------
AC_DEFUN([AC_LTDL_OBJDIR],
[AC_CACHE_CHECK([for objdir],
  libltdl_cv_objdir, [libltdl_cv_objdir="$objdir"
if test -n "$objdir"; then
  :
else
  rm -f .libs 2>/dev/null
  mkdir .libs 2>/dev/null
  if test -d .libs; then
    libltdl_cv_objdir=.libs
  else
    # MS-DOS does not allow filenames that begin with a dot.
    libltdl_cv_objdir=_libs
  fi
rmdir .libs 2>/dev/null
fi])
AC_DEFINE_UNQUOTED(LTDL_OBJDIR, "$libltdl_cv_objdir/",
  [Define to the sub-directory in which libtool stores uninstalled libraries. ])
])# AC_LTDL_OBJDIR

# AC_LTDL_DLPREOPEN
# -----------------
AC_DEFUN([AC_LTDL_DLPREOPEN],
[AC_REQUIRE([AC_LIBTOOL_SYS_GLOBAL_SYMBOL_PIPE])dnl
AC_CACHE_CHECK([whether libtool supports -dlopen/-dlpreopen],
       libltdl_cv_preloaded_symbols, [dnl
  if test -n "$global_symbol_pipe"; then
    libltdl_cv_preloaded_symbols=yes
  else
    libltdl_cv_preloaded_symbols=no
  fi
])
if test x"$libltdl_cv_preloaded_symbols" = x"yes"; then
  AC_DEFINE(HAVE_PRELOADED_SYMBOLS, 1,
    [Define if libtool can extract symbol lists from object files. ])
fi
])# AC_LTDL_DLPREOPEN

# AC_LTDL_DLLIB
# -------------
AC_DEFUN([AC_LTDL_DLLIB],
[LIBADD_DL=
AC_SUBST(LIBADD_DL)

AC_CHECK_FUNC([shl_load],
      [AC_DEFINE([HAVE_SHL_LOAD], [1],
		 [Define if you have the shl_load function.])],
  [AC_CHECK_LIB([dld], [shl_load],
	[AC_DEFINE([HAVE_SHL_LOAD], [1],
		   [Define if you have the shl_load function.])
	LIBADD_DL="$LIBADD_DL -ldld"],
    [AC_CHECK_LIB([dl], [dlopen],
	  [AC_DEFINE([HAVE_LIBDL], [1],
		     [Define if you have the libdl library or equivalent.])
	  LIBADD_DL="-ldl"],
      [AC_TRY_LINK([#if HAVE_DLFCN_H
#  include <dlfcn.h>
#endif
      ],
	[dlopen(0, 0);],
	    [AC_DEFINE([HAVE_LIBDL], [1],
		       [Define if you have the libdl library or equivalent.])],
	[AC_CHECK_LIB([svld], [dlopen],
	      [AC_DEFINE([HAVE_LIBDL], [1],
			 [Define if you have the libdl library or equivalent.])
	      LIBADD_DL="-lsvld"],
	  [AC_CHECK_LIB([dld], [dld_link],
	        [AC_DEFINE([HAVE_DLD], [1],
			   [Define if you have the GNU dld library.])
	 	LIBADD_DL="$LIBADD_DL -ldld"
          ])
        ])
      ])
    ])
  ])
])

if test "x$ac_cv_func_dlopen" = xyes || test "x$ac_cv_lib_dl_dlopen" = xyes; then
 LIBS_SAVE="$LIBS"
 LIBS="$LIBS $LIBADD_DL"
 AC_CHECK_FUNCS(dlerror)
 LIBS="$LIBS_SAVE"
fi
])# AC_LTDL_DLLIB

# AC_LTDL_SYMBOL_USCORE
# ---------------------
AC_DEFUN([AC_LTDL_SYMBOL_USCORE],
[dnl does the compiler prefix global symbols with an underscore?
AC_REQUIRE([AC_LIBTOOL_SYS_GLOBAL_SYMBOL_PIPE])dnl
AC_MSG_CHECKING([for _ prefix in compiled symbols])
AC_CACHE_VAL(ac_cv_sys_symbol_underscore,
[ac_cv_sys_symbol_underscore=no
cat > conftest.$ac_ext <<EOF
void nm_test_func(){}
int main(){nm_test_func;return 0;}
EOF
if AC_TRY_EVAL(ac_compile); then
  # Now try to grab the symbols.
  ac_nlist=conftest.nm
  if AC_TRY_EVAL(NM conftest.$ac_objext \| $global_symbol_pipe \> $ac_nlist) && test -s "$ac_nlist"; then
    # See whether the symbols have a leading underscore.
    if egrep '^. _nm_test_func' "$ac_nlist" >/dev/null; then
      ac_cv_sys_symbol_underscore=yes
    else
      if egrep '^. nm_test_func ' "$ac_nlist" >/dev/null; then
	:
      else
	echo "configure: cannot find nm_test_func in $ac_nlist" >&AC_FD_CC
      fi
    fi
  else
    echo "configure: cannot run $global_symbol_pipe" >&AC_FD_CC
  fi
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.c >&AC_FD_CC
fi
rm -rf conftest*
])
AC_MSG_RESULT($ac_cv_sys_symbol_underscore)
])# AC_LTDL_SYMBOL_USCORE


# AC_LTDL_DLSYM_USCORE
# --------------------
AC_DEFUN([AC_LTDL_DLSYM_USCORE],
[AC_REQUIRE([AC_LTDL_SYMBOL_USCORE])dnl
if test x"$ac_cv_sys_symbol_underscore" = xyes; then
  if test x"$ac_cv_func_dlopen" = xyes ||
     test x"$ac_cv_lib_dl_dlopen" = xyes ; then
	AC_CACHE_CHECK([whether we have to add an underscore for dlsym],
		libltdl_cv_need_uscore, [dnl
		libltdl_cv_need_uscore=unknown
                save_LIBS="$LIBS"
                LIBS="$LIBS $LIBADD_DL"
		_LT_AC_TRY_DLOPEN_SELF(
		  libltdl_cv_need_uscore=no, libltdl_cv_need_uscore=yes,
		  [],			     libltdl_cv_need_uscore=cross)
		LIBS="$save_LIBS"
	])
  fi
fi

if test x"$libltdl_cv_need_uscore" = xyes; then
  AC_DEFINE(NEED_USCORE, 1,
    [Define if dlsym() requires a leading underscore in symbol names. ])
fi
])# AC_LTDL_DLSYM_USCORE


# AC_CHECK_TYPES(TYPES, [ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND],
#                [INCLUDES])
# ---------------------------------------------------------------
# This macro did not exist in Autoconf 2.13, which we do still support
ifdef([AC_CHECK_TYPES], [],
[define([AC_CHECK_TYPES],
  [AC_CACHE_CHECK([for $1], ac_Type,
    [AC_TRY_LINK([$4],
	[if (($1 *) 0)
	  return 0;
	if (sizeof ($1))
	  return 0;],
	[ac_Type=yes],
	[ac_Type=no])])
  if test "x$ac_Type" = xyes; then
    ifelse([$2], [], [:], [$2])
  else
    ifelse([$3], [], [:], [$3])
  fi])
])# AC_CHECK_TYPES


# AC_LTDL_FUNC_ARGZ
# -----------------
AC_DEFUN([AC_LTDL_FUNC_ARGZ],
[AC_CHECK_HEADERS([argz.h])

AC_CHECK_TYPES([error_t],
  [],
  [AC_DEFINE([error_t], [int],
    [Define to a type to use for \`error_t' if it is not otherwise available.])],
  [#if HAVE_ARGZ_H
#  include <argz.h>
#endif])

AC_CHECK_FUNCS([argz_append argz_create_sep argz_insert argz_next argz_stringify])
])# AC_LTDL_FUNC_ARGZ
