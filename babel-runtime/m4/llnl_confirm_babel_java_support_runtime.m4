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
AC_ARG_VAR([JAVAPREFIX],[Directory where Java binaries are installed  @<:@e.g. $JAVAPREFIX/$JAVA@:>@])
AC_ARG_VAR([JAVAC],[Java Compiler])
AC_ARG_VAR([JAVACFLAGS],[Flags for Java Compiler])
AC_ARG_VAR([JAVA],[Java Runtime])
AC_ARG_VAR([JAVAFLAGS],[Flags for Java Runtime])
AC_ARG_VAR([JAVAH],[JNI C stub and header generator])
AC_ARG_VAR([JAR],[Java Archive Tool])
AC_ARG_VAR([JNI_INCLUDES],[Preprocessor flags used @<:@e.g., -I$JAVAPREFIX/include@:>@ to #include <jni.h>])

# First determine enable_java == true or false, and value of $JAVA (before testing JAVAPREFIX)
AC_ARG_ENABLE([java],
AS_HELP_STRING(--enable-java@<:@=JVM@:>@,java languange bindings @<:@default=yes@:>@),
	[enable_java="$enableval"],
	[enable_java="yes"])
test -z "$enable_java" && enable_java="yes" #zero length is yes

if test "x$enable_java" = "xno"; then
  # --enable-java=no is equiv to --disable-java
  msgs="$msgs
	  Java support disabled by request"
elif test "x$enable_java" = "xyes"; then
  test -z "$JAVA" && JAVA=java
else # --enable-java has an explicit value
  if test -n "$JAVA"; then 
    if test "$JAVA" != "$enable_java"; then 
      AC_MSG_WARN([--enable-java=]"$enable_java"[, and JAVA=]"$JAVA"[, using former])
    fi
  fi
  JAVA="$enable_java"
  enable_java=yes
fi;

OLD_PATH=$PATH
if test "x$enable_java" = "xno"; then
  AC_MSG_WARN([Skipping Java configuration in runtime library.])
  JAVA=""
  JAVAC=""
  JAR=""
  JAVAPREFIX=""
  JAVAH=""
else
  # now resolve `dirname $JAVA` and $JAVAPREFIX
  tmp=`basename $JAVA`
  if test "$tmp" = "$JAVA"; then
    if test -z "$JAVAPREFIX"; then
      #easy case.  $JAVA=="java" and $JAVAPREFIX=""
      :
    else
      #not hard. $JAVA=="java" and $JAVAPREFIX="some/path"
      :
    fi
  else
    if test -z "$JAVAPREFIX"; then
      #little strange. $JAVA=="some/path/java" and $JAVAPREFIX=""
      #set JAVAPREFIX=`dirname $JAVA`
      JAVAPREFIX=`dirname $JAVA`
      JAVA=`basename $JAVA`
    else
     #way wierd. $JAVA=="one/path/java" and $JAVAPREFIX="possibly/another/path"
      if test "$tmp" != "$JAVAPREFIX"; then 
        AC_MSG_ERROR([Confused by 'dirname $JAVA'=]"$tmp"[ and $JAVAPREFIX=]"$JAVAPREFIX")
      fi
    fi
  fi


  if test -n "$JAVAPREFIX"; then
    PATH=$JAVAPREFIX:${PATH}
  fi

  LLNL_CHECK_CLASSPATH
  test -z "$JAVAC" && JAVAC=javac
  AC_PROG_JAVAC
  LLNL_PROG_JAVA
  LLNL_CHECK_JAVA_ADDCLASSPATH_FLAG

  test -z "$JAR" && JAR=jar
  LLNL_PROG_JAR
  
  AC_TRY_COMPILE_JAVA
  test -z "$JAVADOC" && JAVADOC=javadoc
  AC_PROG_JAVADOC
  test -z "$JAVAH" && JAVAH=javah
  LLNL_PROG_JAVAH
  if test "X$llnl_cv_header_jni_h" = "Xno"; then
    AC_MSG_WARN([Cannot find jni.h, Java support will be disabled])
    AC_MSG_WARN([Try setting JNI_INCLUDES and rerunning configure])
    enable_java=no
    msgs="$msgs
	  Java support disabled against request (no jni.h found!)"
  fi
  if test -z "$llnl_cv_lib_jvm"; then
    AC_MSG_WARN([Cannot find JVM shared library, Java support will be disabled])
    enable_java=no
    msgs="$msgs
	  Java support disabled against request 
            (no jvm.dll/libjvm.so/libjvm.a found!)"
  fi
fi
if test "X$enable_java" = "Xno"; then
  AC_DEFINE(JAVA_DISABLED, 1, [If defined, Java support was disabled at configure time])
else
  msgs="$msgs
 	  Java enabled.";
fi 
AM_CONDITIONAL(SUPPORT_JAVA, test "X$enable_java" != "Xno")
if test "$llnl_cv_jni_includes" = "none needed"; then
  JNI_INCLUDES=""
else
  JNI_INCLUDES="$llnl_cv_jni_includes"
fi
AC_SUBST(JAVAPREFIX)dnl 
AC_SUBST(JNI_INCLUDES)
JNI_LDFLAGS="$llnl_cv_jni_linker_flags"
AC_SUBST(JNI_LDFLAGS)

LLNL_WHICH_PROG(WHICH_JAVA,$JAVAPREFIX)
LLNL_WHICH_PROG(WHICH_JAVAC,$JAVAPREFIX)
LLNL_WHICH_PROG(WHICH_JAR,$JAVAPREFIX)
LLNL_WHICH_PROG(WHICH_JAVAH,$JAVAPREFIX)

AC_SUBST(JAVAC)dnl
AC_SUBST(JAVACFLAGS)dnl
AC_SUBST(JAVA)dnl
AC_SUBST(JAVAFLAGS)dnl
PATH=$OLD_PATH
])
