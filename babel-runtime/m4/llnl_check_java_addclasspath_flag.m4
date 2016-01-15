dnl @synopsis LLNL_CHECK_JAVA_ADDCLASSPATH_FLAG
dnl  
dnl Defines JAVA_ADDCLASSPATH_FLAG to be either 
dnl -addclasspath or -classpath.  Kaffe prefers former
dnl Sun and GCJ the latter
dnl
dnl @author Gary Kumfert

AC_DEFUN([LLNL_CHECK_JAVA_ADDCLASSPATH_FLAG], 
[AC_REQUIRE([LLNL_PROG_JAVA])dnl
  AC_CACHE_CHECK(if $JAVA uses -addclasspath or -classpath, 
	         llnl_cv_check_java_addclasspath, [
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
    if AC_TRY_COMMAND($JAVAC $JAVACFLAGS $JAVA_TEST) && test -s $CLASS_TEST; then
      :
    else
      echo "configure: failed program was:" >&AS_MESSAGE_LOG_FD()
      cat $JAVA_TEST >&AS_MESSAGE_LOG_FD()
      AC_MSG_ERROR(The Java compiler $JAVAC failed (see config.log, check the CLASSPATH?))
    fi
    if AC_TRY_COMMAND($JAVA $JAVAFLAGS -addclasspath . $TEST) >/dev/null 2>&1; then
      llnl_cv_check_java_addclasspath="-addclasspath"
      rm -fr $JAVA_TEST $CLASS_TEST
    elif AC_TRY_COMMAND($JAVA $JAVAFLAGS -classpath . $TEST) >/dev/null 2>&1; then
      llnl_cv_check_java_addclasspath="-classpath"
      rm -fr $JAVA_TEST $CLASS_TEST
    elif AC_TRY_COMMAND($JAVA $JAVAFLAGS $TEST) >/dev/null 2>&1; then
      echo "configure: failed program was:" >&AS_MESSAGE_LOG_FD()
      cat $JAVA_TEST >&AS_MESSAGE_LOG_FD()
      AC_MSG_ERROR($JAVA $JAVAFLAGS $TEST failed with both -classpath and -addclasspath )
    else 
      llnl_cv_check_java_addclasspath=
      rm -fr $JAVA_TEST $CLASS_TEST
    fi
  ])
  JAVA_ADDCLASSPATH_FLAG=$llnl_cv_check_java_addclasspath
  AC_SUBST(JAVA_ADDCLASSPATH_FLAG)
])

