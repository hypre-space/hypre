AC_DEFUN([LLNL_PROG_JAR],[
AC_REQUIRE([AC_EXEEXT])dnl
if test "x$JAVAPREFIX" = x; then
        test "x$JAR" = x && AC_CHECK_PROGS(JAR, jar$EXEEXT)
else
        test "x$JAR" = x && AC_CHECK_PROGS(JAR, jar, $JAVAPREFIX)
fi
test "x$JAR" = x && AC_MSG_ERROR([no acceptable jar program found in \$PATH])
AC_PROVIDE([$0])dnl
])
