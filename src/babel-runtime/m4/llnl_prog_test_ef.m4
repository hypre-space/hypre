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
  AC_MSG_WARN([Cannot find "test" program such that "test FILE1 -ef FILE2".\n Babel has a python-based backup (hope you have python)! \n Set TEST environment variable and rerun configure to override ])
  TEST_EF="None"
else
  TEST_EF=$llnl_cv_prog_test_ef
fi
  AC_SUBST(TEST_EF)
])
