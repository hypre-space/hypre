
dnl @synopsis LLNL_PREVENT_UNHOLY_GNU_NONGNU_MIX
dnl
dnl Check for unholy mixture of GNU and non-GNU compilers/linkers/etc
dnl on certain platforms.  Linux is more tolerable than others.
dnl
dnl @author Gary Kumfert 
AC_DEFUN([LLNL_PREVENT_UNHOLY_GNU_NONGNU_MIX],
[ AC_REQUIRE([AC_CANONICAL_HOST])dnl
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
