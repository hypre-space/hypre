dnl
dnl @synopsis LLNL_PYTHON_SHARED_LIBRARY
dnl
dnl @author ?

AC_DEFUN([LLNL_PYTHON_SHARED_LIBRARY],[
  AC_REQUIRE([AC_LTDL_SHLIBEXT])dnl
  AC_REQUIRE([LLNL_PYTHON_LIBRARY])dnl
  AC_REQUIRE([AC_LTDL_SHLIBPATH])dnl
  AC_MSG_CHECKING([if Python shared library is available])

  llnl_python_shared_library_found=no

  case "$target_os" in
    cygwin*)
      llnl_python_shared_library="libpython$llnl_cv_python_version.dll"
      ;;
    aix*)
      llnl_python_shared_library="libpython$llnl_cv_python_version.sl"
      ;;
    *)
      llnl_python_shared_library="libpython$llnl_cv_python_version$libltdl_cv_shlibext"
      ;;
  esac

  llnl_python_shared_lib_path=`env | grep "^${libltdl_cv_shlibpath_var}=" | sed "s/^${libltdl_cv_shlibpath_var}=//"`
  llnl_ld_extra=
  if test -f /etc/ld.so.conf; then
    llnl_ld_extra = `$SED -e 's/[:,\t]/ /g;s/=[^=]*$//;s/=[^= ]* / /g' /etc/ld.so.conf | tr '\n' ' '`
  fi
  llnl_ld_extra_dir=
  if test -d /etc/ld.so.conf.d; then
    llnl_ld_extra_dir = `cat /etc/ld.so.conf.d/* 2>/dev/null | $SED -e 's/[:,\t]/ /g;s/=[^=]*$//;s/=[^= ]* / /g' | tr '\n' ' '`
  fi
  for f in `echo $llnl_python_shared_lib_path | tr ';:' '  '` $llnl_cv_python_library/config /bin /lib /usr/lib /usr/lib64 $llnl_ld_extra $llnl_ld_extra_dir ; do
    if test -f "$f/$llnl_python_shared_library"; then
      llnl_python_shared_library_found=yes
      llnl_python_shared_library="$f/$llnl_python_shared_library"
      llnl_python_shared_library_dir="$f"
      break
    fi
  done

  if test "$llnl_python_shared_library_found" != "yes"; then
    case "$target_os" in
      darwin*)
        llnl_python_shared_library="libpython$llnl_cv_python_version.dylib"
	llnl_ld_extra=
	if test -f /etc/ld.so.conf; then
	  llnl_ld_extra = `$SED -e 's/[:,\t]/ /g;s/=[^=]*$//;s/=[^= ]* / /g' /etc/ld.so.conf | tr '\n' ' '`
	fi
	llnl_ld_extra_dir=
	if test -d /etc/ld.so.conf.d; then
	  llnl_ld_extra_dir = `cat /etc/ld.so.conf.d/* 2>/dev/null | $SED -e 's/[:,\t]/ /g;s/=[^=]*$//;s/=[^= ]* / /g' | tr '\n' ' '`
	fi
        for f in `echo $llnl_python_shared_lib_path | tr ';:' '  '` $llnl_cv_python_library/config /bin /lib /usr/lib /usr/lib64 $llnl_ld_extra $llnl_ld_extra_dir ; do
          if test -f "$f/$llnl_python_shared_library"; then
            llnl_python_shared_library_found=yes
            llnl_python_shared_library="$f/$llnl_python_shared_library"
            llnl_python_shared_library_dir="$f"
            break
          fi
        done
        ;;
    esac
  fi

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
