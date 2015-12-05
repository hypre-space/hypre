
dnl dnl @synopsis LLNL_PROG_JAVAH
dnl
dnl LLNL_PROG_JAVAH tests the availability of the javah header generator
dnl and looks for the jni.h header file. If available, JAVAH is set to
dnl the full path of javah.  Unlike Luc's implementation, this doesn't
dnl update CPPFLAGS.  Instead it defines JNI_INCLUDES.
dnl
dnl @author Luc Maisonobe
dnl @version $Id: llnl_prog_javah.m4,v 1.7 2007/09/27 19:35:18 painter Exp $
dnl
dnl It also creates JNI_LDFLAGS for wherever libjava and libjvm are
dnl 
AC_DEFUN([LLNL_PROG_JAVAH],[
AC_PATH_PROG(JAVAH,javah)
LLNL_HEADER_JNI
LLNL_JNI_INCLUDE_FLAGS
LLNL_LIB_JAVA
LLNL_LIB_JVM_DIR
LLNL_LIB_JVM
LLNL_JNI_LINKER_FLAGS
])


dnl
dnl defines llnl_cv_header_jni_h if jni.h can be found from 
dnl parsing JNI_INCLUDES or ac_cv_path_JAVAH
dnl
AC_DEFUN([LLNL_HEADER_JNI],[
AC_CACHE_CHECK([for location of jni.h],[llnl_cv_header_jni_h],[
llnl_cv_header_jni_h=no;
if test -n "$JNI_INCLUDES"; then

  incl_guess=`echo "$JNI_INCLUDES" | sed 's,\-I, ,g'`

  for i in $incl_guess ; do
    if test -e "$i/jni.h"; then
      llnl_cv_header_jni_h="$i/jni.h";
    fi
  done 
fi
if test "x$llnl_cv_header_jni_h" = xno; then
  if test "x$ac_cv_path_JAVAH" != x ; then 
changequote(, )dnl
     javah_guess="`echo $ac_cv_path_JAVAH | sed 's,\(.*\)//*[^/]*//*[^/]*$,\1/include,'` /usr/java/include /usr/local/java/include /System/Library/Frameworks/JavaVM.framework/Headers"
changequote([, ])dnl
     for i in $javah_guess ; do
      if test -e "$i/jni.h"; then
        llnl_cv_header_jni_h="$i/jni.h"
      fi
    done 
  fi
fi
])])


dnl
dnl
dnl sets llnl_cv_jni_includes to either list of -I directives for "#include <jni.h>" to work
dnl                           or empty
dnl                      
dnl
AC_DEFUN([LLNL_JNI_INCLUDE_FLAGS],[
AC_CACHE_CHECK([what additional include directives are needed for <jni.h>],[llnl_cv_jni_includes],[
ac_save_CPPFLAGS="$CPPFLAGS"
llnl_cv_jni_includes="no"
if test -n "$JNI_INCLUDES"; then
  llnl_fix_jni_includes="$JNI_INCLUDES -I"
  JNI_INCLUDES=""
  for i in $llnl_fix_jni_includes; do
    case $i in
    -I) 
        ;;
    -I*)
       JNI_INCLUDES="$JNI_INCLUDES $i"
       ;;
    *)
       if test -d $i; then
         JNI_INCLUDES="$JNI_INCLUDES -I$i"
       fi
       ;;
    esac
  done
  unset llnl_fix_jni_includes
fi
if test -z "$JNI_INCLUDES"; then
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <jni.h>]],[])],
	[llnl_cv_jni_includes="none needed"])
fi
dnl next try with JNI_INCLUDES
if test "x$llnl_cv_jni_includes" = xno; then 
  if test -n "$JNI_INCLUDES"; then
    CPPFLAGS="$ac_save_CPPFLAGS $JNI_INCLUDES"
  fi 
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <jni.h>]],[])],[llnl_cv_jni_includes="$JNI_INCLUDES"])
fi
if test "x$llnl_cv_jni_includes" = xno -a -n "$JNI_INCLUDES"; then
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <jni.h>]],[])],
	[AC_MSG_WARN([ignoring JNI_INCLUDES setting])
         llnl_cv_jni_includes="none needed"])
fi
dnl finally, try to compute exactly
 if test "x$llnl_cv_jni_includes" = xno; then
changequote(, )dnl
    if test -f "$llnl_cv_header_jni_h"; then
      ac_dir=`dirname "$llnl_cv_header_jni_h"`
    else
      ac_dir=`echo $ac_cv_path_JAVAH | sed 's,\(.*\)//*[^/]*//*[^/]*$,\1/include,'`
    fi
    ac_machdep=`echo $build_os | sed 's,[-0-9].*,,' | sed 's,cygwin,win32,'`
changequote([, ])dnl
    if test -d "$ac_dir"; then 
	if test -d "$ac_dir/$ac_machdep"; then	
	  :
        else 
	  AC_MSG_WARN([computed machine dependent dir ($ac_dir/$ac_machdep) for <jni.h> does not exist])
        fi
    else 
	AC_MSG_WARN([computed include dirs ($ac_dir) for <jni.h> do not exist])
    fi 
    computed_includes="$JNI_INCLUDES -I$ac_dir/$ac_machdep"
    CPPFLAGS="$ac_save_CPPFLAGS $computed_includes"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <jni.h>]],[])],
	[llnl_cv_jni_includes="$computed_includes"])
    if test "x$llnl_cv_jni_includes" = xno; then
dnl try specifying both the generic and machine specific directories
	computed_includes="$JNI_INCLUDES -I$ac_dir -I$ac_dir/$ac_machdep"
	CPPFLAGS="$ac_save_CPPFLAGS $computed_includes"
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <jni.h>]],[])],
	    [llnl_cv_jni_includes="$computed_includes"])
    fi
 fi
CPPFLAGS="$ac_save_CPPFLAGS"
])
if test "x$llnl_cv_jni_includes" = xno; then 
  AC_MSG_ERROR([cannot compile anything with <jni.h>, please set JNI_INCLUDES and reconfigure])
fi 
])


dnl
dnl creates llnl_cv_lib_java which should be an absolute path to libjava.so
dnl
AC_DEFUN([LLNL_LIB_JAVA],[AC_REQUIRE([LLNL_HEADER_JNI])dnl
AC_CACHE_CHECK([for path to libjava.{a,so}],
	[llnl_cv_lib_java],
[javatopdir=`dirname "$llnl_cv_header_jni_h"`
 javatopdir=`dirname $javatopdir`
 case $host_os in 
   cygwin* | mingw* | pw23* ) 
     llnl_cv_lib_java=`find $javatopdir -follow -name "java.dll" -exec dirname {} \; 2>/dev/null | tr "\n" " "`
     ;;
   aix*) 
     llnl_cv_lib_java=`find $javatopdir -name "libjava.*" -exec dirname {} \; 2>/dev/null | tr "\n" " "`
     ;;
   *) 
     llnl_cv_lib_java=`find $javatopdir -follow -name "libjava.*" -exec dirname {} \; 2>/dev/null | tr "\n" " "`
     ;;
 esac
 ])
LIBJAVA_DIR=`echo "$llnl_cv_lib_java" | sed -e 's/  */:/g'`
AC_SUBST(LIBJAVA_DIR)
])

dnl
dnl creates llnl_cv_lib_jvm which should be an absolute path to libjava.so
dnl
AC_DEFUN([LLNL_LIB_JVM],[AC_REQUIRE([LLNL_LIB_JAVA])dnl
AC_CACHE_CHECK([for path to libjvm.{a,so} or client/libjvm.{a,so} ],
	[llnl_cv_lib_jvm],
[javatopdir=`dirname "$llnl_cv_header_jni_h"`
 javatopdir=`dirname "$javatopdir"`
 case $host_os in 
   cygwin* | mingw* | pw23* ) 
     llnl_cv_lib_jvm=`find $javatopdir -follow \
	-name "jvm.dll" -print 2> /dev/null | head -n 1`
     ;;
   darwin*)
     llnl_cv_lib_jvm=`find $javatopdir -follow \
	-name "libjvm_compat.*" -print  2> /dev/null | head -n 1`
     ;;
   aix*)
     llnl_cv_lib_jvm=`find $javatopdir  \
	-name "libjvm.*" -print 2> /dev/null | head -n 1`
     if test -z "$llnl_cv_lib_jvm"; then
	llnl_cv_lib_jvm=`find $javatopdir \
	   -name "libkaffevm.*" -print 2> /dev/null | head -n 1`
     fi
     ;;
   *)
     llnl_cv_lib_jvm=`find $javatopdir -follow \
	-name "libjvm.*" -print 2> /dev/null | head -n 1`
     if test -z "$llnl_cv_lib_jvm"; then
	llnl_cv_lib_jvm=`find $javatopdir -follow \
	   -name "libkaffevm.*" -print 2> /dev/null | head -n 1`
     fi
     ;;
 esac
])
AC_DEFINE_UNQUOTED(JVM_SHARED_LIBRARY, "$llnl_cv_lib_jvm",[Fully qualified string name of the Java Virtual Machine shared library])
])

dnl
dnl creates llnl_cv_lib_jvm_dir which should be an absolute path to libjava.so
dnl
AC_DEFUN([LLNL_LIB_JVM_DIR],[AC_REQUIRE([LLNL_LIB_JAVA])dnl
AC_CACHE_CHECK([for directory where libjvm.{a,so} or client/libjvm.{a,so} resides],
	[llnl_cv_lib_jvm_dir],
[javatopdir=`dirname "$llnl_cv_header_jni_h"`
 javatopdir=`dirname "$javatopdir"`
 case $host_os in 
   cygwin* | mingw* | pw23* ) 
     llnl_cv_lib_jvm_dir=`find $javatopdir -follow \
	-name "jvm.dll" -exec dirname {} \; 2> /dev/null | tr "\n" " "`
     ;;
   aix*)
     llnl_cv_lib_jvm_dir=`find $javatopdir \
	-name "libjvm.*" -exec dirname {} \; 2> /dev/null | tr "\n" " "`
     if test -z "$llnl_cv_lib_jvm_dir"; then
	llnl_cv_lib_jvm_dir=`find $javatopdir \
	   -name "libkaffevm.*" -exec dirname {} \; 2> /dev/null | tr "\n" " "`
     fi
     ;;
   *)
     llnl_cv_lib_jvm_dir=`find $javatopdir -follow  \
	-name "libjvm.*" -exec dirname {} \;  2> /dev/null | tr "\n" " "`
     if test -z "$llnl_cv_lib_jvm_dir"; then
	llnl_cv_lib_jvm_dir=`find $javatopdir -follow  \
	   -name "libkaffevm.*" -exec dirname {} \; 2> /dev/null | tr "\n" " "`
     fi
     ;;
 esac
])
LIBJVM_DIR=`echo "$llnl_cv_lib_jvm_dir" | sed -e 's/  */:/g'`
AC_SUBST(LIBJVM_DIR)
])

AC_DEFUN([LLNL_JNI_LINKER_FLAGS],[AC_REQUIRE([LLNL_LIB_JAVA])dnl
AC_REQUIRE([LLNL_LIB_JVM_DIR])dnl
AC_CACHE_CHECK([what JNI_LDFLAGS needed],[llnl_cv_jni_linker_flags],[
if test "x$JNI_LDFLAGS" = "x"; then
  JNI_LDFLAGS=
  for i in $llnl_cv_lib_java $llnl_cv_lib_jvm_dir; do
    JNI_LDFLAGS="$JNI_LDFLAGS -L$i -R$i"
  done
  JNI_LDFLAGS="$JNI_LDFLAGS"
fi
llnl_cv_jni_linker_flags="$JNI_LDFLAGS"
])
])

