# # Process this file with autoconf to produce a configure script.

# Table of contents:
#  Variable initialization
#  Basic startup stuff
#  Additional configure options.
#  Critical applications
#  C Compiler
#     a.  Libraries (existence)
#     b.  Header Files
#     c.  Typedefs, Structs, Compiler Characteristics
#     d.  Specific Library Functions.
#  F77 Compiler
#     a. Libraries (existence)
#  C++ Compiler
#     a. Libraries (existence)
#     b. Header Files
#  Java Support
#  Python Support
#  F90 Compiler
#     a. Libraries (existence)
#  Libtool Options
#  Cygwin conditional Makefile Support
#  Shared/Dynamic Library Details
#  babel_internal.h headers and footers
#  File Generation

############################################################
#
# Variable initialization
#
enable_static=yes
enable_shared=yes
with_fortran77=yes
with_fortran90=yes
with_cxx=yes
with_python=yes
with_java=yes
msgs="
	Recap:"

############################################################
#
# Basic startup stuff
#
AC_INIT([babel-runtime],[0.10.8],[babel-bugs@cca-forum.org])
AC_CONFIG_SRCDIR([sidl/sidlArray.h])
AC_CONFIG_AUX_DIR([config])
AC_CONFIG_MACRO_DIR([m4])
AC_CANONICAL_TARGET

AM_INIT_AUTOMAKE([foreign no-dependencies no-installinfo no-installman no-texinfo.tex 1.8])
AC_CONFIG_HEADERS([sidl/babel_internal.h])

LLNL_CHECK_AUTOMAKE(1.9.3)
LLNL_CHECK_AUTOCONF(2.59)


############################################################
#
# Additions for hypre
#
AC_PREFIX_DEFAULT(`pwd`/../hypre)
with_fortran90=no
# ... was with_fortran90=yes    which doesn't work on frost
enable_shared=no

############################################################
#
# Additional configure options. (all "yes" by default)
#
AM_CONDITIONAL(BABEL_RUNTIME_ONLY, ! ((grep UCRL-CODE-2002-054 $srcdir/../COPYRIGHT > /dev/null 2>&1) && (test -e ../bin/babel) ))

AC_PROG_CC
############################################################
#
# F90 Compiler
#
LLNL_CONFIRM_BABEL_F90_SUPPORT
if test "X$enable_fortran90" = "Xyes"; then #begin real F90 checking
LLNL_CONFIRM_BABEL_F90_SUPPORT2 
fi #end real F90 checking
LLNL_CONFIRM_BABEL_F90_SUPPORT3

# XML
LLNL_LIBXML_CONFIG
LLNL_PROG_TEST_EF
AC_PROG_LD

# CCA
CCA_BUNDLE_TAG

############################################################
#
# Python Support.
#
LLNL_CONFIRM_BABEL_PYTHON_SUPPORT

############################################################
#
# Critical applications.
#
AC_PROG_INSTALL
AC_PROG_CPP
AC_PROG_AWK
AC_PROG_LD
AC_PROG_LN_S
AC_PROG_INSTALL
AC_PROG_MAKE_SET

############################################################
#
# C Compiler
#
LLNL_CONFIRM_BABEL_C_SUPPORT

ACX_PTHREAD([],[])

############################################################
#
# F77 Compiler
#
LLNL_CONFIRM_BABEL_F77_SUPPORT

############################################################
#
# C++ Compiler
#
LLNL_CONFIRM_BABEL_CXX_SUPPORT
LLNL_CONFIRM_BABEL_CXX_SUPPORT2

############################################################
#
# Java Support
#
LLNL_CONFIRM_BABEL_JAVA_SUPPORT

############################################################
#
# Libtool Options
#
AC_LIBTOOL_WIN32_DLL
AC_LIBTOOL_DLOPEN
AM_PROG_LIBTOOL
AC_SUBST(LIBTOOL_DEPS)
AC_LIB_LTDL

############################################################
#
# Cygwin conditional Makefile Support.
#
case $host_os in
  *cygwin* ) CYGWIN=yes;;
         * ) CYGWIN=no;;
esac
AM_CONDITIONAL(SUPPORT_CYGWIN, test "X$CYGWIN" = "Xyes")

############################################################
#
# Shared/Dynamic Library Details.
#
LLNL_PURE_STATIC_RUNTIME
AM_CONDITIONAL(SUPPORT_STATIC, test "X$enable_static" = "Xyes")
AM_CONDITIONAL(SUPPORT_SHARED, test "X$enable_shared" = "Xyes" -a "X$XML2_CONFIG" != "Xno")
if test "X$enable_shared" != "Xyes" -o "X$XML2_CONFIG" = "Xno"; then
  AC_DEFINE(SIDL_STATIC_LIBRARY, 1, [Define SIDL_STATIC_LIBRARY to force static loading of libraries])
fi
if test "X$enable_static" != "Xyes" -a "X$XML2_CONFIG" != "Xno"; then
  AC_DEFINE(SIDL_DYNAMIC_LIBRARY, 1, [Define SIDL_DYNAMIC_LIBRARY to force dynamic loading of libraries])
fi

############################################################
#
# Experimental sidl extensions
#
if test -z "$with_sidlx"; then
  with_sidlx=yes
fi
AC_ARG_WITH(sidlx,   [  --without-sidlx         disable experimental (undocumented) libsidlx.{a,so}])
AM_CONDITIONAL(WITH_SIDLX, test "X$with_sidlx" != "Xno" )

############################################################
#
# babel_internal.h headers and footers
#
AH_TOP([
#ifndef included_babel_internal_h
#define included_babel_internal_h
])

AH_BOTTOM([
/*
 * Set flags for dynamic or static loading of implementations in Babel stubs.
 * One and only one of SIDL_STATIC_LIBRARY and SIDL_DYNAMIC_LIBRARY may be set.
 * If neither is set, then SIDL_DYNAMIC_LIBRARY is chosen as the default if
 * PIC is set and SIDL_STATIC_LIBRARY is chosen otherwise.  This behavior is
 * consistent with GNU libtool.  In general, we want to generate dynamic
 * loading with shared libraries (indicated by -DPIC in libtool) and static
 * loading with static libraries.
 */
#if (!defined(SIDL_STATIC_LIBRARY) && !defined(SIDL_DYNAMIC_LIBRARY))
#ifdef PIC
#define SIDL_DYNAMIC_LIBRARY
#else
#define SIDL_STATIC_LIBRARY
#endif
#endif

#if (defined(SIDL_STATIC_LIBRARY) && defined(SIDL_DYNAMIC_LIBRARY))
#error Cannot define both SIDL_STATIC_LIBRARY and SIDL_DYNAMIC_LIBRARY
#endif
#if (!defined(SIDL_STATIC_LIBRARY) && !defined(SIDL_DYNAMIC_LIBRARY))
#error Must define one of SIDL_STATIC_LIBRARY or SIDL_DYNAMIC_LIBRARY
#endif

/*
 * The USE_DL_IMPORT flag is required for proper Python linking under CYGWIN.
 */
#if defined(__CYGWIN__) && !defined(USE_DL_IMPORT)
#define USE_DL_IMPORT
#endif

#endif
])

############################################################
#
# File Generation
#
AC_CONFIG_FILES([Makefile
                 m4/Makefile
		 bin/Makefile
		 config/Makefile
                 java/Makefile
                 sidl/Makefile
		 sidlx/Makefile])
AC_OUTPUT
echo "$msgs
" >&5
echo "$msgs
"
