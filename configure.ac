#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# Set VERSION so we only need to edit in one place (i.e., here)
m4_define(HYPRE_VERSION, 1.8.3b)
m4_define(HYPRE_DATE, 2003/10/22 23:46:36)
m4_define(HYPRE_TOP_SRC_DIR, `pwd`)

# Variable Initialization
casc_user_chose_compilers=no
casc_user_chose_blas=no
casc_user_chose_lapack=no
casc_user_chose_cflags=no
casc_user_chose_cxxflags=no
casc_user_chose_fflags=no
casc_using_mpi=yes
casc_using_debug=no
casc_using_openmp=no   
casc_using_babel=no   
HYPRE_BABELLIBDIR=""
BABEL_RUNTIME_DIR=""
BABELDIR=""
#casc_using_mli=no   

AC_PREREQ(2.57)
AC_REVISION($Id$)
AC_INIT([hypre], HYPRE_VERSION, [http://www-casc.llnl.gov/bugs/enter_bug.cgi?product=HYPRE], [hypre])
AC_CONFIG_HEADER([HYPRE_config.h:config/HYPRE_config.h.in])
AC_COPYRIGHT([This work was produced at the University of California,
Lawrence Livermore National Laboratory (UC LLNL) under contract
no. W-7405-ENG-48 (Contract 48) between the U.S. Department of Energy
(DOE) and The Regents of the University of California (University) for
the operation of UC LLNL. The rights of the Federal Government are
reserved under Contract 48 subject to the restrictions agreed upon by
the DOE and University as allowed under DOE Acquisition Letter 97-1.])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_SRCDIR([HYPRE.h])
dnl Determine the host type
AC_CANONICAL_HOST
# Turn off shared libraries by default, since build process takes longer.
AC_DISABLE_SHARED
AM_CONDITIONAL([STATICONLY],[test "$enable_shared" = no])
AM_MAINTAINER_MODE

AM_INIT_AUTOMAKE([1.7.5 foreign no-installinfo no-installman no-texinfo.tex])
dnl Change the default prefix (/usr/local) to ./hypre
AC_PREFIX_DEFAULT("HYPRE_TOP_SRC_DIR/hypre")
m4_pattern_allow([HYPRE_[A-Z_]+])
DATETIME="HYPRE_DATE"
AC_SUBST(DATETIME)

if test "x$CFLAGS" != "x" ; then
  casc_user_chose_cflags=yes
fi
if test "x$CXXFLAGS" != "x" ; then
  casc_user_chose_cxxflags=yes
fi
if test "x$FFLAGS" != "x" ; then
  casc_user_chose_fflags=yes
fi

ACX_DEBUG

ACX_MPI

ACX_TIMING

ACX_OPENMP

ACX_STRICT_CHECKING

ACX_INSURE

ACX_PURIFY

ACX_ASCI_HOST

AC_ARG_WITH(babel,
AC_HELP_STRING([--with-babel],[use babel]),
[ if test "$withval" = "yes"; then
    casc_using_babel=yes
    BABEL_INSTALL_DIR=HYPRE_TOP_SRC_DIR/babel-runtime
    BABEL_HYPRE_INCLUDES="\
    -I HYPRE_TOP_SRC_DIR/hypre/include\
    -I$BABEL_INSTALL_DIR/sidl\
    -I HYPRE_TOP_SRC_DIR/babel/bHYPRE"
    BABEL_LIB_DIR="HYPRE_TOP_SRC_DIR/hypre/lib"
    BABELLIBFLAGS="\
    -L/HYPRE_TOP_SRC_DIR/babel/bHYPREClient-C -lbHYPREClient-C\
    -L/HYPRE_TOP_SRC_DIR/babel/bHYPREClient-F -lbHYPREClient-F\
    -L/HYPRE_TOP_SRC_DIR/babel/bHYPRE -lbHYPRE\
    -L${BABEL_LIB_DIR} -lsidl"
 
    babeldir="${BABEL_INSTALL_DIR}"
    AC_SUBST(babeldir)

    HYPRE_BABELLIBDIR="HYPRE_TOP_SRC_DIR/babel"
    BABEL_RUNTIME_DIR="HYPRE_TOP_SRC_DIR/babel-runtime"
  fi], [casc_using_babel=no]
)

#AC_ARG_WITH(mli,
#AC_HELP_STRING([--with-mli],[use MLI]),
#[ if test "$withval" = "yes"; then
#    MLI_DIR="HYPRE_TOP_SRC_DIR/mli/matrix"
#    MLILIBS="-lHYPRE_mli_mat"
#    MLILIBDIRS="-L$MLI_DIR"
#    MLIINCLUDE="-I$MLI_DIR"
#    MLILIBDIRS="$MLILIBDIRS -L/usr/openwin/lib"
#    MLILIBS="$MLILIBS -lX11"
#    casc_using_mli=yes
#  fi], [casc_using_mli=no]
#)

AC_ARG_WITH(FEI_BASE_DIR,
AC_HELP_STRING([--with-FEI_BASE_DIR],
[specify the directory where the fei header files live.]), 
[ if test "$withval" != ""; then
    FEI_BASE_DIR=$withval
    FEI_BASE_DIR_DEFINED="yes"
  fi], [FEI_BASE_DIR_DEFINED="no"]
)
AM_CONDITIONAL([USE_FEI_BASE_DIR_DEFAULT], [test "$FEI_BASE_DIR_DEFINED" = "no"])

AC_ARG_WITH(FEI_LIB_DIR,
AC_HELP_STRING([--with-FEI_LIB_DIR],
[specify an additional directory for fei library intalls.]), 
[ if test "$withval" != ""; then
    FEI_LIB_DIR=$withval
    FEI_LIB_DIR_DEFINED="yes"
  fi], [FEI_LIB_DIR_DEFINED="no"]
)
AM_CONDITIONAL([USE_FEI_LIB_DIR_DEFAULT], [test "$FEI_LIB_DIR_DEFINED" = "no"])

AC_ARG_WITH(FEI_INC_DIR,
AC_HELP_STRING([--with-FEI_INC_DIR],
[specify an additional directory for fei include installs.]), 
[ if test "$withval" != ""; then
    FEI_INC_DIR=$withval 
    FEI_INC_DIR_DEFINED="yes"
  fi], [FEI_INC_DIR_DEFINED="no"]
)
AM_CONDITIONAL([USE_FEI_INC_DIR_DEFAULT], [test "$FEI_INC_DIR_DEFINED" = "no"])

if test "$casc_user_chose_compilers" != "yes" ; then
  if test "$casc_using_openmp" = "yes" ; then
    AC_CHECK_PROGS(CC, mpguidec icc pgcc xlc cc)
    AC_CHECK_PROGS(CXX, mpguidec++ icc pgCC xlC CC)
    AC_CHECK_PROGS(F77, mpguidef77 ifc pgf77 xlf f77)
  elif test "$casc_using_mpi" = "no" ; then
    AC_CHECK_PROGS(CC, icc pgcc xlc cc kcc c89 gcc)
    AC_CHECK_PROGS(CXX, icc pgCC xlC c++ cxx CC KCC cl g++ gcc)
    AC_CHECK_PROGS(F77, ifc pgf77 xlf f77 kf77 fort77 fort g77)
  else
    AC_CHECK_PROGS(CC, mpcc mpikcc mpicc hcc icc pgcc xlc cc kcc c89 gcc)
    AC_CHECK_PROGS(CXX, mpCC mpiKCC mpicxx mpiCC hcp icc pgCC xlC c++ cxx CC KCC g++)
    AC_CHECK_PROGS(F77, mpxlf mpf77 mpikf77 mpif77 ifc pgf77 xlf f77 kf77 fort77 f90 g77)
  fi
fi
# Checks for programs.
AC_PROG_INSTALL
AC_PROG_AWK
AC_PROG_CPP
AC_PROG_LN_S
AC_PROG_MAKE_SET
# Checks for compilers.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_F77
AC_F77_LIBRARY_LDFLAGS
AC_F77_WRAPPERS
AC_EXEEXT

dnl Initialize libtool
##AC_LIBLTDL_CONVENIENCE
#AC_SUBST(LTDLINCL)
#AC_SUBST(LIBLTDL)
AC_LIBTOOL_DLOPEN
AC_LIBTOOL_WIN32_DLL
AC_PROG_LIBTOOL
##AC_LIBTOOL_SETUP
AC_SUBST(LIBTOOL_DEPS)

# Checks for libraries.
## FIXME: Replace `main' with a function in `-lamg':
#AC_CHECK_LIB([amg], [main])
## FIXME: Replace `main' with a function in `-lcegdb':
#AC_CHECK_LIB([cegdb], [main])
## FIXME: Replace `main' with a function in `-ldsparskit':
#AC_CHECK_LIB([dsparskit], [main])
## FIXME: Replace `main' with a function in `-ldxml':
#AC_CHECK_LIB([dxml], [main])
## FIXME: Replace `main' with a function in `-lf2c':
#AC_CHECK_LIB([f2c], [main])
## FIXME: Replace `main' with a function in `-lfei_isis':
#AC_CHECK_LIB([fei_isis], [main])
## FIXME: Replace `main' with a function in `-lgen':
#AC_CHECK_LIB([gen], [main])
## FIXME: Replace `main' with a function in `-lgs':
#AC_CHECK_LIB([gs], [main])
## FIXME: Replace `main' with a function in `-lic':
#AC_CHECK_LIB([ic], [main])
## FIXME: Replace `main' with a function in `-lilu':
#AC_CHECK_LIB([ilu], [main])
## FIXME: Replace `main' with a function in `-llapack':
#AC_CHECK_LIB([lapack], [main])
## FIXME: Replace `main' with a function in `-lm':
#AC_CHECK_LIB([m], [main])
## FIXME: Replace `main' with a function in `-lmalloc':
#AC_CHECK_LIB([malloc], [main])
## FIXME: Replace `main' with a function in `-lml':
#AC_CHECK_LIB([ml], [main])
## FIXME: Replace `main' with a function in `-lmpi':
##AC_CHECK_LIB([mpi], [MPI_Init])
## FIXME: Replace `main' with a function in `-lmpich':
##AC_CHECK_LIB([mpich], [MPI_Init])
## FIXME: Replace `main' with a function in `-lnsl':
#AC_CHECK_LIB([nsl], [main])
## FIXME: Replace `main' with a function in `-lpetscmat':
#AC_CHECK_LIB([petscmat], [main])
## FIXME: Replace `main' with a function in `-lpetscsles':
#AC_CHECK_LIB([petscsles], [main])
## FIXME: Replace `main' with a function in `-lpetscsys':
#AC_CHECK_LIB([petscsys], [main])
## FIXME: Replace `main' with a function in `-lpetscvec':
#AC_CHECK_LIB([petscvec], [main])
## FIXME: Replace `main' with a function in `-lpmpich':
#AC_CHECK_LIB([pmpich], [main])
## FIXME: Replace `main' with a function in `-lpthread':
#AC_CHECK_LIB([pthread], [main])
## FIXME: Replace `main' with a function in `-lrpc':
#AC_CHECK_LIB([rpc], [main])
## FIXME: Replace `main' with a function in `-lsocket':
#AC_CHECK_LIB([socket], [main])
## FIXME: Replace `main' with a function in `-lsunmath':
#AC_CHECK_LIB([sunmath], [main])

# Checks for header files.
#AC_PATH_X
AC_HEADER_DIRENT
AC_HEADER_STDC
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([float.h inttypes.h limits.h malloc.h memory.h netdb.h netinet/in.h stddef.h stdlib.h string.h strings.h sys/param.h sys/socket.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_C_VOLATILE
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_CLOSEDIR_VOID
AC_FUNC_ERROR_AT_LINE
#AC_FUNC_MALLOC
#AC_FUNC_REALLOC
AC_TYPE_SIGNAL
AC_FUNC_STAT
AC_FUNC_STRTOD
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([atexit bzero floor getcwd gethostbyname gethostname gethrtime gettimeofday memset pow rint socket sqrt strcasecmp strchr strcspn strdup strrchr strspn strtol])

AC_ARG_WITH(COMM_SIMPLE,
AC_HELP_STRING([--with-COMM_SIMPLE],
[Do not use MPI derived data types.  This option is automatically 
chosen for IBM, but may be selected for other platforms as well.]),
[ if test "$withval" = "yes"; then
    CPPFLAGS="$CPPFLAGS -DHYPRE_COMM_SIMPLE"
  fi]
)

if test "$casc_using_mpi" = "no" ; then
  AC_DEFINE(HYPRE_SEQUENTIAL, 1, [Disable MPI, enable serial codes])
else
  ACX_CHECK_MPI(LIBS="$LIBS $MPILIBS")
fi
case "${host}" in
  sparc-sun-solaris2*)
    AC_DEFINE(HYPRE_SOLARIS, 1, [Configured for a Sun])
    CSUPERLUFLAGS=-DAdd_
  ;;
  alpha*-*-linux* | *-*-linux* )
    AC_DEFINE(HYPRE_LINUX, 1, [Configured for Linux])
    CSUPERLUFLAGS=-DAdd_
  ;;
  alpha*-*-osf*)
    AC_DEFINE(HYPRE_ALPHA, 1, [Configured for Compacq])
    CSUPERLUFLAGS=-DAdd_
  ;;
  powerpc*-*-aix*)
    AC_DEFINE(HYPRE_RS6000, 1, [Configured for IBM])
    CSUPERLUFLAGS=-DNoChange
  ;;
  *-hp-hpux*)
    CSUPERLUFLAGS=-DNoChange
  ;;
  mips-sgi-irix5*)
    AC_DEFINE(HYPRE_IRIX64, 1, [Configured for SGI])
    CSUPERLUFLAGS=-DAdd_
  ;;
esac
AC_SUBST(CSUPERLUFLAGS)

dnl * set generic default for blas usage
ACX_BLAS([casc_user_chose_blas=yes;BLASLIBFLAGS=$BLAS_LIBS], 
  [casc_user_chose_blas=no;BLASLIBFLAGS="-lHYPRE_blas"]) 
AM_CONDITIONAL([BLASNEEDED],[test "$casc_user_chose_blas" = no])
AC_SUBST(BLASLIBFLAGS)
ACX_LAPACK([casc_user_chose_lapack=yes;LAPACKLIBFLAGS=$LAPACK_LIBS], 
  [casc_user_chose_lapack=no;LAPACKLIBFLAGS="-lHYPRE_lapack"]) 
AM_CONDITIONAL([LAPACKNEEDED],[test "$casc_user_chose_lapack" = no])
AC_SUBST(LAPACKLIBFLAGS)
AC_CHECK_LIBM
AC_SUBST(LIBM)

CCLD="--tag=CC $CC"
CXXLD="--tag=CXX $CXX"
if test "$casc_using_debug" = "yes" ; then
# deal with libtool quirks
  if test "$PREPEND" = "insure" ; then
    CCLD="--tag=CC $PREPEND $XCCFLAGS"
    CXXLD="--tag=CXX $PREPEND $XCXXFLAGS"
  elif test "$PREPEND" = "purify" ; then
    CCLD="--tag=CC $PREPEND $XCCFLAGS $CC"
    CXXLD="--tag=CXX $PREPEND $XCXXFLAGS $CXX"
  fi
else
  ACX_OPTIMIZATION_FLAGS
fi
AC_SUBST(CCLD)
AC_SUBST(CXXLD)

dnl *BABEL-related header files, libraries and directories for those
dnl * libraries
AC_SUBST(BABEL_DIR)
AC_SUBST(BABEL_HYPRE_INCLUDES)
AC_SUBST(BABELLIBFLAGS)
AC_SUBST(HYPRE_BABELLIBDIR)

dnl *MLI-related header files, libraries and directories for those
dnl * libraries
AC_SUBST(MLI_DIR)
AC_SUBST(MLILIBS)
AC_SUBST(MLILIBDIRS)
AC_SUBST(MLIINCLUDE)

dnl *FEI-helper variables for installing libraries, and includes, and
dnl * build options
AC_SUBST(FEI_BASE_DIR)
AC_SUBST(FEI_LIB_DIR)
AC_SUBST(FEI_INC_DIR)

AC_CONFIG_FILES([
  utilities/HYPRE_utilities.h
  Makefile
  FEI_mv/Makefile
  FEI_mv/SuperLU/Makefile
  FEI_mv/fei-base/Makefile
  FEI_mv/fei-hypre/Makefile
  FEI_mv/femli/Makefile
  FEI_mv/femli/amgs/Makefile
  FEI_mv/femli/base/Makefile
  FEI_mv/femli/blas/Makefile
  FEI_mv/femli/cintface/Makefile
  FEI_mv/femli/fedata/Makefile
  FEI_mv/femli/lib/Makefile
  FEI_mv/femli/mapper/Makefile
  FEI_mv/femli/matrix/Makefile
  FEI_mv/femli/solver/Makefile
  FEI_mv/femli/test/Makefile
  FEI_mv/femli/util/Makefile
  FEI_mv/femli/vector/Makefile
  IJ_mv/Makefile
  blas/Makefile
  config/Makefile
  distributed_ls/Makefile
  distributed_ls/Euclid/Makefile
  distributed_ls/ParaSails/Makefile
  distributed_ls/pilut/Makefile
  distributed_matrix/Makefile
  docs/Makefile
  krylov/Makefile
  lapack/Makefile
  lib/Makefile
  matrix_matrix/Makefile
  parcsr_es/Makefile
  parcsr_es/LOBPCG/Makefile
  parcsr_ls/Makefile
  parcsr_mv/Makefile
  seq_mv/Makefile
  sstruct_ls/Makefile
  sstruct_mv/Makefile
  struct_ls/Makefile
  struct_mv/Makefile
  test/Makefile
  test/AUTOTEST/Makefile
  test/TEST_fei/Makefile
  test/TEST_ij/Makefile
  test/TEST_sstruct/Makefile
  test/TEST_struct/Makefile
  test/TEST_timing/Makefile
  utilities/Makefile])
if test "$casc_using_babel" = "yes"; then
  if test -d babel; then
    AC_CONFIG_FILES([babel/Makefile
      babel/bHYPRE/Makefile
      babel/bHYPREClient-C/Makefile
      babel/bHYPREClient-F/Makefile])
    BABELDIR=babel
  fi
  if test -d babel-runtime; then
    export CC CXX F77 F90 CPP LD
    export CFLAGS CXXFLAGS FFLAGS F90FLAGS CPPFLAGS LDFLAGS 
    AC_CONFIG_SUBDIRS(babel-runtime)
    BABELDIR="$BABELDIR babel-runtime"
  fi
fi
AC_SUBST(BABELDIR)

#if test "$casc_using_mli" = "yes"; then
#  if test -d mli; then
#    AC_CONFIG_FILES([mli/Makefile
#      mli/matrix/Makefile])
#  fi
#fi
#if test -d parchord_ls; then
#  AC_CONFIG_FILES([parchord_ls/Makefile])
#fi
#AM_CONDITIONAL(PARCHORDLSDIR, test -d parchord_ls)
#if test -d seq_ls; then
#  AC_CONFIG_FILES([seq_ls/Makefile
#    seq_ls/amge/Makefile
#    seq_ls/amge_ag/Makefile
#    seq_ls/pamg/Makefile])
#fi
#AM_CONDITIONAL(SEQLSDIR, test -d seq_ls)
#if test -d tools; then
#  AC_CONFIG_FILES([tools/Makefile])
#fi
#AM_CONDITIONAL(TOOLSDIR, test -d tools)
AC_OUTPUT
