
dnl **********************************************************************
dnl * ACX_CONFIG_OUTPUT_LIST(DIR-LIST[, OUTPUT-FILE])
dnl *
dnl * The intent of this macro is to make configure handle the possibility
dnl * that a portion of the directory tree of a project may not be
dnl * present.  This will modify the argument list of AC_OUTPUT to contain
dnl * only output file names for which corresponding input files exist.
dnl * If you are not concerned about the possible absence of the necessary
dnl * input (.in) files, it is better to not use this macro and to
dnl * explicitly list all of the output files in a call to AC_OUTPUT.
dnl * Also, If you wish to create a file Foo from a file with a name
dnl * other than Foo.in, this macro will not work, and you must use
dnl * AC_OUTPUT.
dnl *
dnl * This macro checks for the existence of the file OUTPUT-FILE.in in
dnl * each directory specified in the whitespace-separated DIR-LIST.  
dnl * (Directories should be specified by relative path from the directory 
dnl * containing configure.in.) If OUTPUT-FILE is not specified, the
dnl * default is 'Makefile'.  For each directory that contains 
dnl * OUTPUT-FILE.in, the relative path of OUTPUT-FILE is added to the 
dnl * shell variable OUTPUT-FILE_list.  When AC_OUTPUT is called,
dnl * '$OUTPUT-FILE_list' should be included in the argument list.  So if
dnl * you have a directory tree and each subdirectory contains a 
dnl * Makefile.in, DIR-LIST should be a list of every subdirectory and
dnl * OUTPUT-FILE can be omitted, because 'Makefile' is the default.  When
dnl * configure runs, it will check for the existence of a Makefile.in in
dnl * each directory in DIR-LIST, and if so, the relative path of each
dnl * intended Makefile will be added to the variable Makefile_list.
dnl *
dnl * This macro can be called multiple times, if there are files other
dnl * than Makefile.in with a .in suffix other that are intended to be 
dnl * processed by configure. 
dnl *
dnl * Example
dnl *     If directories dir1 and dir2 both contain a file named Foo.in, 
dnl *     and you wish to use configure to create a file named Foo in each
dnl *     directory, then call 
dnl *     ACX_CONFIG_OUTPUT_LIST(dir1 dir2, Foo)
dnl *     If you also called this macro for Makefile as described above,
dnl *     you should call
dnl *     AC_OUTPUT($Makefile_list $Foo_list)
dnl *     at the end of configure.in .
dnl *********************************************************************


AC_DEFUN(ACX_CONFIG_OUTPUT_LIST,
[
   dnl * m_OUTPUT_LIST is a macro to store the name of the variable
   dnl * which will contain the list of output files
   define([m_OUTPUT_LIST], ifelse([$2], , Makefile_list, [$2_list]))

   if test -z "$srcdir"; then
      srcdir=.
   fi

   dnl * use "Makefile" if second argument not given
   if test -n "$2"; then
      casc_output_file=$2
   else   
      casc_output_file=Makefile
   fi   
      
   dnl * Add a file to the output list if its ".in" file exists.
   for casc_dir in $1; do
      if test -f $srcdir/$casc_dir/$casc_output_file.in; then
         m_OUTPUT_LIST="$m_OUTPUT_LIST $casc_dir/$casc_output_file"
      fi
   done
])dnl

dnl **********************************************************************
dnl * ACX_INSURE
dnl *
dnl compile using insure++
dnl FLAGS are optionals to pass to insure, the default is for 
dnl FLAG="-Zoi "report_file insure.log" which sends output to a file
dnl called insure.log. To redirect output to standard error, use
dnl -Zoi "report_file insure.log". Reassign cc and CC compilers
dnl -- no checking is done to ensure insure is present on the machine.
dnl 
dnl Note:
dnl because of the configure testing of the compiler (and CFLAGS)
dnl the FLAGS needs to be set late in the configure process.
dnl
AC_DEFUN([ACX_INSURE],
[AC_ARG_WITH([insure],
AC_HELP_STRING(
[--with-insure=FLAGS],
[FLAGS are optionals to pass to insure. To redirect output to a
file; FLAG="-Zoi "report_file insure.log". Its probably best
to supply options to insure through the .psrc file.
-- no checking is done to ensure insure is present on the machine]),
[case "${withval}" in
  yes) PREPEND=insure
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    casc_using_debug=yes
    casc_using_mpi=no
    ;;
  no) casc_user_chose_compilers=no
    ;;
  *) PREPEND=insure
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCCFLAGS="$withval"
    XCXXFLAGS="$withval"
    casc_using_debug=yes
    casc_using_mpi=no
    ;;
esac])

#AC_ARG_WITH([insure-to-file],
#AC_HELP_STRING(
#[--with-insure-to-file],
#[direct insure output to the file "insure.log"]),
#[case "${withval}" in
#  yes) PREPEND=insure
#    CFLAGS="$CFLAGS -g"
#    CXXFLAGS="$CXXFLAGS -g"
#    XCCFLAGS="-Zoi \"report_file insure.log\""
#    XCXXFLAGS="-Zoi \"report_file insure.log\""
#    casc_using_debug=yes
#    casc_using_mpi=no
#    ;;
#  no) casc_user_chose_compilers=no
#    ;;
#  *) PREPEND=insure
#    CFLAGS="$CFLAGS -g"
#    CXXFLAGS="$CXXFLAGS -g"
#    XCCFLAGS="-Zoi \"report_file insure.log\""
#    XCXXFLAGS="-Zoi \"report_file insure.log\""
#    casc_using_debug=yes
#    casc_using_mpi=no
#    ;;
#esac])
])

dnl **********************************************************************
dnl * ACX_PURIFY
dnl *
dnl compile using purify
dnl [FLAGS are optionals to pass to insure, the default is for 
dnl FLAG="-log-file=purify.log -append-logfile=yes -best-effort" which 
dnl appends output to a file called purify.log. To redirect output to 
dnl the Viewer, use FLAGS="-windows=yes". Assign cc and CC as the C 
dnl and C++ compilers and prepends "purify" to compile/link line-- no 
dnl checking is done to ensure purify is present on the machine]),
dnl 
dnl Note:
dnl because of the configure testing of the compiler (and CFLAGS)
dnl the FLAGS needs to be set late in the configure process.
dnl Also,
dnl libtool has problems with purify (and the log-file), needing
dnl a --tag=CC option, still have a problem with the ld phase
dnl
AC_DEFUN([ACX_PURIFY],
[AC_ARG_WITH([purify],
AC_HELP_STRING(
[--with-purify=FLAGS],
[FLAGS are optionals to pass to insure. To appends output to a file
use FLAG="-log-file=purify.log -append-logfile=yes -best-effort".
To redirect output to the Viewer, use FLAGS="-windows=yes". 
-- no checking is done to ensure purify is present on the machine]),
[case "${withval}" in
  yes) PREPEND="purify"
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    casc_using_debug=yes
    casc_using_mpi=no
    ;;
  no) casc_user_chose_compilers=no
    ;;
  *) PREPEND="purify"
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCCFLAGS="$withval"
    XCXXFLAGS="$withval"
    casc_using_debug=yes
    casc_using_mpi=no
    ;;
esac])
AC_ARG_WITH([purify-to-file],
AC_HELP_STRING(
[--with-purify-to-file],
[direct purify output to the file "purify.log"]),
[case "${withval}" in
  yes) PREPEND="purify"
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCCFLAGS="-log-file=purify.log -append-logfile=yes -best-effort"
    XCXXFLAGS="-log-file=purify.log -append-logfile=yes -best-effort"
    casc_using_debug=yes
    casc_using_mpi=no
    ;;
  no) casc_user_chose_compilers=no
    ;;
  *) PREPEND="purify"
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCCFLAGS="-log-file=purify.log -append-logfile=yes -best-effort"
    XCXXFLAGS="-log-file=purify.log -append-logfile=yes -best-effort"
    casc_using_debug=yes
    casc_using_mpi=no
    ;;
esac])
])
dnl **********************************************************************
dnl * ACX_STRICT_CHECKING
dnl *
dnl compile using strict ansi checking
dnl compiles with out MPI ('--without-MPI') and assigns KCC
dnl as the c and c++ compilers, unless CC and CXX are already set to
dnl gcc and g++.For C compiles KCC uses --c --strict as the compiler
dnl flag this enforces syntax described by ISO 9899-1990, the C language
dnl standard. Additional compiler flags, --display_error_number --lint
dnl are enabled for lint-type checking. Individual types of warnings
dnl can be suppressed using --diag_suppress and the error numbers
dnl provided by --display_error_number
dnl
AC_DEFUN([ACX_STRICT_CHECKING],
[AC_ARG_WITH([strict-checking],
AC_HELP_STRING(
[--with-strict-checking],
[Try to find a compiler option that warns when a function prototype
is not fully defined, as many other non-ANSI features as possible.
Currently this macro knows about GCC, and KCC.]),
[ 
  AC_CHECK_PROGS(CC, icc pgcc xlc gcc kcc KCC)
  AC_CHECK_PROGS(CXX, icc pgCC xlC g++ gcc KCC)
  AC_CHECK_PROGS(F77, ifc pgf77 xlf g77)
  if test "x$GCC" = "xyes"; then
    CFLAGS="$CFLAGS -Wall -pedantic"
    CXXFLAGS="$CXXFLAGS -Wall -Wshadow -fno-implicit-templates"
    CXXFLAGS="$CXXFLAGS -Woverloaded-virtual -ansi -pedantic"
    casc_using_debug=yes
    casc_using_mpi=no
  elif test "x$CC" = "xicc"; then
    CFLAGS="$CFLAGS -Xc -Wall -x c"
    CXXFLAGS="$CXXFLAGS -Xc -Wall -x c++"
    casc_using_debug=yes
    casc_using_mpi=no
  elif test "x$CC" = "xpgcc"; then
    CFLAGS="$CFLAGS -Xa -Minform,inform"
    CXXFLAGS="$CXXFLAGS -A --display_error_number -Minform,inform"
    casc_using_debug=yes
    casc_using_mpi=no
  elif test "x$CC" = "xxlc"; then
    CFLAGS="$CFLAGS -qinfo=dcl:eff:pro:rea:ret:use"
    CXXFLAGS="$CXXFLAGS  -qinfo=dcl:eff:obs:pro:rea:ret:use"
    casc_using_debug=yes
    casc_using_mpi=no
  elif test "x$CC" = "xKCC" || test "x$CC" = "xkcc"; then
    CFLAGS="$CFLAGS --c --strict --lint --display_error_number"
    CFLAGS="$CFLAGS --diag_suppress 45,236,450,826"
    CFLAGS="$CFLAGS,1018,1021,1022,1023,1024,1030,1041"
    CXXFLAGS="$CXXFLAGS --strict --lint --display_error_number"
    CXXFLAGS="$CXXFLAGS --diag_suppress 381,450,1023,1024"
    casc_using_debug=yes
    casc_using_mpi=no
  else
    CC=gcc
    CXX=g++
    CFLAGS="$CFLAGS -Wall -pedantic"
    CXXFLAGS="$CXXFLAGS -Wall -Wshadow -fno-implicit-templates"
    CXXFLAGS="$CXXFLAGS -Woverloaded-virtual -ansi -pedantic"
    casc_user_chose_compilers=yes
    casc_using_mpi=no
  fi
])])
dnl **********************************************************************
dnl * ACX_CHECK_MPI
dnl *
dnl try and determine what the MPI flags should be
dnl ACX_CHECK_MPI([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl ACTION-IF-FOUND is a list of shell commands to run 
dnl   if an MPI library is found, and
dnl ACTION-IF-NOT-FOUND is a list of commands to run it 
dnl   if it is not found. If ACTION-IF-FOUND is not specified, 
dnl   the default action will define HAVE_MPI. 

dnl
AC_DEFUN([ACX_CHECK_MPI],[
AC_PREREQ(2.50) dnl for AC_LANG_CASE

if test x = x"$MPILIBS"; then
  AC_LANG_CASE([C], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
    [C++], [AC_CHECK_FUNC(MPI_Init, [MPILIBS=" "])],
    [Fortran 77], [AC_MSG_CHECKING([for MPI_Init])
      AC_TRY_LINK([],[      call MPI_Init], [MPILIBS=" "
        AC_MSG_RESULT(yes)], [AC_MSG_RESULT(no)])])
fi
if test x = x"$MPILIBS"; then
  AC_CHECK_LIB(mpi, MPI_Init, [MPILIBS="-lmpi"])
fi
if test x = x"$MPILIBS"; then
  AC_CHECK_LIB(mpich, MPI_Init, [MPILIBS="-lmpich"])
fi

dnl We have to use AC_TRY_COMPILE and not AC_CHECK_HEADER because the
dnl latter uses $CPP, not $CC (which may be mpicc).
AC_LANG_CASE([C], [if test x != x"$MPILIBS"; then
  AC_MSG_CHECKING([for mpi.h])
  AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [MPILIBS=""
                     AC_MSG_RESULT(no)])
fi],
[C++], [if test x != x"$MPILIBS"; then
  AC_MSG_CHECKING([for mpi.h])
  AC_TRY_COMPILE([#include <mpi.h>],[],[AC_MSG_RESULT(yes)], [MPILIBS=""
                     AC_MSG_RESULT(no)])
fi])

AC_SUBST(MPILIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x = x"$MPILIBS"; then
  $2
  :
else
  ifelse([$1],,[AC_DEFINE(HAVE_MPI,1,[Define if you have the MPI library.])],[$1])
  :
fi
])

ACX_MPI
dnl **********************************************************************
dnl * ACX_MPI
dnl *
dnl Compiles with MPI [default].
dnl
AC_DEFUN([ACX_MPI],
[AC_ARG_WITH(MPI,
AC_HELP_STRING([--with-MPI=DIR],
[Compiles with MPI [default]. DIR is the top-level install directory
for MPI.  Selecting --without-MPI may affect which compiler is chosen]),
[case "${withval}" in 
  yes) casc_using_mpi=yes ;;
  no) casc_using_mpi=no ;;
  *) casc_using_mpi=yes
    if test -d "${withval}" ; then
      if test -d "${withval}/include" ; then
        MPIINCLUDE=${withval}/include
        CPPFLAGS="$CPPFLAGS -I${withval}/include"
      elif test -d "${withval}/h" ; then
        MPIINCLUDE=${withval}/h
        CPPFLAGS="$CPPFLAGS -I${withval}/h"
      else
        AC_MSG_ERROR(DIR=${withval}\/include does not exist for --with-MPI)
      fi
      if test -d "${withval}/lib" ; then
        if test -r "${withval}/lib/libmpi.a" ; then
          MPILIBS=mpi
          MPILIBDIRS=${withval}/lib
          LIBS="$LIBS -lmpi"
          LIBDIRS="$LIBDIRS -L${withval}/lib"
        elif test -r "${withval}/lib/libmpich.a" ; then
          MPILIBS=mpich
          MPILIBDIRS=${withval}/lib
          LIBS="$LIBS -lmpich"
          LIBDIRS="$LIBDIRS -L${withval}/lib"
        else
          AC_MSG_ERROR(DIR=${withval}\/lib/libmpi.a or libmpich.a does not exist for --with-MPI)
        fi
      else
        AC_MSG_ERROR(DIR=${withval}\/lib does not exist for --with-MPI)
      fi
    else
      AC_MSG_ERROR(DIR=${withval} does not exist for --with-MPI)
    fi ;;
esac],[casc_using_mpi=yes])
])

dnl **********************************************************************
dnl * ACX_DEBUG
dnl *
dnl compile for debugging
dnl
AC_DEFUN([ACX_DEBUG],
[AC_ARG_ENABLE(debug,
AC_HELP_STRING([--enable-debug], [compile for debugging.]),
[case "${enableval}" in
  yes) casc_using_debug=yes
    CFLAGS="-g $CFLAGS"
    CXXFLAGS="-g $CXXFLAGS"
    FFLAGS="-g $FFLAGS" ;;
  no)  casc_using_debug=no;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-debug) ;;
esac],[casc_using_debug=no])
])
dnl **********************************************************************
dnl * ACX_TIMING
dnl *
dnl determine timing routines to use
dnl
AC_DEFUN([ACX_TIMING],
[AC_ARG_WITH(timing,
AC_HELP_STRING([--with-timing],[use HYPRE timing routines]),
[if test "$withval" = "yes"; then
  AC_DEFINE(HYPRE_TIMING,1,[HYPRE timing routines are being used])
fi])
])

dnl **********************************************************************
dnl * ACX_OPENMP
dnl *
dnl compile with OpenMP
dnl
AC_DEFUN([ACX_OPENMP],
[AC_ARG_WITH(openmp,
AC_HELP_STRING([--with-openmp],
[use openMP--this may affect which compiler is chosen.
Supported using guidec on IBM and Compaq.]),
[case "${withval}" in
  yes) casc_using_openmp=yes
    AC_DEFINE([HYPRE_USING_OPENMP], 1, [Enable OpenMP support]) ;;
  no)  casc_using_openmp=no;;
  *) AC_MSG_ERROR(bad value ${withval} for --with-openmp) ;;
esac],[casc_using_openmp=no])
])

dnl **********************************************************************
dnl * ACX_OPTIMIZATION_FLAGS
dnl *
dnl try and determine what the optimized compile FLAGS
dnl
AC_DEFUN([ACX_OPTIMIZATION_FLAGS],
[if test "x${CFLAGS}" = "x"
then
  if test "x${GCC}" = "xyes"
  then
    dnl **** default settings for gcc
    CFLAGS="-O2"
##  CFLAGS="$CFLAGS -fno-common -Wall -pedantic -Wpointer-arith -Wnested-externs"
##  dnl **** check for strength-reduce bug
##  ACX_GCC_STRENGTH_REDUCE(CFLAGS="$CFLAGS -fno-strength-reduce")
  
##  dnl **** some arch-specific optimizations/settings for gcc
##  case "${host}" in
##    i486-*) CPU_FLAGS="-m486";;
##    i586-*) ACX_CHECK_CC_FLAGS(-mcpu=pentium,cpu_pentium,
##               [CPU_FLAGS=-mcpu=pentium],
##               [ACX_CHECK_CC_FLAGS(-mpentium,pentium,
##                       [CPU_FLAGS=-mpentium], [CPU_FLAGS=-m486])])
##            ;;
##    i686-*) ACX_CHECK_CC_FLAGS(-mcpu=pentiumpro,cpu_pentiumpro,
##               [CPU_FLAGS=-mcpu=pentiumpro],
##               [ACX_CHECK_CC_FLAGS(-mpentiumpro,pentiumpro,
##                       [CPU_FLAGS=-mpentiumpro], [CPU_FLAGS=-m486])])
##            ;;
##  esac
    
##  CFLAGS="$CPU_FLAGS $CFLAGS"
  else
    case "${CC}" in
      kcc|mpikcc)
        CFLAGS="-fast +K3"
        ;;
      KCC|mpiKCC)
        CFLAGS="--c -fast +K3"
        ;;
      icc)
        CFLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -openmp"
        fi
        ;;
      pgcc|mpipgcc)
        CFLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          CFLAGS="$CFLAGS -mp"
        fi
        ;;
      cc|c89|mpcc|mpiicc|xlc|ccc)
        case "${host}" in
          alpha*-dec-osf4.*)
            CFLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            CFLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -omp"
            fi
            ;;
          hppa*-hp-hpux*)
            CFLAGS="-Aa -D_HPUX_SOURCE -O"
            ;;
          mips-sgi-irix6.[[4-9]]*)
            CFLAGS="-O -64 -OPT:Olimit=0"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            CFLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            CFLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            CFLAGS="-O3 -qstrict -qmaxmem=8192"
            if test "$casc_using_openmp" = "yes" ; then
              CFLAGS="$CFLAGS -qsmp=omp"
            fi
            ;;
          *)
            CFLAGS="-O"
            ;;
        esac
        ;;
      *)
        CFLAGS="-O"
        ;;
    esac
  fi
fi
if test "x${CXXFLAGS}" = "x"
then
  if test "x${GXX}" = "xyes"
  then
    dnl **** default settings for gcc
    CXXFLAGS="-O2"
  else
    case "${CXX}" in
      KCC|mpiKCC)
        CXXFLAGS="-fast +K3"
        ;;
      icc)
        CXXFLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -openmp"
        fi
        ;;
      pgCC|mpipgCC)
        CXXFLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          CXXFLAGS="$CXXFLAGS -mp"
        fi
        ;;
      CC|aCC|mpCC|mpiicc|xlC|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            CXXFLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            CXXFLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -omp"
            fi
            ;;
          hppa*-hp-hpux*)
            CXXFLAGS="-D_HPUX_SOURCE -O"
            ;;
          mips-sgi-irix6.[[4-9]]*)
            CXXFLAGS="-O -64"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            CXXFLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            CXXFLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            CXXFLAGS="-O3 -qstrict -qmaxmem=8192"
            if test "$casc_using_openmp" = "yes" ; then
              CXXFLAGS="$CXXFLAGS -qsmp=omp"
            fi
            ;;
          *)
            CXXFLAGS="-O"
            ;;
        esac
        ;;
      *)
        CXXFLAGS="-O"
        ;;
    esac
  fi
fi
if test "x${FFLAGS}" = "x"
then
  if test "x${G77}" = "xyes"
  then
    FFLAGS="-O"
  else
    case "${CXX}" in
      kf77|mpikf77)
        FFLAGS="-fast +K3"
        ;;
      ifc)
        FFLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -openmp"
        fi
        ;;
      pgf77|mpipgf77)
        FFLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          FFLAGS="$FFLAGS -mp"
        fi
        ;;
      f77|f90|mpxlf|mpif77|mpiifc|xlf|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            FFLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            FFLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -omp"
            fi
            ;;
          mips-sgi-irix6.[[4-9]]*)
            FFLAGS="-O -64"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            FFLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            FFLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            FFLAGS="-O3 -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              FFLAGS="$FFLAGS -qsmp=omp"
            fi
            ;;
          sparc-sun-solaris2*)
            FFLAGS="-silent -O"
            ;;
          *)
            FFLAGS="-O"
            ;;
        esac
        ;;
      *)
        FFLAGS="-O"
        ;;
    esac
  fi
fi])
      

dnl **********************************************************************
dnl * ACX_ASCI_HOST
dnl *
dnl determine certain compiler options for a few ASCI customers
dnl
AC_DEFUN([ACX_ASCI_HOST],
[HOSTNAME="`hostname`"
  if test -z "$HOSTNAME"
  then
    HOSTNAME=unknown
  else
    case $HOSTNAME in
      sasn100 )
        AC_CHECK_PROGS(CC, cicc cc)
        AC_CHECK_PROGS(CXX, ciCC CC) 
        AC_CHECK_PROGS(F77, cif77 f77)
        if  test "$casc_using_mpi" = "yes"
        then
          CFLAGS="$CFLAGS -I/opt/intel/tflop/current/tflops/cougar/include"
          CXXFLAGS="$CXXFLAGS -I/opt/intel/tflop/current/tflops/cougar/include"
          LDFLAGS="$LDFLAGS -L/opt/intel/tflop/current/tflops/cougar/lib/puma -lmpi"
        fi
      ;;
      janus )
        AC_CHECK_PROGS(CC, pgcc cc)
        AC_CHECK_PROGS(CXX, pgCC CC) 
        AC_CHECK_PROGS(F77, pgf77 f77)
        if  test "$casc_using_mpi" = "yes"
        then
          CFLAGS="$CFLAGS -I/cougar/include"
          CXXFLAGS="$CXXFLAGS -I/cougar/include"
          LDFLAGS="$LDFLAGS -L/cougar/lib/puma -lmpi"
        fi
        CFLAGS="$CFLAGS -cougar -mp -Mx,123,0x2000"
        CXXFLAGS="$CXXFLAGS -cougar -mp -Mx,123,0x2000"
        LDFLAGS="$LDFLAGS -cougar -mp -Mx,123,0x2000"
      ;;
      nirvana )
        AC_CHECK_PROGS(CC, cc kcc gcc)
        AC_CHECK_PROGS(CXX, CC KCC g++) 
        AC_CHECK_PROGS(F77, f77 g77)
        if  test "$casc_using_mpi" = "yes"
        then
          LDFLAGS="$LDFLAGS -L/cougar/lib/puma -lmpi"
        fi
      ;;
      n02 )
        AC_CHECK_PROGS(CC, cc kcc gcc)
        AC_CHECK_PROGS(CXX, CC KCC g++) 
        AC_CHECK_PROGS(F77, f77 g77)
        if  test "$casc_using_mpi" = "yes"
        then
          LDFLAGS="$LDFLAGS -L/usr/lib -lmpi"
        fi
        CFLAGS="$CFLAGS -64"
        CXXFLAGS="$CXXFLAGS -64"
        FFLAGS="$FFLAGS -64"
        LDFLAGS="$LDFLAGS -64"
      ;;
      tckk*|tc2k* )
        if test "$casc_using_openmp" = "no" ; then
          AC_CHECK_PROGS(CC,cc kcc gcc c89) 
          AC_CHECK_PROGS(CXX,cxx KCC g++)
          AC_CHECK_PROGS(F77,f77 kf77 g77)
        else
          AC_CHECK_PROGS(CC,guidec) 
          AC_CHECK_PROGS(CXX,guidec++)
          AC_CHECK_PROGS(F77,guidef77)
        fi
        if  test "$casc_using_mpi" = "yes"
        then
          CFLAGS="$CFLAGS -I/usr/include"
          CXXFLAGS="$CXXFLAGS -I/usr/include"
          LDFLAGS="$LDFLAGS -L/usr/lib -lmpi"
        fi
      ;;
    esac 
  fi
])
