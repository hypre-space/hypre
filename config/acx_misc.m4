
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
[FLAGS are optionals to pass to insure, the default is for 
FLAG="-Zoi \"report_file insure.log\" which sends output to a file
called insure.log. To redirect output to standard error, use
-Zoi \"report_file insure.log\". Reassign cc and CC compilers
-- no checking is done to ensure insure is present on the machine]),
[case "${withval}" in
  yes) CC=insure
    CXX=insure
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCFLAGS="-Zoi \"report_file insure.log\""
    XCXXFLAGS="-Zoi \"report_file insure.log\""
    casc_user_chose_compilers=yes
    ;;
  no) casc_user_chose_compilers=no
    ;;
  *) CC=insure
    CXX=insure
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCFLAGS="$withval"
    XCXXFLAGS="$CXX $withval"
    casc_user_chose_compilers=yes
    ;;
esac])
])

dnl **********************************************************************
dnl * ACX_PURIFY
dnl *
dnl compile using purify
dnl [FLAGS are optionals to pass to insure, the default is for 
dnl FLAG="-log-file=purify.log -append-logfile=yes -best-effort" which 
dnl appends output to a file called purify.log. To redirect output to 
dnl the Viewer, use FLAGS=\"-windows=yes\". Assign cc and CC as the C 
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
[FLAGS are optionals to pass to insure, the default is for 
FLAG="-log-file=purify.log -append-logfile=yes -best-effort" which 
appends output to a file called purify.log. To redirect output to 
the Viewer, use FLAGS=\"-windows=yes\". Assign cc and CC as the C 
and C++ compilers and prepends "purify" to compile/link line-- no 
checking is done to ensure purify is present on the machine]),
[case "${withval}" in
  yes) PREPEND="purify"
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCFLAGS="-log-file=purify.log -append-logfile=yes -best-effort"
    XCXXFLAGS="-log-file=purify.log -append-logfile=yes -best-effort"
    CC="$PREPEND cc"
    CXX="$PREPEND CC"
    casc_user_chose_compilers=yes
    ;;
  no) casc_user_chose_compilers=no
    ;;
  *) PREPEND="purify"
    CFLAGS="$CFLAGS -g"
    CXXFLAGS="$CXXFLAGS -g"
    XCFLAGS="$withval"
    XCXXFLAGS="$withval"
    CC="$PREPEND cc"
    CXX="$PREPEND CC" 
    casc_user_chose_compilers=yes
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
[compiles with out MPI ('--without-MPI') and assigns KCC
as the c and c++ compilers, unless CC and CXX are already set to
gcc and g++.For C compiles KCC uses --c --strict as the compiler
flag this enforces syntax described by ISO 9899-1990, the C language
standard. Additional compiler flags, --display_error_number --lint
are enabled for lint-type checking. Individual types of warnings
can be suppressed using --diag_suppress and the error numbers
provided by --display_error_number]),
[ if test "x$GCC" = "xyes"; then
  CFLAGS="$CFLAGS -Wall -Wunused -Wmissing-prototypes"
  CFLAGS="$CFLAGS -Wmissing-declarations -ansi -pedantic"
  CXXFLAGS="$CXXFLAGS -Wall -Wno-unused -Wmissing-prototypes"
  CXXFLAGS="$CXXFLAGS -Wmissing-declarations -Wshadow"
  CXXFLAGS="$CXXFLAGS -Woverloaded-virtual -ansi -pedantic"
else
  CC=kcc
  CXX=KCC
  CFLAGS="$CFLAGS --c --strict --lint --display_error_number"
  CFLAGS="$CFLAGS --diag_suppress 45,236,450,826"
  CFLAGS="$CFLAGS,1018,1021,1022,1023,1024,1030,1041"
  CXXFLAGS="$CXXFLAGS --strict --lint --display_error_number"
  CXXFLAGS="$CXXFLAGS --diag_suppress 381,450"
fi
casc_user_chose_compilers=yes
casc_using_mpi=no])
])
dnl **********************************************************************
dnl * ACX_CHECK_MPI
dnl *
dnl try and determine what the MPI flags should be
dnl
AC_DEFUN([ACX_CHECK_MPI],
acx_mpi_ok=no

[AC_CHECK_HEADER(mpi.h,
  [AC_CHECK_LIB(mpi, MPI_Init,[acx_mpi_ok=yes;LIBS="$LIBS -lmpi"],
    [AC_MSG_WARN([* Unable to find libmpi, which is kinda necessary. *])])],
  [AC_MSG_WARN([* * * Missing "mpi.h" header file * * *])])
if test $acx_mpi_ok = no; then
  AC_CHECK_HEADER(mpi.h,
    [AC_CHECK_LIB(mpich, MPI_Init,[LIBS="$LIBS -lmpich"],
      [AC_MSG_WARN([* Unable to find libmpich, which is kinda necessary. *])])],
    [AC_MSG_WARN([* Unable to find libmpi, which is kinda necessary. *])])
fi 
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
if test "x${F77FLAGS}" = "x"
then
  if test "x${G77}" = "xyes"
  then
    F77FLAGS="-O"
  else
    case "${CXX}" in
      kf77|mpikf77)
        F77FLAGS="-fast +K3"
        ;;
      ifc)
        F77FLAGS="-O3 -xW -tpp7"
        if test "$casc_using_openmp" = "yes" ; then
          F77FLAGS="$F77FLAGS -openmp"
        fi
        ;;
      pgf77|mpipgf77)
        F77FLAGS="-fast"
        if test "$casc_using_openmp" = "yes" ; then
          F77FLAGS="$F77FLAGS -mp"
        fi
        ;;
      f77|f90|mpxlf|mpif77|mpiifc|xlf|cxx)
        case "${host}" in
          alpha*-dec-osf4.*)
            F77FLAGS="-std1 -w0 -O2"
            ;;
          alpha*-dec-osf5.*)
            F77FLAGS="-fast"
            if test "$casc_using_openmp" = "yes" ; then
              F77FLAGS="$F77FLAGS -omp"
            fi
            ;;
          mips-sgi-irix6.[[4-9]]*)
            F77FLAGS="-O -64"
            if test "$casc_using_openmp" = "yes" ; then
              F77FLAGS="$F77FLAGS -mp"
            fi
            ;;
          mips-sgi-irix*)
            F77FLAGS="-fullwarn -woff 835 -O2 -Olimit 3500"
            ;;
          rs6000-ibm-aix*)
            F77FLAGS="-D_ALL_SOURCE -O2"
            ;;
          powerpc-ibm-aix*)
            F77FLAGS="-O3 -qstrict"
            if test "$casc_using_openmp" = "yes" ; then
              F77FLAGS="$F77FLAGS -qsmp=omp"
            fi
            ;;
          sparc-sun-solaris2*)
            F77FLAGS="-silent -O"
            ;;
          *)
            F77FLAGS="-O"
            ;;
        esac
        ;;
      *)
        F77FLAGS="-O"
        ;;
    esac
  fi
fi])
      
