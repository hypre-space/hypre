
dnl ******************************************************************
dnl * CASC_PROG_F77 searches the PATH for an available Fortran 77
dnl * compiler. It assigns the name of the compiler to F77.
dnl ******************************************************************

AC_DEFUN(CASC_PROG_F77,
[
dnl   AC_BEFORE([$0], [AC_PROG_CPP])dnl
   AC_CHECK_PROGS(F77, f77 xlf cf77 if77 nf77)
   test -z "$F77" && AC_MSG_ERROR([no acceptable f77 found in \$PATH])
   FFLAGS="-g -O"
   AC_SUBST(FFLAGS)
])dnl


dnl ********************************************************************
dnl * CASC_PROG_MPICC searches the PATH for an available MPI C compiler
dnl * wraparound.  It assigns the name to MPICC.
dnl ********************************************************************

AC_DEFUN(CASC_PROG_MPICC,
[
   AC_CHECK_PROGS(MPICC, mpicc mpcc tmcc hcc)
   test -z "$MPICC" && AC_MSG_ERROR([no acceptable mpicc found in \$PATH])
])dnl


dnl ********************************************************************
dnl * CASC_PROG_MPICXX searches the PATH for an available MPI C++ 
dnl * compiler wraparound.  It assigns the name to MPICXX.
dnl ********************************************************************

AC_DEFUN(CASC_PROG_MPICXX,
[
   AC_CHECK_PROGS(MPICXX, mpCC mpig++ mpiCC hcp)
   test -z "$MPICXX" && AC_MSG_ERROR([no acceptable mpic++ found in \$PATH])
])dnl


dnl **********************************************************************
dnl * CASC_PROG_MPIF77 searches the PATH for an available MPI Fortran 77 
dnl * compiler wraparound.  It assigns the name to MPIF77.
dnl **********************************************************************

AC_DEFUN(CASC_PROG_MPIF77,
[
   AC_CHECK_PROGS(MPIF77, mpif77 mpf77 mpixlf mpxlf tmf77 hf77)
   test -z "$MPIF77" && AC_MSG_ERROR([no acceptable mpif77 found in \$PATH])
])dnl


dnl **********************************************************************
dnl * CASC_PROG_FPP searches the PATH for a preprocessor for Fortran files
dnl * with preprocessor directives
dnl **********************************************************************

AC_DEFUN(CASC_PROG_FPP,
[
   AC_CHECK_PROGS(FPP, fpp cpp "$CC -E" "cc -E" "gcc -E")
   test -z "$FPP" && AC_MSG_ERROR([no acceptable fpp found in \$PATH])
])dnl


dnl **********************************************************************
dnl * CASC_CHECK_F77_PP checks whether the preprocessor needs to
dnl * be called before calling the compiler for Fortran files with
dnl * preprocessor directives.  If the preprocessor is necessary,
dnl * F77NEEDSPP is set to "yes", otherwise it is set to "no"
dnl **********************************************************************

AC_DEFUN(CASC_CHECK_F77_PP,
[
   AC_REQUIRE([CASC_PROG_F77])

   rm -f testpp.o

   AC_MSG_CHECKING(whether $FPP needs to be called before $F77)

   cat > testpp.F << EOF
#define FOO 3
	program testpp
	integer num
        integer sum
        num = FOO
#ifdef FOO
        sum = num + num
#else
        sum = num + num + num
#endif
        end 
EOF

   $F77 -DBAR -c testpp.F 
   if test -f testpp.o; then 
      F77NEEDSPP=no 
   else 
      F77NEEDSPP=yes 
   fi

   echo $F77NEEDSPP
   rm -f testpp.o testpp.F

   AC_SUBST(F77NEEDSPP)
])dnl


dnl ***********************************************************************
dnl * CASC_CHECK_MPIF77_PP checks whether the preprocessor needs to
dnl * be called before calling the compiler for Fortran files with
dnl * preprocessor directives and MPI function calls.  If the preprocessor
dnl * is necessary, MPIF77NEEDSPP is set to "yes", otherwise it is set to
dnl * "no"
dnl ***********************************************************************

AC_DEFUN(CASC_CHECK_MPIF77_PP,
[
   AC_REQUIRE([CASC_PROG_MPIF77])

   rm -f testppmp.o

   AC_MSG_CHECKING(whether $FPP needs to be called before $MPIF77)

   cat > testppmp.F <<EOF
#define FOO 3
	program testppmp
	include 'mpif.h'
	integer rank,size,mpierr,sum
	call MPI_INIT(mpierr)
	call MPI_COMM_SIZE(MPI_COMM_WORLD,size,mpierr)
	call MPI_COMM_RANK(MPI_COMM_WORLD,rank,mpierr)
#ifdef FORTRAN_NO_UNDERSCORE
        sum = rank + size
#else
        sum = rank + rank
#endif
        call MPI_FINALIZE(mpierr)
        end 
EOF

   $MPIF77 -DBAR -c testppmp.F 
   if test -f testppmp.o; then 
      MPIF77NEEDSPP=no 
   else 
      MPIF77NEEDSPP=yes 
   fi

   echo $MPIF77NEEDSPP
   rm -f testppmp.o testppmp.F
   AC_SUBST(MPIF77NEEDSPP)
])dnl


dnl **********************************************************************
dnl * CASC_CHECK_LIB_FORTRAN(LIBRARY, [, ACTION-IF-FOUND [,
dnl *                        ACTION-IF-NOT-FOUND [, OTHER-LIBRARIEs]]])
dnl * 
dnl * Checks whether LIBRARY can be used to link a sample C function
dnl * that contains a call to a sample Fortran 77 function. If linking
dnl * is successful, ACTION-IF-FOUND is executed, otherwise
dnl * ACTION-IF-NOT-FOUND is executed.  The default for ACTION-IF-FOUND is
dnl * to add -lLIBRARY to $LIBS.  The default for ACTION-IF-NOT-FOUND is
dnl * nothing.  OTHER-LIBRARIES can include the full names of libraries in
dnl * the current directory, -l flags specifying other libraries, -L tags
dnl * specifying the location of libraries (This macro may not check the
dnl * same lib directories as would be checked by the linker by default on
dnl * the command line.), and/or the names of object files, all separated
dnl * by a space, whatever might be necessary for successful linkage.
dnl **********************************************************************

AC_DEFUN(CASC_CHECK_LIB_FORTRAN,
[
   AC_REQUIRE([CASC_PROG_F77])
   AC_REQUIRE([PAC_GET_FORTNAMES])

   if test -z "$FORTRANNAMES" ; then
      NAMESTYLE="FORTRANNOUNDERSCORE"
   else
      NAMESTYLE=$FORTRANNAMES
   fi

   cat > testflib_.f << EOF
        subroutine testflib(i)
        integer i
        print *, "This tests which libraries work"
        return
        end
EOF

   $F77 -c testflib_.f

   case $NAMESTYLE in
      FORTRANDOUBLEUNDERSCORE)
         THIS_FUNCTION=testflib_;;

      FORTRANUNDERSCORE)
         THIS_FUNCTION=testflib_;;

      FORTRANCAPS)
         THIS_FUNCTION=TESTFLIB;;

      FORTRANNOUNDERSCORE)
         THIS_FUNCTION=testflib;;

   esac

   CASC_CHECK_LIB($1, $THIS_FUNCTION, $2, $3, testflib_.o $4)

   rm -f testflib_.o testflib_.f

])dnl


dnl *********************************************************************
dnl * CASC_ADD_LIB(LIBRARY, FUNCTION, DIRECTORY-LIST[, PREFIX[, 
dnl *              ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]]])
dnl * checks first if LIBRARY is available on the linking search path and
dnl * if FUNCTION can be linked with LIBRARY.  If so, -lLIBRARY is added
dnl * to the variable [PREFIX]LIBS. (i.e., if prefix is LD, -llibrary is
dnl * added to LDLIBS.)  If not, checks whitespace-separated
dnl * DIRECTORY-LIST to see if LIBRARY exists in a specified directory and
dnl * can be linked with FUNCTION.  If so, the first directory where
dnl * linking is successful is added to the front of [PREFIX]LIBDIRS, and
dnl * -lLIBRARY is added to the end of [PREFIX]LIBS.  If no prefix is
dnl * specified, the directories and libraries are added to LIBS and
dnl * LIBDIRS, respectively.  If the order of -l flags on the linking
dnl * lines is important, CASC_ADD_LIB should be called for each library
dnl * in the order they should appear on linking lines.  Mere existence of
dnl * LIBRARY in the search path or in a specified directory can usually
dnl * be determined by entering 'main' for FUNCTION.  Optional argument
dnl * ACTION-IF-FOUND contains additional instructions to execute as soon
dnl * as LIBRARY is found in any directory.  Optional argument
dnl * ACTION-IF-NOT-FOUND contains instructions to execute if LIBRARY is
dnl * not found anywhere.
dnl **********************************************************************

AC_DEFUN(CASC_ADD_LIB,
[
   define([m_THESE_LIBS],[$4LIBS])
   define([m_THESE_LIBDIRS],[$4LIBDIRS])
   CASC_CHECK_LIB($1, $2, m_THESE_LIBS="$m_THESE_LIBS -l$1"
                          casc_lib_found=yes 
                          ifelse([$5], , , [$5]),

      dnl * If library not found
      for casc_lib_dir in $3; do

         CASC_CHECK_LIB($1, $2, 
            m_THESE_LIBDIRS="-L$casc_lib_dir $m_THESE_LIBDIRS"
            m_THESE_LIBS="$m_THESE_LIBS -l$1"
            casc_lib_found=yes
            ifelse([$5], , , [$5])
            break
            , ,
            -L$casc_lib_dir $m_THESE_LIBDIRS $m_THESE_LIBS -l$1, no)
      done
      , $m_THESE_LIBDIRS $m_THESE_LIBS, no)  dnl * last two arguments for
                                             dnl * first check

   ifelse([$6], , ,
      if test "$casc_lib_found" != "yes"; then
         [$6]
      fi
   )

   unset casc_lib_found

   undefine([m_THESE_LIBS])
   undefine([m_THESE_LIBDIRS])

])dnl


dnl ********************************************************************* 
dnl * CASC_SET_F77LIBS sets the necessary library flags for linking C and
dnl * Fortran 77 codes with the C linker.  The necessary -l flags are put 
dnl * into the variable F77LIBS, and the necessary -L flags are put into  
dnl * the variable F77LIBDIRS.  This macro first checks to see if the
dnl * shell variable $F77LIBS is already set.  If so, the preset value is
dnl * used.  Otherwise this macro only works for known architectures.
dnl *********************************************************************

AC_DEFUN(CASC_SET_F77LIBS,
[
   AC_REQUIRE([CASC_GUESS_ARCH])
   AC_REQUIRE([CASC_PROG_F77])

   if test -z "$casc_f77_libs"; then
      case $ARCH in 
         sun4 | solaris)
            case $F77 in
               *g77)
                   CASC_CHECK_LIB_FORTRAN(f2c,
                                          F77LIBDIRS="-L/home/casc/g77/lib"
                                          F77LIBS="-lf2c"
                                          , ,
                                          -L/home/casc/g77/lib -lf2c);;
               *)
                  CASC_CHECK_LIB_FORTRAN(sunmath,
                                   F77LIBDIRS="-L/opt/SUNWspro/SC4.2/lib"
                                   F77LIBS="-lF77 -lsunmath"
                                   , ,
                                   -L/opt/SUNWspro/SC4.2/lib -lF77 -lsunmath);;
               esac
         ;;
         alpha)
            CASC_CHECK_LIB_FORTRAN(for, F77LIBS="-lfor", , );;

         rs6000)
            CASC_CHECK_LIB_FORTRAN(xlf90, F77LIBS="-lxlf90", , );;

         IRIX64 | iris4d)
            CASC_CHECK_LIB_FORTRAN(I77, 
                                  F77LIBS="-lF77 -lU77 -lI77 -lisam", ,
                                  -lF77 -lU77 -lI77 -lisam);;

         *)
            AC_MSG_WARN(
[unable to set F77LIBFLAGS.  They must be set as a shell variable or
 with a command-line option])
         ;;             

      esac

   else
      if test -n "$casc_f77_lib_dirs"; then
         for casc_lib_dir in $casc_f77_lib_dirs; do
            F77LIBDIRS="-L$casc_lib_dir $F77LIBDIRS"       
         done
      fi

      for casc_lib in $casc_f77_libs; do
         F77LIBS="$F77LIBS -l$casc_lib"
      done
   fi

   F77LIBFLAGS="$F77LIBDIRS $F77LIBS"
])dnl


dnl *********************************************************************
dnl * CASC_FIND_F77LIBS may be a replacement for CASC_SET_F77LIBS.  This
dnl * macro automatically finds the flags necessary to located the 
dnl * libraries needed for a Fortran/C interface.  It is more robust than
dnl * CASC_SET_F77LIBS, because it is not based on the architecture name.
dnl * The test is performed directly on the Fortran compiler using the
dnl * macro LF_FLIBS. When CASC_FIND_F77LIBS is included in configure.in,
dnl * it will set the variable F77LIBFLAGS to be a list of flags, which 
dnl * will probably be a set of -L, -R, -l, and -u flags, as well as
dnl * perhaps the absolute paths of some libraries.  The drawback to this
dnl * macro is that it will usually insert some flags (mostly -L flags) 
dnl * that aren't needed, but hopefully they will be harmless.  I haven't
dnl * seen the extra flags that are included by this macro break anything
dnl * yet.  Hopefully more testing on more machines will give confidence
dnl * that this really works and will be able to set up the Fortran links
dnl * on an unknown system.  If this macro sets up nothing, then
dnl * CASC_SET_F77LIBS is called as a backup
dnl *********************************************************************

AC_DEFUN(CASC_FIND_F77LIBS,
[

   if test -z "$F77LIBFLAGS"; then

      dnl * LF_FLIBS creates variable $flibs_result containing a list of 
      dnl * flags related to the Fortran compiler
      LF_FLIBS

      for casc_flag in $flibs_result; do

         dnl * Here we sort the flags in $flibs_result
         case $casc_flag in
         -l* | /*)
            casc_f77_libs="$casc_f77_libs $casc_flag"
         ;;
         -L*)
            casc_f77_dirs="$casc_flag $casc_f77_dirs"
         ;;
         *)
            casc_other_flags="$casc_other_flags $casc_flag"
         ;;
         esac

      done

      F77LIBFLAGS="$casc_other_flags $casc_f77_dirs"

      dnl * each -l flag is checked using CASC_CHECK_LIB_FORTRAN, until
      dnl * successful linking of a test program is accomplished, at which
      dnl * time the loop is broken.  If successful linking does not occur,
      dnl * CASC_CHECK_LIB will check for the library's existence and add
      dnl * to F77LIBFLAGS if it exists.  All libraries listed by explicit
      dnl * path are added to F77LIBFLAGS
      for casc_flag in $casc_f77_libs; do
         case $casc_flag in
         /*)
            if test -f "$casc_flag"; then
               F77LIBFLAGS="$F77LIBFLAGS $casc_flag"
            fi
         ;;
         -l*)
            casc_lib_name=`echo "$casc_flag" | sed 's/-l//g'`
            CASC_CHECK_LIB_FORTRAN($casc_lib_name,
               F77LIBFLAGS="$F77LIBFLAGS $casc_flag"
               casc_result=yes,
               [CASC_CHECK_LIB($casc_lib_name, main,
                   F77LIBFLAGS="$F77LIBFLAGS $casc_flag", ,
                   $F77LIBFLAGS)],
               $F77LIBFLAGS)

            if test "$casc_result" = yes; then 
               casc_result=
               break
            fi
         ;;
         esac
      done

      if test -z "$F77LIBFLAGS"; then
         CASC_SET_F77LIBS
      fi        

      dnl * This deals with -L/lib causing the wrong libc.a to be
      dnl * used when using IBM MPI.  Only appears in configure
      dnl * when CASC_FIND_MPI is called first.
      ifdef([AC_PROVIDE_CASC_FIND_MPI], 
         if test -n "`echo $F77LIBFLAGS | grep '\-L/lib '`"; then
            if test -n "`echo $F77LIBFLAGS | grep xlf`"; then
               F77LIBFLAGS=`echo $F77LIBFLAGS | sed 's/-L\/lib //g'`
            fi
         fi
      )

   fi

   AC_SUBST(F77LIBFLAGS)

])dnl


dnl *********************************************************************
dnl * CASC_SET_MPI sets up the needed MPI library and directory flags.   
dnl * The location of the file mpi.h is put into the variable MPIINCLUDE
dnl * as a -I flag.  The -l flags that specify the needed libraries and
dnl * the -L flags that specify the paths of those libraries are placed in
dnl * the variables MPILIBS and MPILIBDIRS, respectively.  To set the MPI
dnl * libraries and directories manually, use the --with-mpi-include,
dnl * --with-mpi-libs, and --with-mpi-lib-dirs command-line options when
dnl * invoking configure.  Only one directory should be specified with
dnl * --with-mpi-include, while any number of directories can be specified
dnl * by --with-mpi-lib-dirs.  Any number of libraries can be specified
dnl * with --with-mpi-libs, and the libraries must be referred to by their 
dnl * base names, so libmpi.a is just mpi.  It is adviseable to use all 
dnl * three --with flags whenever one is used, because it is likely that
dnl * when one is chosen is will mess up the automatic choices for the
dnl * other two.  If the architecture is unknown, or if the needed MPI
dnl * settings for the current architecture are not known, then the naive
dnl * settings of MPILIBS="-lmpi" and MPILIBDIRS="-L/usr/local/mpi/lib"
dnl * are tested, and if they exist they are used, otherwise the MPILIB*
dnl * variables are left blank.  In the case of rs6000, the variable
dnl * MPIFLAGS is also set. 
dnl **********************************************************************
 
AC_DEFUN(CASC_SET_MPI,
[

   ifdef([AC_PROVIDE_CASC_FIND_MPI], ,
      AC_ARG_WITH(mpi-include, [  --with-mpi-include=DIR  mpi.h is in DIR],
                  casc_mpi_include_dir=$withval)

      AC_ARG_WITH(mpi-libs, 
[  --with-mpi-libs=LIBS    LIBS is space-separated list of library names
                          needed for MPI, e.g. \"nsl socket mpi\"],
                  casc_mpi_libs=$withval)

      AC_ARG_WITH(mpi-lib-dirs, 
[  --with-mpi-lib-dirs=DIRS
                          DIRS is space-separated list of directories
                          containing the libraries specified by
                          \`--with-mpi-libs', e.g \"/usr/lib /usr/local/mpi/lib\"],
                  casc_mpi_lib_dirs=$withval)
   )

   if test -z "$casc_mpi_libs"; then
      AC_REQUIRE([CASC_GUESS_ARCH])

      case $ARCH in

         sun4 | solaris)
            case $F77 in
               *g77)
                   if test -z "$casc_mpi_include_dir"; then
                      casc_mpi_include_dir=/usr/local/mpi/lam/h
                   fi
                   
                   if test -z "$casc_mpi_lib_dirs"; then
                      casc_mpi_lib_dirs="/usr/local/mpi/lam/lib"
                   fi

                   casc_mpi_libs="socket mpi trillium args tstdio t";;

               *)

                  if test -z "$casc_mpi_include_dir"; then
                     casc_mpi_include_dir=/usr/local/mpi/mpich/include
                  fi

                  if test -z "$casc_mpi_lib_dirs"; then
                     casc_mpi_lib_dirs="/usr/local/mpi/mpich/lib/solaris/ch_p4 \
                                       /usr/lib"
                  fi
            
               casc_mpi_libs="nsl socket mpi";;
               esac

            AC_CHECK_HEADER($casc_mpi_include_dir/mpi.h,
                               MPIINCLUDE="-I$casc_mpi_include_dir") ;;


         alpha)
            if test -z "$casc_mpi_include_dir"; then
               casc_mpi_include_dir=/usr/local/mpi/include
            fi
            AC_CHECK_HEADER($casc_mpi_include_dir/mpi.h,
                               MPIINCLUDE="-I$casc_mpi_include_dir")

            if test -z "$casc_mpi_lib_dirs"; then
               casc_mpi_lib_dirs="/usr/local/mpi/lib/alpha/ch_shmem \
                                  /usr/local/lib"
            fi

            casc_mpi_libs="mpich gs";;

         rs6000) 
            if test -z "$casc_mpi_include_dir"; then
               casc_mpi_include_dir=/usr/lpp/ppe.poe/include
            fi
            AC_CHECK_HEADER($casc_mpi_include_dir/mpi.h,
                               MPIINCLUDE="-I$casc_mpi_include_dir")

            if test -z "$casc_mpi_lib_dirs"; then
               casc_mpi_lib_dirs=/usr/lpp/ppe.poe/lib
            fi

            casc_mpi_libs=mpi

            MPIFLAGS="-binitfini:poe_remote_main";;

         IRIX64 | iris4d) 
            if test -z "$casc_mpi_include_dir"; then
               casc_mpi_include_dir=/usr/local/mpi/include
            fi
            AC_CHECK_HEADER($casc_mpi_include_dir/mpi.h,
                               MPIINCLUDE="-I$casc_mpi_include_dir")

            if test -z "$casc_mpi_lib_dirs"; then
               casc_mpi_lib_dirs=/usr/local/mpi/lib/IRIX64/ch_p4
            fi

            casc_mpi_libs=mpi;; 
        
         *)
AC_MSG_WARN([trying naive MPI settings - can use --with flags to change])
            if test -z "$casc_mpi_include_dir"; then
               casc_mpi_include_dir=/usr/local/mpi/include
            fi
            AC_CHECK_HEADER($casc_mpi_include_dir/mpi.h,
                               MPIINCLUDE="-I$casc_mpi_include_dir")

            if test -z "$casc_mpi_lib_dirs"; then
               casc_mpi_lib_dirs=/usr/local/mpi/lib
            fi
            casc_mpi_libs=mpi ;;
      esac

      for casc_lib in $casc_mpi_libs; do
         CASC_ADD_LIB($casc_lib, main, $casc_mpi_lib_dirs, MPI)
      done

   else
      if test -n "$casc_mpi_include_dir"; then
         MPIINCLUDE="-I$casc_mpi_include_dir"
      else
         MPIINCLUDE=
      fi

      if test -n "$casc_mpi_lib_dirs"; then
         for casc_lib_dir in $casc_mpi_lib_dirs; do
            MPILIBDIRS="-L$casc_lib_dir $MPILIBDIRS"
         done
      else
         MPILIBDIRS=
      fi

      for casc_lib in $casc_mpi_libs; do
         MPILIBS="$MPILIBS -l$casc_lib"
      done
   fi
])dnl


dnl ********************************************************************
dnl * CASC_FIND_MPI will determine the libraries, directories, and other
dnl * flags needed to compile and link programs with MPI function calls.
dnl * This macro runs tests on the script found by the CASC_PROG_MPICC
dnl * macro.  If there is no such mpicc-type script in the PATH and
dnl * MPICC is not set manually, then this macro will not work.  One may
dnl * question why these settings would need to be determined if there
dnl * already is mpicc available, and that is a valid question.  I can
dnl * think of a couple of reasons one may want to use these settings 
dnl * rather than using mpicc directly.  First, these settings allow you
dnl * to choose the C compiler you wish to use rather than using whatever
dnl * compiler is written into mpicc.  Also, the settings determined by
dnl * this macro should also work with C++ and Fortran compilers, so you
dnl * won't need to have mpiCC and mpif77 alongside mpicc.  This is
dnl * especially helpful on systems that don't have mpiCC.  The advantage
dnl * of this macro over CASC_SET_MPI is that this one doesn't require
dnl * a test of the machine type and thus will hopefully work on unknown
dnl * architectures.  The main disadvantage is that it relies on mpicc.
dnl * --with-mpi-include, --with-mpi-libs, and --with-mpi-lib-dirs can be
dnl * used to manually override the automatic test, just as with
dnl * CASC_SET_MPI.  If any one of these three options are used, the
dnl * automatic test will not be run, so it is best to call all three
dnl * whenever one is called.  In addition, the option --with-mpi-flags is
dnl * available here to set any other flags that may be needed, but it
dnl * does not override the automatic test.  Flags set by --with-mpi-flags
dnl * will be added to the variable MPIFLAGS.  This way, if the macro, for
dnl * whatever reason, leaves off a necessary flag, the flag can be added 
dnl * to MPIFLAGS without eliminating anything else.  The other variables
dnl * set are MPIINCLUDE, MPILIBS, and MPILIBDIRS, just as in 
dnl * CASC_SET_MPI.  This macro also incorporates CASC_SET_MPI as a backup
dnl * plan, where if there is no mpicc, it will use the settings
dnl * determined by architecture name in CASC_SET_MPI
dnl ********************************************************************

AC_DEFUN(CASC_FIND_MPI,
[

   dnl * Set up user options.  If user uses any of the fist three options,
   dnl * then automatic tests are not run.

   casc_user_chose_mpi=no
   AC_ARG_WITH(mpi-include, [  --with-mpi-include=DIR  mpi.h is in DIR],
               MPIINCLUDE=-I$withval; casc_user_chose_mpi=yes)

   AC_ARG_WITH(mpi-libs,
[  --with-mpi-libs=LIBS    LIBS is space-separated list of library names 
                          needed for MPI, e.g. \"nsl socket mpi\"],  
               for mpi_lib in $withval; do
                  MPILIBS="$MPILIBS -l$mpi_lib"
               done; casc_user_chose_mpi=yes)


   AC_ARG_WITH(mpi-lib-dirs,
[  --with-mpi-lib-dirs=DIRS
                          DIRS is space-separated list of directories
                          containing the libraries specified by
                          \`--with-mpi-libs', e.g \"/usr/lib /usr/local/mpi/lib\"],
               for mpi_lib_dir in $withval; do
                  MPILIBDIRS="-L$mpi_lib_dir $MPILIBDIRS"
               done; casc_user_chose_mpi=yes)

   dnl * --with-mpi-flags only adds to automatic selections, does not override

   AC_ARG_WITH(mpi-flags,
[  --with-mpi-flags=FLAGS  FLAGS is space-separated list of whatever flags other
                          than -l and -L are needed to link with mpi libraries],
                          MPIFLAGS=$withval)


   if test "$casc_user_chose_mpi" = "no"; then

   dnl * Find an MPICC.  If there is none, call CASC_SET_MPI to choose MPI
   dnl * settings based on architecture name.  If CASC_SET_MPI fails,
   dnl * print warning message.  Manual MPI settings must be used.

      AC_ARG_WITH(MPICC,
[  --with-MPICC=ARG        ARG is mpicc or similar MPI C compiling tool],
         MPICC=$withval,
         [AC_CHECK_PROGS(MPICC, mpicc mpcc tmcc hcc)])

      if test -z "$MPICC"; then
         AC_MSG_WARN([no acceptable mpicc found in \$PATH])
         CASC_SET_MPI
         if test -z "$MPILIBS"; then
            AC_MSG_WARN([MPI not found - must set manually using --with flags])
         fi

      dnl * When $MPICC is there, run the automatic test

      else      

         changequote(, )dnl

         AC_MSG_CHECKING(for location of mpi.h)

         cat > mpconftest.c << EOF
#include <stdio.h>
#include "mpi.h"

main(int argc, char **argv)
{
   int rank, size;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Finalize();
   return 0;
}
EOF

         casc_mplibs=
         casc_mplibdirs=
         casc_flags=
         casc_lmpi_exists=no

         dnl * These are various ways to produce verbose output from $MPICC

         for casc_command in "$MPICC -show" "$MPICC -v" "$MPICC -#" "$MPICC"; do

            casc_this_output=`$casc_command mpconftest.c -o mpconftest 2>&1`

            dnl * If $MPICC uses xlc, then commas must be removed from output
            xlc_p=`echo $casc_this_output | grep xlcentry`
            if test -n "$xlc_p"; then
               casc_this_output=`echo $casc_this_output | sed 's/,/ /g'`
            fi

            dnl * Turn on flag once -lmpi is found in output
            lmpi_p=`echo $casc_this_output | grep "\-lmpi"`
            if test -n "$lmpi_p"; then
               casc_lmpi_exists=yes
            fi

            casc_mpoutput="$casc_mpoutput $casc_this_output"
            casc_this_output=

         done

         rm -rf mpconftest*

         dnl * Add -lmpi if it was never found
         if test "$casc_lmpi_exists" = "no"; then
            casc_mplibs="-lmpi"
         else
            casc_mplibs=
         fi

         casc_want_arg=

         dnl * check every word in output to find possible flags
         for casc_arg in $casc_mpoutput; do
            casc_old_want_arg=$casc_want_arg
            casc_want_arg=  

            if test -n "$casc_old_want_arg"; then
               case "$casc_arg" in
               -*)
                  casc_old_want_arg=
               ;;
               esac
            fi

            case "$casc_old_want_arg" in
            '')
               case $casc_arg in
               /*.a)
                  exists=false
                  for f in $casc_flags; do
                     if test x$casc_arg = x$f; then
                        exists=true
                     fi
                  done
                  if $exists; then
                     casc_arg=
                  else
                     casc_flags="$casc_flags $casc_arg"
                  fi
               ;;
               -lang*)
                  casc_arg=
               ;;
               -[lLR])
                  casc_want_arg=$casc_arg
                  casc_arg=
               ;;
               -[lLR]*)
                  exists=false
                  for f in $casc_flags; do
                     if test x$casc_arg = x$f; then
                        exists=true
                     fi
                  done
                  if $exists; then
                     casc_arg=
                  else
                     casc_flags="$casc_flags $casc_arg"
                  fi
               ;;
               -u)
                  casc_want_arg=$casc_arg
                  casc_arg=
               ;;
               -Y)
                  casc_want_arg=$casc_arg
                  casc_arg=
               ;;
               -I)
                  casc_want_arg=$casc_arg
                  casc_arg=
               ;;
               -I*)
                  exists=false
                  for f in $casc_flags; do
                     if test x$casc_arg = x$f; then
                        exists=true
                     fi
                  done
                  if $exists; then
                     casc_arg=
                  else
                     casc_flags="$casc_flags $casc_arg"
                  fi
               ;;
               *)
                  casc_arg=
               ;;
               esac

            ;;
            -[lLRI])
               casc_arg="casc_old_want_arg $arg"
            ;;
            -u)
               casc_arg="-u $casc_arg"
            ;;
            -Y)
               casc_arg=`echo $casc_arg | sed -e 's%^P,%%'`
               SAVE_IFS=$IFS
               IFS=:
               casc_list=
               for casc_elt in $casc_arg; do
                  casc_list="$casc_list -L$casc_elt"
               done
               IFS=$SAVE_IFS
               casc_arg="$casc_list"
            ;;
            esac

            dnl * separate found flags into includes, libdirs, libs, flags
            if test -n "$casc_arg"; then
               case $casc_arg in
               -I*)

                  if test -z "$MPIINCLUDE"; then
                     casc_include_dir=`echo "$casc_arg" | sed 's/-I//g'`

                     if test -f "$casc_include_dir/mpi.h"; then
                        MPIINCLUDE=$casc_arg
                     else
                        casc_arg=
                     fi
                  else
                     casc_arg=
                  fi
               ;;
               -[LR]*)

                  casc_mplibdirs="$casc_mplibdirs $casc_arg"
               ;;
               -l* | /*)

                  casc_mplibs="$casc_mplibs $casc_arg"
               ;;
               *)
                  casc_mpflags="$casc_mpflags $casc_arg"
               ;;
               esac

               LIBS_SAVE=$LIBS
               LIBS="$MPIINCLUDE $casc_mpflags $casc_mplibdirs $casc_mplibs"

               changequote([, ])dnl


               dnl * Test to see if flags found up to this point are
               dnl * sufficient to compile and link test program.  If not,
               dnl * the loop keeps going to the next word
               AC_TRY_LINK(
                  ifelse(AC_LANG, CPLUSPLUS,

[#ifdef __cplusplus
extern "C"
#endif
])dnl
[#include <stdio.h>
#include "mpi.h"
], [int rank, size;
   int argc;
   char **argv;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Finalize();
],
                  casc_result=yes)

               LIBS=$LIBS_SAVE

               if test "$casc_result" = yes; then
                  casc_result=
                  break
               fi
            fi
         done

         dnl * After loop is done, set variables to be substituted
         MPILIBS=$casc_mplibs
         MPILIBDIRS=$casc_mplibdirs
         MPIFLAGS="$MPIFLAGS $casc_mpflags"

         dnl * This deals with -L/lib causing the wrong libc.a to be
         dnl * used when using IBM MPI.  Only appears in configure
         dnl * when CASC_FIND_F77LIBS is called first.
         ifdef([AC_PROVIDE_CASC_FIND_F77LIBS], 
            if test -n "`echo $F77LIBFLAGS | grep '\-L/lib '`"; then
               if test -n "`echo $F77LIBFLAGS | grep xlf`"; then
                  F77LIBFLAGS=`echo $F77LIBFLAGS | sed 's/-L\/lib //g'`
               fi
            fi
         )

         AC_MSG_RESULT($MPIINCLUDE)
         AC_MSG_CHECKING(for MPI library directories)
         AC_MSG_RESULT($MPILIBDIRS)
         AC_MSG_CHECKING(for MPI libraries)
         AC_MSG_RESULT($MPILIBS)
         AC_MSG_CHECKING(for other MPI-related flags)
         AC_MSG_RESULT($MPIFLAGS)
      fi
   fi

   AC_SUBST(MPIINCLUDE)
   AC_SUBST(MPILIBDIRS)
   AC_SUBST(MPILIBS)
   AC_SUBST(MPIFLAGS)

])dnl



dnl ***********************************************************************
dnl CASC_CHECK_LIB(LIBRARY, FUNCTION [, ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND
dnl              [, OTHER-LIBRARIES [, CACHE-CHOICE]]]])
dnl * This is the same as AC_CHECK_LIB, except when it tests for LIBRARY
dnl * it puts the flag -lLIBRARY after $LIBS and OTHER-LIBRARIES.  The Sun
dnl * cc compiler does not search for LIBRARY in any directories specified
dnl * by -L in OTHER-LIBRARIES when -lLIBRARY is listed first.  The
dnl * functionality of this macro is the same as that of AC_CHECK_LIB in
dnl * the Autoconf documentation.  
dnl * CACHE-CHOICE [$6]added by N. Elliott, 6-24-98.  If CACHE-CHOICE is 'no',
dnl * the results of this test are not cached.  CACHE-CHOICE should be
dnl * used only when this test is called recursively.
dnl **********************************************************************

AC_DEFUN(CASC_CHECK_LIB,
[AC_MSG_CHECKING([for -l$1])
dnl Use a cache variable name containing both the library and function name,
dnl because the test really is for library $1 defining function $2, not
dnl just for library $1.  Separate tests with the same $1 and different $2s
dnl may have different results.
ac_lib_var=`echo $1['_']$2 | tr './+\055' '__p_'`
AC_CACHE_VAL(ac_cv_lib_$ac_lib_var,
[ac_save_LIBS="$LIBS"
LIBS="$5 $LIBS -l$1"
AC_TRY_LINK(dnl
ifelse([$2], [main], , dnl Avoid conflicting decl of main.
[/* Override any gcc2 internal prototype to avoid an error.  */
]ifelse(AC_LANG, CPLUSPLUS, [#ifdef __cplusplus 
extern "C"
#endif
])dnl
[/* We use char because int might match the return type of a gcc2
    builtin and then its argument prototype would still apply.  */
char $2();
]),
            [$2()],
            eval "ac_cv_lib_$ac_lib_var=yes",
            eval "ac_cv_lib_$ac_lib_var=no")dnl
LIBS="$ac_save_LIBS"
])dnl
if eval "test \"`echo '$ac_cv_lib_'$ac_lib_var`\" = yes"; then
  AC_MSG_RESULT(yes)  
  ifelse([$3], ,
[changequote(, )dnl
  ac_tr_lib=HAVE_LIB`echo $1 | tr 'abcdefghijklmnopqrstuvwxyz' 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'`
changequote([, ])dnl
  AC_DEFINE_UNQUOTED($ac_tr_lib)
  LIBS="-l$1 $LIBS"
], [
ifelse([$6], no, unset ac_cv_lib_$ac_lib_var)
$3])
else
  AC_MSG_RESULT(no) 
ifelse([$4], , , [
ifelse([$6], no, unset ac_cv_lib_$ac_lib_var)
$4
])dnl
fi
ifelse([$6], no, unset ac_cv_lib_$ac_lib_var)
])


dnl *********************************************************************
dnl * CASC_SET_COPT(OPTIMIZATION-FLAGS)
dnl * Call this macro to set C compiler optimization flags to
dnl * OPTIMIZATION-FLAGS.  They will be stored in COPT.  Existing values
dnl * of COPT in the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_COPT,
[
   if test -z "$COPT"; then
      COPT="$1"
   fi
   AC_SUBST(COPT)
])dnl


dnl *********************************************************************
dnl * CASC_SET_CXXOPT(OPTIMIZATION-FLAGS)
dnl * Call this macro to set C++ compiler optimization flags to
dnl * OPTIMIZATION-FLAGS.  They will be stored in CXXOPT.  Existing values
dnl * of CXXOPT in the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_CXXOPT,
[
   if test -z "$CXXOPT"; then
      CXXOPT="$1"
   fi
   AC_SUBST(CXXOPT)
])dnl


dnl *********************************************************************
dnl * CASC_SET_FOPT(OPTIMIZATION-FLAGS)
dnl * Call this macro to set F77 compiler optimization flags to
dnl * OPTIMIZATION-FLAGS.  They will be stored in FOPT.  Existing values
dnl * of FOPT in the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_FOPT,
[
   if test -z "$FOPT"; then
      FOPT="$1"
   fi
   AC_SUBST(FOPT)
])dnl


dnl *********************************************************************
dnl * CASC_SET_CDEBUG(DEBUG-FLAGS)
dnl * Call this macro to set C compiler debugging flags to DEBUG-FLAGS.
dnl * They will be stored in CDEBUG.  Existing values of CDEBUG in
dnl * the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_CDEBUG,
[
   if test -z "$CDEBUG"; then
      CDEBUG="$1"
   fi
   AC_SUBST(CDEBUG)
])dnl


dnl *********************************************************************
dnl * CASC_SET_CXXDEBUG(DEBUG-FLAGS)
dnl * Call this macro to set C++ compiler debugging flags to DEBUG-FLAGS.
dnl * They will be stored in CXXDEBUG.  Existing values of CXXDEBUG in
dnl * the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_CXXDEBUG,
[
   if test -z "$CXXDEBUG"; then
      CXXDEBUG="$1"
   fi
   AC_SUBST(CXXDEBUG)
])dnl


dnl *********************************************************************
dnl * CASC_SET_FDEBUG(DEBUG-FLAGS)
dnl * Call this macro to set F77 compiler debugging flags to DEBUG-FLAGS.
dnl * They will be stored in FDEBUG.  Existing values of FDEBUG in
dnl * the shell will be used if they exist.
dnl *********************************************************************

AC_DEFUN(CASC_SET_FDEBUG,
[
   if test -z "$FDEBUG"; then
      FDEBUG="$1"
   fi
   AC_SUBST(FDEBUG)
])dnl


dnl **********************************************************************
dnl * CASC_CHOOSE_OPT_OR_DEBUG
dnl * Before this macro is called in configure.in, the macros
dnl * CASC_SET_COPT and CASC_SET_CDEBUG and/or their C++ and Fortran
dnl * counterparts should be invoked to set both optimization and
dnl * debugging flags.  The effect of this macro is to turn off one set of
dnl * flags when the other is selected by the user.  This macro invokes
dnl * the macro AC_ARG_ENABLE to give the configure script the
dnl * command-line option "--enable-opt-debug=ARG", where ARG can equal
dnl * 'opt', 'debug', or 'both'.  If 'opt' then all debugging compiler
dnl * flags are turned off, and if 'debug' then all optimization compiler
dnl * flags are turned off.  Also, the variable OPTCHOICE is set to 'O'
dnl * for optimization and 'g' for debugging.  OPTCHOICE was added because
dnl * PetSc libraries are installed in mirrored directories called 'libO'
dnl * and 'libg'.  If ARG is 'both', then neither are turned off.  If an
dnl * invalid value of ARG is given, then neither are turned off, and a
dnl * warning message is printed.  If --enable-opt-debug=ARG is not 
dnl * called, then the default action is to enable optimization flags and
dnl * to turn off debugging flags and set OPTCHOICE to 'O'.
dnl **********************************************************************

AC_DEFUN(CASC_CHOOSE_OPT_OR_DEBUG,
[
dnl   if test -z "$casc_opt_or_debug"; then
dnl      casc_opt_or_debug=opt
dnl   fi
AC_ARG_ENABLE(opt-debug,
[  --enable-opt-debug=ARG
         ARG=debug  --  enable debug flags, disable optimization
         ARG=opt    --  enable optimization flags, disable debugging
         ARG=both   --  both optimization and debugging flags enabled],

   casc_opt_or_debug="$enableval",
   casc_opt_or_debug=opt; OPTCHOICE=O
)

   AC_MSG_CHECKING(optimization/debugging choice)
   case $casc_opt_or_debug in

      opt)
         echo "opt"
         CDEBUG=
         CXXDEBUG=
         FDEBUG=
         OPTCHOICE=O
      ;;
      debug)
         echo "debug"
         COPT=
         CXXOPT=
         FOPT=  
         OPTCHOICE=g
      ;;
      both)
         echo "both"
         OPTCHOICE=O
      ;;
      *)
         echo "$casc_opt_or_debug"
            AC_MSG_WARN(
        [Invalid argument given to the flag --enable-opt-debug.
         The only acceptable choices are '--enable-opt-debug=opt',
         '--enable-opt-debug=debug', and '--enable-opt-debug=both'.  Both 
         optimization flags and debugging flags remain unchanged.])
      ;;
   esac

   AC_SUBST(OPTCHOICE)

])dnl


dnl **********************************************************************
dnl * CASC_CONFIG_OUTPUT_LIST(DIR-LIST[, OUTPUT-FILE])
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
dnl *     CASC_CONFIG_OUTPUT_LIST(dir1 dir2, Foo)
dnl *     If you also called this macro for Makefile as described above,
dnl *     you should call
dnl *     AC_OUTPUT($Makefile_list $Foo_list)
dnl *     at the end of configure.in .
dnl *********************************************************************


AC_DEFUN(CASC_CONFIG_OUTPUT_LIST,
[
   define([m_OUTPUT_LIST], ifelse([$2], , Makefile_list, [$2_list]))

   if test -z "$srcdir"; then
      srcdir=.
   fi
   if test -n "$2"; then
      casc_output_file=$2
   else   
      casc_output_file=Makefile
   fi   
      
   for casc_dir in $1; do
      if test -f $srcdir/$casc_dir/$casc_output_file.in; then
         m_OUTPUT_LIST="$m_OUTPUT_LIST $casc_dir/$casc_output_file"
      fi
   done
])dnl


dnl *********************************************************************
dnl * CASC_CHECK_HEADER(HEADER-FILE, DIRECTORY-LIST[, ACTION-IF-FOUND[,
dnl *                   ACTION-IF-NOT-FOUND]])
dnl * This macro is an alternative to AC_CHECK_HEADER.  It does
dnl * essentially the same thing, but it allows the user to specify
dnl * a directory list if HEADER-FILE can not be found in the current path
dnl * for #includes, and it adds to the variable INCLUDES the first
dnl * directory in DIRECTORY-LIST from where HEADER-FILE can be included.
dnl *********************************************************************

AC_DEFUN(CASC_CHECK_HEADER,
[
   for casc_dir in '' $2 ; do
      if test -n "$casc_dir"; then
         casc_header=$casc_dir/$1
      else
         casc_header=$1
      fi
      AC_CHECK_HEADER( $casc_header, 
         if test -n "$casc_dir"; then
            INCLUDES="$INCLUDES -I$casc_dir"
         fi
         casc_header_found=yes
         ifelse([$3], , , [$3])
         break )

   done

   ifelse([$4], , ,
      if test "$casc_header_found" != "yes"; then
         [$4]
      fi
   )

   unset casc_header_found
])dnl


dnl **********************************************************************
dnl * CASC_GUESS_ARCH
dnl * Guesses the current architecture, unless ARCH has been preset.
dnl * Uses the utility 'tarch', which is a Bourne shell script that should
dnl * be in the same directory as the configure script.  If tarch is not
dnl * present or if it fails, ARCH is set to the value, if any of shell
dnl * variable HOSTTYPE, otherwise ARCH is set to "unknown".
dnl **********************************************************************

AC_DEFUN(CASC_GUESS_ARCH,
[
   AC_MSG_CHECKING(the architecture)

   if test -z "$ARCH"; then

      casc_tarch_dir=
      for casc_dir in $srcdir $srcdir/.. $srcdir/../..; do
         if test -f $casc_dir/tarch; then
            casc_tarch_dir=$casc_dir
            casc_tarch=$casc_tarch_dir/tarch
            break
         fi
      done

      if test -z "$casc_tarch_dir"; then
         echo "cannot find tarch, using \$HOSTTYPE as the architecture"
         ARCH=$HOSTTYPE
      else
         ARCH="`$casc_tarch`"

         if test -z "$ARCH" -o "$ARCH" = "unknown"; then
            ARCH=$HOSTTYPE
         fi
      fi

      if test -z "$ARCH"; then
         ARCH=unknown
         echo "architecture is unknown"
      else
         echo $ARCH
      fi    
   else
      echo $ARCH
   fi

   AC_SUBST(ARCH)

])dnl


dnl **********************************************************************
dnl * CASC_CXX_NAMESPACE checks if the C++ compiler supports the namespace
dnl * feature.  It tries to compile and link a simple program, and it
dnl * defines the preprocessor macro HAVE_NAMESPACE if namespace is indeed
dnl * supported
dnl **********************************************************************

AC_DEFUN(CASC_CXX_NAMESPACE,
[
   AC_REQUIRE([AC_PROG_CXX])
   AC_MSG_CHECKING(whether ${CXX} supports namespace)
   AC_CACHE_VAL(casc_cv_have_namespace,
   [
      AC_LANG_SAVE
      AC_LANG_CPLUSPLUS

      cat > conftest.$ac_ext <<EOF

#include "confdefs.h"
namespace CONFTEST {
   int t();

   int t(void) 
   { return 0; } 
}
using namespace CONFTEST;
int main(void) { int x=t(); x++; return 0; }
EOF

      if { (eval echo configure:__oline__: \"$ac_link\") 1>&5; (eval $ac_link) \
                                                       2>&5; }; then
         rm -rf conftest*
         casc_cv_have_namespace=yes
      else
         rm -rf conftest*
         casc_cv_have_namespace=no
      fi

      rm -f conftest*

      AC_LANG_RESTORE
   ])

   AC_MSG_RESULT($casc_cv_have_namespace)
   if test "$casc_cv_have_namespace" = yes; then
      AC_DEFINE(HAVE_NAMESPACE)
   fi
])dnl 


dnl * The following are macros copied from outside sources


dnl ********************************************************************
dnl * PAC_GET_FORTNAMES is distributed with mpich.  It checks what format
dnl * is used to call Fortran subroutines in C functions.  This macro
dnl * defines the shell variable $FORTRANNAMES and creates -D  
dnl * preprocessor flags that tell what the Fortran name-mangling is.  The
dnl * preprocessor macros defined are FORTRAN_DOUBLE_UNDERSCORE,
dnl * FORTRAN_UNDERSCORE, FORTRAN_CAPS, and FORTRAN_NO_UNDERSCORE.  The
dnl * possible values for FORTRANNAMES are the same words without
dnl * underscores.
dnl * 
dnl * Changes:
dnl *    AC_DEFINE lines to define preprocessor macros that are assigned
dnl *    to DEFS added by Noah Elliott May 18, 1998
dnl ********************************************************************

define(PAC_GET_FORTNAMES,[
   # Check for strange behavior of Fortran.  For example, some FreeBSD
   # systems use f2c to implement f77, and the version of f2c that they
   # use generates TWO (!!!) trailing underscores 
   # Currently, WDEF is not used but could be...
   #
   # Eventually, we want to be able to override the choices here and
   # force a particular form.  This is particularly useful in systems
   # where a Fortran compiler option is used to force a particular
   # external name format (rs6000 xlf, for example).
   cat > confftest.f <<EOF
       subroutine mpir_init_fop( a )
       integer a
       a = 1
       return
       end
EOF
   $F77 $FFLAGS -c confftest.f > /dev/null 2>&1
   if test ! -s confftest.o ; then
        print_error "Unable to test Fortran compiler"
        print_error "(compiling a test program failed to produce an "
        print_error "object file)." 
        NOF77=1
        HAS_FORTRAN=0
   elif test -z "$FORTRANNAMES" ; then
    # We have to be careful here, since the name may occur in several  
    # forms.  We try to handle this by testing for several forms
    # directly.
    if test $arch_CRAY ; then
     # Cray doesn't accept -a ...
     nameform1=`strings confftest.o | grep mpir_init_fop_  | head -1`
     nameform2=`strings confftest.o | grep MPIR_INIT_FOP   | head -1`
     nameform3=`strings confftest.o | grep mpir_init_fop   | head -1`
     nameform4=`strings confftest.o | grep mpir_init_fop__ | head -1`
    else
     nameform1=`strings -a confftest.o | grep mpir_init_fop_  | head -1`
     nameform2=`strings -a confftest.o | grep MPIR_INIT_FOP   | head -1`
     nameform3=`strings -a confftest.o | grep mpir_init_fop   | head -1`
     nameform4=`strings -a confftest.o | grep mpir_init_fop__ | head -1`
    fi
    /bin/rm -f confftest.f confftest.o
    if test -n "$nameform4" ; then
        AC_DEFINE(FORTRAN_DOUBLE_UNDERSCORE)
        echo "Fortran externals are lower case and have 1 or 2 trailing underscores"
        FORTRANNAMES="FORTRANDOUBLEUNDERSCORE"
    elif test -n "$nameform1" ; then
        # We don't set this in CFLAGS; it is a default case
        AC_DEFINE(FORTRAN_UNDERSCORE)
        echo "Fortran externals have a trailing underscore and are lowercase"
        FORTRANNAMES="FORTRANUNDERSCORE"
    elif test -n "$nameform2" ; then
        AC_DEFINE(FORTRAN_CAPS)
        echo "Fortran externals are uppercase"
        FORTRANNAMES="FORTRANCAPS"
    elif test -n "$nameform3" ; then
        AC_DEFINE(FORTRAN_NO_UNDERSCORE)
        echo "Fortran externals are lower case"
        FORTRANNAMES="FORTRANNOUNDERSCORE"
    else
        print_error "Unable to determine the form of Fortran external names"
        print_error "Make sure that the compiler $F77 can be run on this system"
#       print_error "If you have problems linking, try using the -nof77 option"
#        print_error "to configure and rebuild MPICH."
        print_error "Turning off Fortran (-nof77 being assumed)."
        NOF77=1
        HAS_FORTRAN=0
    fi
    fi
    if test -n "$FORTRANNAMES" ; then
        WDEF="-D$FORTRANNAMES"
    fi  
    ])dnl


dnl ****************************************************************
dnl * LF_FLIBS was copied from E. Gkioulekas' autotools package for use in
dnl * CASC_FIND_F77LIBS.  It's probably not good to be used all by itself.
dnl * It uses the output the Fortran compiler gives when given a -v flag
dnl * to produce a list of flags that the Fortran compiler uses.  From
dnl * this list CASC_FIND_F77LIBS sets up the Fortran/C interface flags.
dnl *
dnl * Changes:
dnl * AC_SUBST(FLIBS) suppressed by N. Elliott 7-10-98
dnl *****************************************************************
   
AC_DEFUN(LF_FLIBS,[
  AC_MSG_CHECKING(for Fortran libraries)
  dnl
  dnl Write a minimal program and compile it with -v. I don't know
  dnl what to do if your compiler doesn't have -v
  dnl
  changequote(, )dnl
  echo "      END" > conftest.f
  foutput=`${F77-f77} -v -o conftest conftest.f 2>&1`
  dnl
  dnl The easiest thing to do for xlf output is to replace all the commas
  dnl with spaces.  Try to only do that if the output is really from xlf,
  dnl since doing that causes problems on other systems.
  dnl
  xlf_p=`echo $foutput | grep xlfentry`
  if test -n "$xlf_p"; then
    foutput=`echo $foutput | sed 's/,/ /g'`
  fi
  dnl
  ld_run_path=`echo $foutput | \
    sed -n -e 's/^.*LD_RUN_PATH *= *\([^ ]*\).*/\1/p'`
  dnl
  dnl We are only supposed to find this on Solaris systems...
  dnl Uh, the run path should be absolute, shouldn't it?
  dnl
  case "$ld_run_path" in
    /*)
      if test "$ac_cv_prog_gcc" = yes; then
        ld_run_path="-Xlinker -R -Xlinker $ld_run_path"
      else
        ld_run_path="-R $ld_run_path"
      fi
    ;;
    *)
      ld_run_path=
    ;;
  esac
  dnl
  flibs=
  lflags=
  dnl
  dnl If want_arg is set, we know we want the arg to be added to the list,
  dnl so we don't have to examine it.
  dnl
  want_arg=
  dnl
  for arg in $foutput; do
    old_want_arg=$want_arg
    want_arg=
  dnl
  dnl None of the options that take arguments expect the argument to
  dnl start with a -, so pretend we didn't see anything special.
  dnl
    if test -n "$old_want_arg"; then
      case "$arg" in
        -*)
        old_want_arg=
        ;;
      esac
    fi
    case "$old_want_arg" in
      '')
        case $arg in
        /*.a)
          exists=false
          for f in $lflags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            lflags="$lflags $arg"
          fi
        ;;
        -bI:*)
          exists=false
          for f in $lflags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            if test "$ac_cv_prog_gcc" = yes; then
              lflags="$lflags -Xlinker $arg"
            else
              lflags="$lflags $arg"
            fi
          fi
        ;;
        -lang* | -lcrt0.o | -lc )
          arg=
        ;;
        -[lLR])
          want_arg=$arg
          arg=
        ;;
        -[lLR]*)
          exists=false
          for f in $lflags; do
            if test x$arg = x$f; then
              exists=true
            fi
          done
          if $exists; then
            arg=
          else
            case "$arg" in
              -lkernel32)
                case "$canonical_host_type" in
                  *-*-cygwin32)
                  ;;
                  *)
                    lflags="$lflags $arg"
                  ;;
                esac
              ;;
              -lm)
              ;;
              *)
                lflags="$lflags $arg"
              ;;
            esac
          fi
        ;;
        -u)
          want_arg=$arg
          arg=
        ;;
        -Y)
          want_arg=$arg
          arg=
        ;;
        *)
          arg=
        ;;
        esac
      ;;
      -[lLR])
        arg="$old_want_arg $arg"
      ;;
      -u)
        arg="-u $arg"
      ;;
      -Y)
  dnl
  dnl Should probably try to ensure unique directory options here too.
  dnl This probably only applies to Solaris systems, and then will only
  dnl work with gcc...
  dnl
        arg=`echo $arg | sed -e 's%^P,%%'`
        SAVE_IFS=$IFS
        IFS=:
        list=
        for elt in $arg; do
        list="$list -L$elt"
        done
        IFS=$SAVE_IFS
        arg="$list"
      ;;
    esac
  dnl
    if test -n "$arg"; then
      flibs="$flibs $arg"
    fi
  done
  if test -n "$ld_run_path"; then
    flibs_result="$ld_run_path $flibs"
  else
    flibs_result="$flibs"
  fi
  changequote([, ])dnl
  rm -f conftest.f conftest.o conftest
  dnl
  dnl Phew! Done! Now, output the result
  dnl
  FLIBS="$flibs_result"
  AC_MSG_RESULT([$FLIBS])
dnl  AC_SUBST(FLIBS)
])dnl

dnl ****************************************************************
dnl * ICE_CXX_BOOL checks if the C++ compiler accepts the `bool' keyword.
dnl * if it does, define the preprocessor macro `HAVE_BOOL'.
dnl ****************************************************************

AC_DEFUN(ICE_CXX_BOOL,
[
AC_REQUIRE([AC_PROG_CXX])
AC_MSG_CHECKING(whether ${CXX} supports bool types)
AC_CACHE_VAL(ice_cv_have_bool,
[
AC_LANG_SAVE
AC_LANG_CPLUSPLUS
AC_TRY_COMPILE(,[bool b = true;],
ice_cv_have_bool=yes,
ice_cv_have_bool=no)
AC_LANG_RESTORE
])
AC_MSG_RESULT($ice_cv_have_bool)
if test "$ice_cv_have_bool" = yes; then
AC_DEFINE(HAVE_BOOL)
fi
])dnl



dnl ****************************************************************
dnl * ICE_CXX_EXPLICIT_TEMPLATE_INSTANTIATION checks if the C++ compiler
dnl * supports explicit template instantiation.  If so, define the
dnl * preprocessor macro `HAVE_EXPLICIT_TEMPLATE_INSTANTIATION.'
dnl ****************************************************************

AC_DEFUN(ICE_CXX_EXPLICIT_TEMPLATE_INSTANTIATION,
[ AC_REQUIRE([AC_PROG_CXX])
AC_MSG_CHECKING(whether ${CXX} supports explicit template instantiation)
AC_CACHE_VAL(ice_cv_have_explicit_template_instantiation,
[
AC_LANG_SAVE
AC_LANG_CPLUSPLUS
AC_TRY_COMPILE([
template <class T> class Pointer { public: T *value; };
template class Pointer<char>;
], [/* empty */],
ice_cv_have_explicit_template_instantiation=yes,
ice_cv_have_explicit_template_instantiation=no)
AC_LANG_RESTORE
])
AC_MSG_RESULT($ice_cv_have_explicit_template_instantiation)
if test "$ice_cv_have_explicit_template_instantiation" = yes; then
AC_DEFINE(HAVE_EXPLICIT_TEMPLATE_INSTANTIATION)
fi
])dnl


