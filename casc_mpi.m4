dnl ********************************************************************
dnl * CASC_PROG_MPICC searches the PATH for an available MPI C compiler
dnl * wraparound.  It assigns the name to MPICC.
dnl ********************************************************************

AC_DEFUN(CASC_PROG_MPICC,
[
   AC_CHECK_PROGS(MPICC, mpcc mpicc tmcc hcc)
   test -z "$MPICC" && AC_MSG_ERROR([no acceptable mpicc found in \$PATH])
])dnl


dnl ********************************************************************
dnl * CASC_PROG_MPICXX searches the PATH for an available MPI C++ 
dnl * compiler wraparound.  It assigns the name to MPICXX.
dnl ********************************************************************

AC_DEFUN(CASC_PROG_MPICXX,
[
   AC_CHECK_PROGS(MPICXX, mpKCC mpCC mpig++ mpiCC hcp)
   test -z "$MPICXX" && AC_MSG_ERROR([no acceptable mpic++ found in \$PATH])
])dnl


dnl **********************************************************************
dnl * CASC_PROG_MPIF77 searches the PATH for an available MPI Fortran 77 
dnl * compiler wraparound.  It assigns the name to MPIF77.
dnl **********************************************************************

AC_DEFUN(CASC_PROG_MPIF77,
[
   AC_CHECK_PROGS(MPIF77, mpf77 mpxlf mpif77 mpixlf tmf77 hf77)
   test -z "$MPIF77" && AC_MSG_ERROR([no acceptable mpif77 found in \$PATH])
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

   # This follows the same procedur as CASC_CHECK_F77_PP, except it tests
   # $MPIF77 using a test program that includes MPI functions.

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
dnl * when one is chosen it will mess up the automatic choices for the
dnl * other two.  If the architecture is unknown, or if the needed MPI
dnl * settings for the current architecture are not known, then the naive
dnl * settings of MPILIBS="-lmpi" and MPILIBDIRS="-L/usr/local/mpi/lib"
dnl * are tested, and if they exist they are used, otherwise the MPILIB*
dnl * variables are left blank.  In the case of rs6000, the variable
dnl * MPIFLAGS is also set. 
dnl **********************************************************************
 
AC_DEFUN(CASC_SET_MPI,
[

   dnl * If called from within CASC_FIND_MPI, then the configure-line
   dnl * options will already exist.  This ifdef creates them otherwise.
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

      dnl * Set everything to known values
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
                     MPIINCLUDE="-I/usr/local/mpi/mpich/include \
                                 -I/usr/local/mpi/mpich/lib/solaris/ch_p4"
                  fi

                  if test -z "$casc_mpi_lib_dirs"; then
                     casc_mpi_lib_dirs="/usr/local/mpi/mpich/lib/solaris/ch_p4 \
                                       /usr/lib"
                  fi
            
               casc_mpi_libs="nsl socket mpi";;
               esac

            if test -z "$MPIINCLUDE"; then
               AC_CHECK_HEADER($casc_mpi_include_dir/mpi.h,
                               MPIINCLUDE="-I$casc_mpi_include_dir")
            fi
         ;;

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
dnl * MPICC is not set manually, then this macro will not work.
dnl *
dnl * One may question why these settings would need to be determined if
dnl * there already is mpicc available, and that is a valid question.  I
dnl * can think of a couple of reasons one may want to use these settings 
dnl * rather than using mpicc directly.  First, these settings allow you
dnl * to choose the C compiler you wish to use rather than using whatever
dnl * compiler is written into mpicc.  Also, the settings determined by
dnl * this macro should also work with C++ and Fortran compilers, so you
dnl * won't need to have mpiCC and mpif77 alongside mpicc.  This is
dnl * especially helpful on systems that don't have mpiCC.  The advantage
dnl * of this macro over CASC_SET_MPI is that this one doesn't require
dnl * a test of the machine type and thus will hopefully work on unknown
dnl * architectures.  The main disadvantage is that it relies on mpicc.
dnl *
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
               for mpi_dir in $withval; do
                  MPIINCLUDE="$MPIINCLUDE -I$withval"
               done; casc_user_chose_mpi=yes)

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
         [AC_CHECK_PROGS(MPICC, mpcc mpicc tmcc hcc)])

      if test -z "$MPICC"; then
         AC_MSG_WARN([no acceptable mpicc found in \$PATH])
         CASC_SET_MPI
         if test -z "$MPILIBS"; then
            AC_MSG_WARN([MPI not found - must set manually using --with flags])
         fi

      dnl * When $MPICC is there, run the automatic test
      dnl * here begins the hairy stuff

      else      

         changequote(, )dnl

         AC_MSG_CHECKING(for location of mpi.h)

         dnl * Create a minimal MPI program.  It will be compiled using
         dnl * $MPICC with verbose output.
         cat > mpconftest.c << EOF
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
         dnl * All of their outputs are stuffed into variable $casc_mpoutput

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

         dnl * little test to identify $CC as IBM's xlc
         echo "main() {}" > cc_conftest.c
         cc_output=`${CC-cc} -v -o cc_conftest cc_conftest.c 2>&1`
         xlc_p=`echo $cc_output | grep xlcentry`
         if test -n "$xlc_p"; then
            casc_compiler_is_xlc=yes
         fi 
         rm -rf cc_conftest*

         dnl * $MPICC might not produce '-lmpi', but we still need it.
         dnl * Add -lmpi to $casc_mplibs if it was never found
         if test "$casc_lmpi_exists" = "no"; then
            casc_mplibs="-lmpi"
         else
            casc_mplibs=
         fi

         casc_want_arg=

         dnl * Loop through every word in output to find possible flags.
         dnl * If the word is the absolute path of a library, it is added
         dnl * to $casc_flags.  Any "-llib", "-L/dir", "-R/dir" and
         dnl * "-I/dir" is kept.  If '-l', '-L', '-R', '-I', '-u', or '-Y'
         dnl * appears alone, then the next word is checked.  If the next
         dnl * word is another flag beginning with '-', then the first
         dnl * word is discarded.  If the next word is anything else, then
         dnl * the two words are coupled in the $casc_arg variable.
         dnl * "-binitfini:poe_remote_main" is a flag needed especially
         dnl * for IBM MPI, and it is added to kept if it is found.
         dnl * Any other word is discarded.  Also, after a word is found
         dnl * and kept once, it is discarded if it appears again

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
               -binitfini:poe_remote_main)
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
               casc_arg="casc_old_want_arg $casc_arg"
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

            dnl * Still inside the big for loop, we separate each flag
            dnl * into includes, libdirs, libs, flags
            if test -n "$casc_arg"; then
               case $casc_arg in
               -I*)

                  dnl * if the directory given in this flag contains mpi.h
                  dnl * then the flag is assigned to $MPIINCLUDE
                  if test -z "$MPIINCLUDE"; then
                     casc_cppflags="$casc_cppflags $casc_arg"
                     casc_include_dir=`echo "$casc_arg" | sed 's/-I//g'`

                     SAVE_CPPFLAGS="$CPPFLAGS"
                     CPPFLAGS="$casc_cppflags"
                     changequote([, ])dnl

                     unset ac_cv_header_mpi_h
                     AC_CHECK_HEADER(mpi.h,
                                     MPIINCLUDE="$casc_cppflags")

                     changequote(, )dnl
                     CPPFLAGS="$SAVE_CPPFLAGS"

                  else
                     casc_arg=
                  fi
               ;;
               -[LR]*)

                  dnl * These are the lib directory flags
                  casc_mplibdirs="$casc_mplibdirs $casc_arg"
               ;;
               -l* | /*)

                  dnl * These are the libraries
                  casc_mplibs="$casc_mplibs $casc_arg"
               ;;
               -binitfini:poe_remote_main)
                  if test "$casc_compiler_is_xlc" = "yes"; then
                     casc_mpflags="$casc_mpflags $casc_arg"
                  fi
               ;;
               *)
                  dnl * any other flag that has been kept goes here
                  casc_mpflags="$casc_mpflags $casc_arg"
               ;;
               esac

               dnl * Upcoming test needs $LIBS to contain the flags we've found
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
[#include "mpi.h"
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

         dnl * IBM MPI uses /usr/lpp/ppe.poe/libc.a instead of /lib/libc.a
         dnl * so we need to make sure that -L/lib is not part of the 
         dnl * linking line when we use IBM MPI.  This only appears in
         dnl * configure when CASC_FIND_MPI is called first.
         ifdef([AC_PROVIDE_CASC_FIND_F77LIBS], 
            if test -n "`echo $F77LIBFLAGS | grep '\-L/lib '`"; then
               if test -n "`echo $F77LIBFLAGS | grep xlf`"; then
                  F77LIBFLAGS=`echo $F77LIBFLAGS | sed 's/-L\/lib //g'`
               fi
            fi
         )

         if test -n "`echo $MPILIBS | grep pmpich`" &&
            test -z "`echo $MPILIBS | grep pthread`"; then
               LIBS_SAVE=$LIBS
               LIBS="$MPIINCLUDE $MPIFLAGS $MPILIBDIRS $MPILIBS -lpthread"
               AC_TRY_LINK(
                  ifelse(AC_LANG, CPLUSPLUS,

[#ifdef __cplusplus
extern "C"
#endif
])dnl
[#include "mpi.h"
], [int rank, size;
   int argc;
   char **argv;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Finalize();
],
                  MPILIBS="$MPILIBS -lpthread")
               LIBS=$LIBS_SAVE
         fi
          


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
