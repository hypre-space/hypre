dnl **********************************************************************
dnl * HYPRE_GUESS_ARCH
dnl * First find the hostname and assigns it to an exported macro $HOSTNAME.
dnl * Guesses a one-word name for the current architecture, unless ARCH
dnl * has been preset.  This is an alternative to the built-in macro
dnl * AC_CANONICAL_HOST, which gives a three-word name.  Uses the utility
dnl * 'tarch', which is a Bourne shell script that should be in the same  
dnl * directory as the configure script.  If tarch is not present or if it
dnl * fails, ARCH is set to the value, if any, of shell variable HOSTTYPE,
dnl * otherwise ARCH is set to "unknown".
dnl **********************************************************************

AC_DEFUN(HYPRE_GUESS_ARCH,
[
   AC_MSG_CHECKING(the hostname)
   casc_hostname=hostname
   HOSTNAME="`$casc_hostname`"

   if test -z "$HOSTNAME" 
   then
   dnl * if $HOSTNAME is still empty, give it the value "unknown".
      HOSTNAME=unknown
      AC_MSG_WARN(hostname is unknown)
   else
      AC_MSG_RESULT($HOSTNAME)
   fi
   

   AC_MSG_CHECKING(the architecture)

   dnl * $ARCH could already be set in the environment or earlier in configure
   dnl * Use the preset value if it exists, otherwise go throug the procedure
   if test -z "$ARCH"; then

      dnl * configure searches for the tool "tarch".  It should be in the
      dnl * same directory as configure.in, but a couple of other places
      dnl * will be checked.  casc_tarch stores a relative path for "tarch".
      casc_tarch_dir=
      for casc_dir in $srcdir $srcdir/.. $srcdir/../.. $srcdir/config; do
         if test -f $casc_dir/tarch; then
            casc_tarch_dir=$casc_dir
            casc_tarch=$casc_tarch_dir/tarch
            break
         fi
      done

      dnl * if tarch was not found or doesn't work, try using env variable
      dnl * $HOSTTYPE
      if test -z "$casc_tarch_dir"; then
         AC_MSG_WARN(cannot find tarch, using \$HOSTTYPE as the architecture)
         ARCH=$HOSTTYPE
      else
         ARCH="`$casc_tarch`"

         if test -z "$ARCH" || test "$ARCH" = "unknown"; then
            ARCH=$HOSTTYPE
         fi
      fi

      dnl * if $ARCH is still empty, give it the value "unknown".
      if test -z "$ARCH"; then
         ARCH=unknown
         AC_MSG_WARN(architecture is unknown)
      else
         AC_MSG_RESULT($ARCH)
      fi    
   else
      AC_MSG_RESULT($ARCH)
   fi

   AC_SUBST(ARCH)
   AC_SUBST(HOSTNAME)

])dnl

dnl *********************************************************************
dnl * HYPRE_ADD_LIB(LIBRARY, FUNCTION, DIRECTORY-LIST[, PREFIX[, 
dnl *              ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]]])
dnl * checks whitespace-separated
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

AC_DEFUN(HYPRE_ADD_LIB,
[
   # define some macros to hopefully improve readability
   define([m_THESE_LIBS],[$4LIBS])
   define([m_THESE_LIBDIRS],[$4LIBDIRS])

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

   # ACTION-IF-NOT_FOUND for when the library is found nowhere
   ifelse([$6], , ,
      if test "$casc_lib_found" != "yes"; then
         [$6]
      fi
   )

   unset casc_lib_found

   undefine([m_THESE_LIBS])
   undefine([m_THESE_LIBDIRS])

])dnl

dnl ********************************************************************
dnl * HYPRE_FIND_MPI will determine the libraries, directories, and other
dnl * flags needed to compile and link programs with MPI function calls.
dnl * This macro runs tests on the script found by the CASC_PROG_MPICC
dnl * macro.  If there is no such mpicc-type script in the PATH
dnl * then this macro will not work.
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

AC_DEFUN(HYPRE_FIND_MPI,
[

   casc_find_mpi_cache_used=yes

   AC_MSG_CHECKING(for MPI)
   AC_CACHE_VAL(casc_cv_mpi_include, casc_find_mpi_cache_used=no)
   AC_CACHE_VAL(casc_cv_mpi_libs, casc_find_mpi_cache_used=no)
   AC_CACHE_VAL(casc_cv_mpi_lib_dirs, casc_find_mpi_cache_used=no)
   AC_CACHE_VAL(casc_cv_mpi_flags, casc_find_mpi_cache_used=no)
   AC_MSG_RESULT( )

   if test "$casc_find_mpi_cache_used" = "yes"; then
      AC_MSG_CHECKING(for location of mpi.h)
      MPIINCLUDE=$casc_cv_mpi_include
      AC_MSG_RESULT("\(cached\) $MPIINCLUDE")

      AC_MSG_CHECKING(for MPI library directories)
      MPILIBDIRS=$casc_cv_mpi_lib_dirs
      AC_MSG_RESULT("\(cached\) $MPILIBDIRS")

      AC_MSG_CHECKING(for MPI libraries)
      MPILIBS=$casc_cv_mpi_libs
      AC_MSG_RESULT("\(cached\) $MPILIBS")

      AC_MSG_CHECKING(for other MPI-related flags)
      MPIFLAGS=$casc_cv_mpi_flags
      AC_MSG_RESULT("\(cached\) $MPIFLAGS")
   else
   

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

      dnl * --with-mpi-flags only adds to automatic selections, 
      dnl * does not override

      AC_ARG_WITH(mpi-flags,
[  --with-mpi-flags=FLAGS  FLAGS is space-separated list of whatever flags other
                          than -l and -L are needed to link with mpi libraries],
                          MPIFLAGS=$withval)

         AC_CHECK_PROGS(MPICC, mpcc mpicc tmcc hcc)

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
            dnl * $CC with verbose output.
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
            dnl * All of their outputs are stuffed into variable
            dnl * $casc_mpoutput

            for casc_command in "$MPICC -show"\
                                "$MPICC -v"\
                                "$MPICC -#"\
                                "$MPICC"; do

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
            dnl * for IBM MPI, and it is always kept if it is found.
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

                  dnl * Upcoming test needs $LIBS to contain the flags 
                  dnl * we've found
                  LIBS_SAVE=$LIBS
                  LIBS="$MPIINCLUDE $casc_mpflags $casc_mplibdirs $casc_mplibs"

                  if test -n "`echo $LIBS | grep '\-R/'`"; then
                     LIBS=`echo $LIBS | sed 's/-R\//-R \//'`
                  fi

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
      

      AC_CACHE_VAL(casc_cv_mpi_include, casc_cv_mpi_include=$MPIINCLUDE)
      AC_CACHE_VAL(casc_cv_mpi_lib_dirs, casc_cv_mpi_lib_dirs=$MPILIBDIRS)
      AC_CACHE_VAL(casc_cv_mpi_libs, casc_cv_mpi_libs=$MPILIBS)
      AC_CACHE_VAL(casc_cv_mpi_flags, casc_cv_mpi_flags=$MPIFLAGS)
   fi

   AC_SUBST(MPIINCLUDE)
   AC_SUBST(MPILIBDIRS)
   AC_SUBST(MPILIBS)
   AC_SUBST(MPIFLAGS)

])dnl



