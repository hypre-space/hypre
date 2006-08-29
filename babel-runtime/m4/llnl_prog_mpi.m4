#
# LLNL_PROG_MPI
#
# Let user specify if they want MPI or not
#
# Input env vars: MPI_PREFIX
# Configure flags:  --with-mpi
#
#
# lang 	main script	raw compiler	compile flags	link flags
#  C  	MPI_CC 		MPI_CC_CC	MPI_CC_CFLAGS	MPI_CC_LDFLAGS
#  C++  MPI_CXX		MPI_CXX_CXX	MPI_CXX_CFLAGS	MPI_CXX_LDFLAGS

AC_DEFUN([LLNL_PROG_MPI],
[
  AC_MSG_CHECKING([whether to probe for related MPI compilers])
  AC_ARG_VAR([MPI_PREFIX],[Root directory where MPI's bin/ include/ and lib/ dirs are installed.])
  llnl_cv_with_mpi=no
  test "${MPI_PREFIX+set}" = set && llnl_cv_with_mpi=yes
  AC_ARG_WITH([mpi],
    AS_HELP_STRING([--with-mpi@<:@=eprefix@:>@],[(experimental) MPI compiler locations @<:@default=no@:>@]),
    [ case $withval in
         no) llnl_cv_with_mpi=no ;;
         yes) llnl_cv_with_mpi=yes ;;
         *) llnl_cv_with_mpi=yes; mpi_prefix="$withval" ;; 
      esac; ],) #end AC_ARG_WITH
  AC_MSG_RESULT([$llnl_cv_with_mpi])
                       
  # now do all the testing for each configured language
  if test "$llnl_cv_with_mpi" = yes; then 
    if test "${MPI_PREFIX+set}" = set ; then
      mpi_searchpath=${MPI_PREFIX}/bin:${PATH}
    else
      mpi_searchpath=${PATH}
    fi;

    LLNL_PROG_MPICC

    if test "$MPI_CC" != skip; then
      AC_MSG_CHECKING([for MPI version])
      cat <<\_ACEOF >conftest.c
#include "mpi.h"
MPI_VERSION AAAAA
MPI_SUBVERSION BBBBB
MPICH_NAME CCCCC
LAM_MPI DDDDD
_ACEOF
      ${MPI_CC} -E conftest.c > conftest.s	
      MPI_VERSION=`grep AAAAA conftest.s | sed -e 's/\ .*$//;'`
      MPI_SUBVERSION=`grep BBBBB conftest.s | sed -e 's/\ .*$//;'`
      AC_MSG_RESULT([$MPI_VERSION.$MPI_SUBVERSION])
      MPI_VENDOR=unknown
      AC_MSG_CHECKING([for MPI vendor])
      if grep CCCCC conftest.s | grep 1 > /dev/null; then
         MPI_VENDOR=mpich
      elif grep DDDDD conftest.s | grep 1 > /dev/null; then
         MPI_VENDOR=lam
      fi;
      AC_MSG_RESULT([$MPI_VENDOR])
      AC_SUBST([MPI_VENDOR])
      AC_SUBST([MPI_VERSION])
      AC_SUBST([MPI_SUBVERSION])
      rm -f conftest.c conftest.s
    fi
 
    LLNL_PROG_MPICXX
    LLNL_PROG_MPIF77
    LLNL_PROG_MPIFC
  fi
])

#the strategy for all of these is the same
#  if ( the language is enabled ) { 
#     1. Find what MPI_* compiler front-end to use
#     2. Ensure the MPI_* can compile and link a MPI test code
#     3. Figure out what flag gets MPI_* to show its flags
#     4. Capture the underlying compiler, compile flags, and link flags
#     5. if underlying compiler is different from user compiler
#          verify that using underlying compiler, compile flags, and link 
#	   flags by hand will compile and link MPI test code.
#     6. verify that running USER's compiler, MPI's compile flags, and MPI's
#        link flags by hand will compile and link MPI test code.

AC_DEFUN([LLNL_PROG_MPICC],[
  dnl  NOTE:  ac_ct_$1 is set in autoconf standard macros for checking programs 
  dnl         and tools.  It seems sufficient to test if a compiler has been configured for
  if test "x$CC" != "x"; then 

    #
    # 1. Find what MPI_* compiler front-end to use
    #
    AC_ARG_VAR([MPI_CC], [Default MPI-enabled C compiler.])
    if test "x$MPI_CC" = xskip; then
      AC_MSG_NOTICE([Skipping MPI-enabled C compiler.])
    else
    if test "${MPI_CC+set}" != set; then 
      AC_MSG_NOTICE([Scanning for MPI-enabled C compiler.])
      AC_CHECK_PROGS([MPI_CC], [mpi$CC mp$CC MPI$CC MP$CC mpicc hcc mpcc mpcc_r mpxlc cmpicc], [none], [$mpi_searchpath])
    fi  
    AC_SUBST([MPI_CC])
    if test "$MPI_CC" = none; then
      AC_MSG_FAILURE([cannot find a MPI_CC compiler in $mpi_searchpath])
    fi

    #
    # 2. Ensure the MPI_* can compile and link a test MPI code
    #
    AC_CACHE_CHECK([if ($MPI_CC) compiles and links sample MPI code],
	           [llnl_cv_mpi_cc_works],
		   [llnl_cv_mpi_cc_works=no;
		    AC_LANG_PUSH(C)
		    user_CC="$CC"
		    CC="$MPI_CC"
		    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
@%:@include "mpi.h"
int argc2=1;
char *argv2 @<:@ @:>@ = {"test"};
char **argv3=argv2; /*xlC workaround*/
]], [[MPI_Init(&argc2,&argv3)]])],[llnl_cv_mpi_cc_works=yes])
 		   CC="$user_CC"
                   AC_LANG_POP
    ])
    if test "$llnl_cv_mpi_cc_works" = no; then
      AC_MSG_FAILURE([Cannot compile with MPI_CC compiler ($MPI_CC)])
    fi
      
    #
    # 3. Figure out what flag gets MPI_* to show its flags
    # 
    possible_verbose_flags="-compile-info -compile_info -link-info -link_info -show -showme -v"
    if test "$llnl_cv_mpi_cc_works" = yes; then 
      user_CC="$CC"
      user_CFLAGS="$CFLAGS"
      verbose_flag=no
      verbose_flag_works=no
      for flag in $possible_verbose_flags; do 
         if test "$verbose_flag_works" = no; then
           case $flag in 
              -*) AC_MSG_CHECKING([for verbose output with ($MPI_CC $flag)])
	          candidate_verbose_flag="$flag"
		  ;;
  	       *) AC_MSG_CHECKING([for verbose output with $MPI_CC -$flag])
	          candidate_verbose_flag="-$flag"
	  	  ;;
	   esac
           CC="$MPI_CC"
           CFLAGS="$user_CFLAGS $candidate_verbose_flag"
  	   AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])
	   (eval $CC $CFLAGS conftest.$ac_ext) > conf.out 2>&1 
           ac_status=$?
           if test "$ac_status" = 0 ; then
             if grep -- ' -[[IDLl]]' conf.out > /dev/null ; then  
  	       verbose_flag="$candidate_verbose_flag"
	       verbose_flag_works=yes
	     fi
           fi
           AC_MSG_RESULT([$verbose_flag_works])
         fi 
       done
       CC="$user_CC"
       CFLAGS="$user_CFLAGS"
    fi

    #
    # 4. Capture the underlying compiler, compile flags, and link flags
    #
    if test "$verbose_flag_works" = yes; then 
	user_CC="$CC"
	user_CFLAGS="$CFLAGS"
	user_LDFLAGS="$LDFLAGS"
        user_LIBS="$LIBS"
	CC="$MPI_CC"
	if test "$verbose_flag" = "-compile_info"; then 
 	  CFLAGS="-compile_info"
  	  LIBS="-link_info"
        elif test "$verbose_flag" = "-compile-info"; then
	  CFLAGS="-compile-info"
          LIBS="-link-info"
        else
 	  CFLAGS="$CFLAGS $verbose_flag"
          LIBS="$LDFLAGS $verbose_flag"
        fi

	# first the compiler
	AC_MSG_CHECKING([what raw C compiler flags ($MPI_CC) invokes])
        llnl_mpi_cc_c_v_output=`$MPI_CC -c $CFLAGS conftest.$ac_ext 2>&1`
        for i in $llnl_mpi_cc_c_v_output; do MPI_CC_CC=$i; break; done;
	AC_MSG_RESULT([$MPI_CC_CC])
	if test "$MPI_CC_CC" != "$user_CC"; then 
 	  AC_MSG_WARN([your CC ($user_CC) may not be the same as $MPI_CC's CC ($MPI_CC_CC)])
 	  AC_MSG_WARN([please make sure these are compatible, or consider changing  your CC])
	fi 

        # now compile flags
	AC_MSG_CHECKING([what compile flags ($MPI_CC) passes to ($MPI_CC_CC)])	  
	AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])
	MPI_CC_CFLAGS= 
	for i in $llnl_mpi_cc_c_v_output; do 
          case $i in 
  	    -[[DIUb]]*)
	       MPI_CC_CFLAGS="$MPI_CC_CFLAGS $i"
               ;;
          esac
	done

	# now prune things in user_CFLAGS out of MPI_CC_CFLAGS
	for i in $user_CFLAGS; do
	  MPI_CC_CFLAGS=`echo $MPI_CC_CFLAGS | sed -e "s,$i,,"`
        done
	AC_MSG_RESULT([$MPI_CC_CFLAGS])

	# now the link flags
	AC_MSG_CHECKING([what link flags ($MPI_CC) passes to ($MPI_CC_CC)])
        llnl_mpi_cc_link_v_output=`$MPI_CC -o conftest$ac_exeext $CFLAGS $CPPFLAGS $LDFLAGS conftest.$ac_ext $LIBS 2>&1`
	MPI_CC_LDFLAGS=
	for i in $llnl_mpi_cc_link_v_output; do 
          case $i in 
  	     [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl*)
	        MPI_CC_LDFLAGS="$MPI_CC_LDFLAGS $i"
	     ;;
          esac
	done

	# now prune things in user_LDFLAGS out of MPI_CC_LDFLAGS
	for i in $user_LDFLAGS; do
	   MPI_CC_LDFLAGS=`echo $MPI_CC_LDFLAGS | sed -e "s,$i,,"`
        done
	AC_MSG_RESULT([$MPI_CC_LDFLAGS])

	#
	#  5. if underlying compiler is different from user compiler
	#     verify that running underlying compiler, compile flags, and link 
	#     flags by hand will compile and link MPI test code.
	#
        other_works=no;
        if test "$MPI_CC_CC" != "$user_CC"; then 
 	  AC_MSG_CHECKING([if ($MPI_CC_CC \$MPI_CC_CFLAGS \$MPI_CC_LDFLAGS) compiles and links same MPI sample code ($MPI_CC) did])
	  CC="$MPI_CC_CC"
	  CFLAGS="$user_CFLAGS $MPI_CC_CFLAGS"
	  LIBS="$user_LDFLAGS $MPI_CC_LDFLAGS"
          AC_LINK_IFELSE([AC_LANG_PROGRAM([[
@%:@include "mpi.h"
int argc2=1;
char *argv2 @<:@ @:>@ = {"test"};
char **argv3=argv2; /*xlC workaround*/
]], [[MPI_Init(&argc2,&argv3)]])],[other_works=yes])
	  AC_MSG_RESULT([$other_works])
	fi

	#
	#     6. verify that running USER's compiler, MPI's compile flags, and MPI's
	#        link flags by hand will compile and link MPI test code.
	#
	AC_MSG_CHECKING([if ($user_CC \$MPI_CC_CFLAGS \$MPI_CC_LDFLAGS) compiles and links same MPI sample code ($MPI_CC) did])
	mpi_flags_work=no
	CC="$user_CC"
	CFLAGS="$user_CFLAGS $MPI_CC_CFLAGS"
	LIBS="$user_LDFLAGS $MPI_CC_LDFLAGS"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
@%:@include "mpi.h"
int argc2=1;
char *argv2 @<:@ @:>@ = {"test"};
char **argv3=argv2; /*xlC workaround*/
]], [[MPI_Init(&argc2,&argv3)]])],[mpi_flags_work=yes])
        AC_MSG_RESULT([$mpi_flags_work])

        if test "$mpi_flags_work" = no -a "$other_works" = yes ; then
           AC_MSG_WARN([flags that work with $MPI_CC's compiler ($MPI_CC_CC), do not work with yours ($user_CC)])
           AC_MSG_WARN([recommend either an MPI built with $user_CC, or switch to the $MPI_CC_CC compiler])
 	  AC_MSG_FAILURE([cannot infer a working set of MPI flags for CC=$user_CC.])
	elif test "$mpi_flags_work" = no; then
 	  AC_MSG_FAILURE([cannot infer a working set of MPI flags for CC=$user_CC.])
	fi;

	AC_SUBST([MPI_CC_CFLAGS])
	AC_SUBST([MPI_CC_LDFLAGS])
	CC="$user_CC"
	CFLAGS="$user_CFLAGS"
	LDFLAGS="$user_LDFLAGS"
	LIBS="$user_LIBS"
    fi #test $verbose_flag_works = yes
    fi
  fi #end if ${ac_ct_CC+set}" = set; 
])

############################################################

AC_DEFUN([LLNL_PROG_MPICXX],[
  if test "x$CXX" != "x" ; then

    #
    # 1. Find what MPI_* compiler front-end to use
    #
    AC_LANG_PUSH(C++)dnl     
    AC_ARG_VAR([MPI_CXX], [Default MPI-enabled C++ compiler.])
    if test "x$MPI_CXX" = xskip; then 
      AC_MSG_NOTICE([Skipping MPI-enabled C+ compiler.])
    else
    if test "${MPI_CXX+set}" != set; then 
      AC_MSG_NOTICE([scanning for MPI-enabled C++ compiler.])
      AC_CHECK_PROGS([MPI_CXX], [mpi$CXX mp$CXX MPI$CXX MP$CXX mpic++ mpc++ mpicxx hcxx mpcxx mpcxx_r mpxlC cmpicxx], [none], $mpi_searchpath)
    fi 
   AC_SUBST([MPI_CXX])
   if test "$MPI_CXX" = none; then
     AC_MSG_FAILURE([cannot find a MPI_CXX compiler in '$mpi_searchpath'])
   fi


    #
    # 2. Ensure the MPI_* can compile and link a test MPI code
    #
    AC_CACHE_CHECK([if ($MPI_CXX) compiles and links sample MPI code],
	           [llnl_cv_mpi_cxx_works],
		   [llnl_cv_mpi_cxx_works=no;
		    user_CXX="$CXX"
		    CXX="$MPI_CXX"
		    AC_LANG_PUSH(C++)
		    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
@%:@include "mpi.h"
int argc2=1;
char *argv2 @<:@ @:>@ = {"test"};
char **argv3=argv2; /*xlC workaround*/
]], [[MPI_Init(&argc2,&argv3)]])],[llnl_cv_mpi_cxx_works=yes])
 		   CXX="$user_CXX"
		   AC_LANG_POP
    ])
    if test "$llnl_cv_mpi_cxx_works" = no; then
      AC_MSG_FAILURE([Cannot compile with MPI_CXX compiler ($MPI_CXX)])
    fi

#     #
#     # 2.a. Ensure the MPI_* can compile and link a test MPI code
#     #
#     AC_CACHE_CHECK([if ($MPI_CXX) compiles and links 2nd sample MPI code],
# 	           [llnl_cv_mpi_cxx2_works],
# 		   [llnl_cv_mpi_cxx2_works=no;
# 		    user_CXX="$CXX"
# 		    CXX="$MPI_CXX"
# 		    AC_LINK_IFELSE([AC_LANG_PROGRAM([[
# @%:@include "mpi.h"
# ]], [[MPI::Init()]])],[llnl_cv_mpi_cxx_works2=yes])
#  		   CXX="$user_CXX"
#     ])
#     if test "$llnl_cv_mpi_cxx_works" = no; then
#       AC_MSG_FAILURE([Cannot compile 2nd with MPI_CXX compiler ($MPI_CXX)])
#     fi
  
    #
    # 3. Figure out what flag gets MPI_* to show its flags
    # 
    possible_verbose_flags="$candidate_verbose_flag -compile_info -link_info -show -showme -v"
    if test "$llnl_cv_mpi_cxx_works" = yes; then 
      user_CXX="$CXX"
      user_CXXFLAGS="$CXXFLAGS"
      verbose_flag=no
      verbose_flag_works=no
      for flag in $possible_verbose_flags; do 
         if test "$verbose_flag_works" = no; then
           case $flag in 
              -*) AC_MSG_CHECKING([for verbose output with ($MPI_CXX $flag)])
	          candidate_verbose_flag="$flag"
		  ;;
  	       *) AC_MSG_CHECKING([for verbose output with $MPI_CXX -$flag])
	          candidate_verbose_flag="-$flag"
	  	  ;;
	   esac
           CXX="$MPI_CXX"
           CXXFLAGS="$user_CXXFLAGS $candidate_verbose_flag"
  	   AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])
	   (eval $CXX $CXXFLAGS conftest.$ac_ext) > conf.out 2>&1 
           ac_status=$?
           if test "$ac_status" = 0 ; then
             if grep -- ' -[[IDLl]]' conf.out > /dev/null ; then  
  	       verbose_flag="$candidate_verbose_flag"
	       verbose_flag_works=yes
	     fi
           fi
           AC_MSG_RESULT([$verbose_flag_works])
         fi 
       done
       CXX="$user_CXX"
       CXXFLAGS="$user_CXXFLAGS"
    fi

    #
    # 4. Capture the underlying compiler, compile flags, and link flags
    #
    if test "$verbose_flag_works" = yes; then 
	user_CXX="$CXX"
	user_CXXFLAGS="$CXXFLAGS"
        user_CXXLIBS="$LIBS"
	CXX="$MPI_CXX"
	if test "$verbose_flag" = "-compile_info"; then 
 	  CXXFLAGS="-compile_info"
  	  CXXLIBS="-link_info"
        elif test "$verbose_flag" = "-compile-info"; then
	  CXXFLAGS="-compile-info"
          CXXLIBS="-link-info"
         else
 	  CXXFLAGS="$CXXFLAGS $verbose_flag"
          CXXLIBS="$CXXLIBS $verbose_flag"
        fi
        
	# first the compiler
	AC_MSG_CHECKING([what raw C++ compiler flags ($MPI_CXX) invokes])
        llnl_mpi_cxx_c_v_output=`$MPI_CXX -c $CXXFLAGS conftest.$ac_ext 2>&1`
        for i in $llnl_mpi_cxx_c_v_output; do MPI_CXX_CXX=$i; break; done;
	AC_MSG_RESULT([$MPI_CXX_CXX])
	if test "$MPI_CXX_CXX" != "$user_CXX"; then 
 	  AC_MSG_WARN([your CXX ($user_CXX) may not be the same as $MPI_CXX's CXX ($MPI_CXX_CXX)])
 	  AC_MSG_WARN([please make sure these are compatible, or consider changing  your CXX])
	fi 

        # now compile flags
	AC_MSG_CHECKING([what compile flags ($MPI_CXX) passes to ($user_CXX)])	  
	AC_LANG_CONFTEST([AC_LANG_PROGRAM([])])
        llnl_mpi_cxx_c_v_output=`$MPI_CXX -c $CXXFLAGS conftest.$ac_ext 2>&1`
	MPI_CXX_CFLAGS= 
	for i in $llnl_mpi_cxx_c_v_output; do 
          case $i in 
  	    -[[DIUb]]*)
	       MPI_CXX_CFLAGS="$MPI_CXX_CFLAGS $i"
               ;;
          esac
	done

	# now prune things in user_CXXFLAGS out of MPI_CXX_CFLAGS
	for i in $user_CXXFLAGS; do
	  MPI_CXX_CFLAGS=`echo $MPI_CXX_CFLAGS | sed -e "s,$i,,"`
        done
	AC_MSG_RESULT([$MPI_CXX_CFLAGS])
	AC_MSG_CHECKING([what link flags ($MPI_CXX) passes to ($user_CXX)])
        llnl_mpi_cxx_link_v_output=`$MPI_CXX -o conftest$ac_exeext $CXXFLAGS $CPPFLAGS conftest.$ac_ext $CXXLIBS 2>&1`
        MPI_CXX_LDFLAGS=
	for i in $llnl_mpi_cxx_link_v_output; do 
          case $i in 
  	     [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
	        MPI_CXX_LDFLAGS="$MPI_CXX_LDFLAGS $i"
	     ;;
          esac
	done

	# now prune things in user_CXXLIBS out of MPI_CXX_LDFLAGS
	for i in $user_CXXLIBS; do
	   MPI_CXX_LDFLAGS=`echo $MPI_CXX_LDFLAGS | sed -e "s,$i,,"`
        done
	AC_MSG_RESULT([$MPI_CXX_LDFLAGS])

	#
	#  5. if underlying compiler is different from user compiler
	#     verify that running underlying compiler, compile flags, and link 
	#     flags by hand will compile and link test app.
	#
        other_works=no;
        if test "$MPI_CXX_CXX" != "$user_CXX"; then 
 	  AC_MSG_CHECKING([if ($MPI_CXX_CXX \$MPI_CXX_CFLAGS \$MPI_CXX_LDFLAGS) compiles and links same MPI code ($MPI_CXX) did])
	  CXX="$MPI_CXX_CXX"
	  CXXFLAGS="$user_CXXFLAGS $MPI_CXX_CFLAGS"
	  LIBS="$user_CXXLIBS $MPI_CXX_LDFLAGS"
          AC_LINK_IFELSE([AC_LANG_PROGRAM([[
@%:@include "mpi.h"
int argc2=1;
char *argv2 @<:@ @:>@ = {"test"};
char **argv3=argv2; /*xlC workaround*/
]], [[MPI_Init(&argc2,&argv3)]])],[other_works=yes])
	  AC_MSG_RESULT([$other_works])
	fi

	#
	#     6. verify that running USER's compiler, MPI's compile flags, and MPI's
	#        link flags by hand will compile and link test app
	#
	AC_MSG_CHECKING([if ($user_CXX \$MPI_CXX_CFLAGS \$MPI_CXX_LDFLAG) compiles and links same MPI code ($MPI_CXX) did])
	mpi_flags_work=no
	CXX="$user_CXX"
	CXXFLAGS="$user_CXXFLAGS $MPI_CXX_CFLAGS"
	LIBS="$user_CXXLIBS $MPI_CXX_LDFLAGS"
        AC_LINK_IFELSE([AC_LANG_PROGRAM([[
@%:@include "mpi.h"
int argc2=1;
char *argv2 @<:@ @:>@ = {"test"};
char **argv3=argv2; /*xlC workaround*/
]], [[MPI_Init(&argc2,&argv3)]])],[mpi_flags_work=yes])
        AC_MSG_RESULT([$mpi_flags_work])


        if test "$mpi_flags_work" = no -a "$other_works" = yes ; then
           AC_MSG_WARN([flags that work with $MPI_CXX's compiler ($MPI_CXX_CXX), do not work with yours ($user_CXX)])
           AC_MSG_WARN([recommend either an MPI built with $user_CXX, or switch to the $MPI_CXX_CXX compiler])
 	  AC_MSG_FAILURE([cannot infer a working set of MPI flags for CXX=$user_CXX.])
	elif test "$mpi_flags_work" = no; then
 	  AC_MSG_FAILURE([cannot infer a working set of MPI flags for CXX=$user_CXX.])
	fi;

	AC_SUBST([MPI_CXX_CFLAGS])
        AC_SUBST([MPI_CXX_LDFLAGS])
	CXX="$user_CXX"
	CXXFLAGS="$user_CXXFLAGS"
        CXXLIBS="$user_CXXLIBS"
    fi #test $verbose_flag_works = yes
    fi
    AC_LANG_POP
  fi #end if ${ac_ct_CXX+set}" = set 
])


AC_DEFUN([LLNL_PROG_MPIF77],[
  #if test "x$enable_fortran77" = xyes; then #alternate test?
  if test "${ac_ct_F77+set}" = set; then
    AC_MSG_NOTICE([Sorry, test for MPI-enabled F77 compiler not yet implemented.])
  fi
])


AC_DEFUN([LLNL_PROG_MPIFC],[
  #if test "x$enable_fortran90" = xyes; then #alternate test?
  if test "${ac_ct_FC+set}" = set; then
    AC_MSG_NOTICE([Sorry, test for MPI-enabled F90 compiler not yet implemented.])
  fi
])
