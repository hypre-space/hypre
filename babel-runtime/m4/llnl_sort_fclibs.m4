
dnl 
dnl @synopsis LLNL_SORT_F90LIBS
dnl 
dnl With certain Fortran compilers, the libs macro can be out of order.
dnl This macros moves all the arguments beginning with "-l" at the end
dnl but does not alter the relative ordering of "-l" arguments and non-"-l" 
dnl arguments; otherwise, 
dnl   If the answer is yes, 
dnl     it defines AR_CXX=$CXX, ARFLAGS_CXX=-xar, and RANLIB_CXX=echo
dnl   otherwise AR_CXX=ar, ARFLAGS_CXX=cuv, RANLIB_CXX=ranlib
dnl
dnl @version 
dnl @author 
dnl
dnl Note:  Clone of F77 version.
dnl

AC_DEFUN([LLNL_SORT_FCLIBS],
[AC_REQUIRE([LLNL_F90_LIBRARY_LDFLAGS])dnl
f90libs1=
f90libs2=
for arg in $FCLIBS; do
  arg1=
  arg2=
  case "$arg" in 
    -l*)
      arg2=$arg
      ;;
    /*.a)
      arg1=-L`dirname $arg`
      arg2=`basename $arg .a`
      arg2=`echo $arg2 | sed 's/^lib/-l'/'`
      ;;
    /*.so)
      arg1=-L`dirname $arg`
      arg2=`basename $arg .so`
      arg2=`echo $arg2 | sed 's/^lib/-l'/'`
      ;;
    *)
      arg1=$arg
      ;;
  esac; 
  if test -n "$arg1"; then
    exists=false
    for f in $f90libs1; do
      if test x$arg1 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      f90libs1="$f90libs1 $arg1"
    fi
  fi
  if test -n "$arg2"; then
    exists=false
    for f in $f90libs2; do
      if test x$arg2 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      f90libs2="$f90libs2 $arg2"
    fi
  fi
done
FCLIBS="$f90libs1 $f90libs2"
AC_SUBST(FCLIBS)
])
