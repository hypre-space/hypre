
dnl 
dnl @synopsis LLNL_SORT_FLIBS
dnl 
dnl With certain Fortran compilers, the FLIBS macro can be out of order.
dnl This macros moves all the arguments beginning with "-l" at the end
dnl but does not alter the relative ordering of "-l" arguments and non-"-l" 
dnl arguments; otherwise,
dnl   If the answer is yes, 
dnl     it defines AR_CXX=$CXX, ARFLAGS_CXX=-xar, and RANLIB_CXX=echo
dnl   otherwise AR_CXX=ar, ARFLAGS_CXX=cuv, RANLIB_CXX=ranlib
dnl
dnl @version 
dnl @author Gary Kumfert <kumfert1@llnl.gov>
dnl
AC_DEFUN([LLNL_SORT_FLIBS],
[AC_REQUIRE([LLNL_F77_LIBRARY_LDFLAGS])dnl
flibs1=
flibs2=
for arg in $FLIBS; do
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
    for f in $flibs1; do
      if test x$arg1 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      flibs1="$flibs1 $arg1"
    fi
  fi
  if test -n "$arg2"; then
    exists=false
    for f in $flibs2; do
      if test x$arg2 = x$f; then 
        exists=true
      fi
    done
    if $exists; then
      :
    else
      flibs2="$flibs2 $arg2"
    fi
  fi
done
FLIBS="$flibs1 $flibs2"
AC_SUBST(FLIBS)
])
