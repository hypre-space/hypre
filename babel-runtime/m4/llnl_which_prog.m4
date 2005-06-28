
dnl @synopsis LLNL_WHICH_PROG(WHICH_VARIABLE, [altpath])
dnl  
dnl Searches for VARIABLE in PATH.   sets WHICH_VARIABLE to the absolute path of VARIABLE.
dnl
dnl @author Gary Kumfert

AC_DEFUN([LLNL_WHICH_PROG],
[
which_progvar=$1
progvar=`echo $which_progvar | sed -e 's/^WHICH_//'`
eval "$which_progvar="
explicit_path=$2
mypath=$PATH
if test -n "$explicit_path"; then
  mypath=${explicit_path}:${mypath}
fi
#progval=`eval echo \\$$progvar`
#now only get the first
progval=
for k in `eval echo \\$$progvar`; do progval=$k; break; done;
if test -n "$progval"; then
  case $progval in 
    /*)#already absolute path... just verify it exists
      if test -e "$progval"; then
        eval "$which_progvar=$progval"
      fi
      ;;
    *)# marching through PATH
      cpath=`echo $mypath | sed -e 's/:/ /g'`
      for i in $cpath; do 
        if test -e "$i/$progval"; then
          eval "$which_progvar=$i/$progval"
          break;
        fi;
      done;
      ;;
  esac  
fi
which_progval=`eval echo \\$$which_progvar`
$1=$which_progval
AC_SUBST($1)
])


