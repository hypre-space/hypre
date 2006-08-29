AC_DEFUN([_STAR_RESTFP_FIX_F77],
   [case $build_os in
    darwin*)
      FLIBS="-L/usr/lib $FLIBS"
    ;;
    esac
])

AC_DEFUN([_STAR_RESTFP_FIX_FC],
   [case $build_os in
    darwin*)
      FCLIBS="-L/usr/lib $FCLIBS"
    ;;
    esac
])
