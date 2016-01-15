dnl macro CCA_BUNDLE_TAG([SEARCHPATH])
dnl --------------------------------------------------------------------
dnl Cause the variable CCA_BUNDLE_VERSION defined by configure.
dnl The value set will be developer unless the file CCA_BUNDLE_RELEASE is found.
dnl The default search path is $ac_aux_dir:. 
dnl If a SEARCHPATH is given it will be checked, then the default.
dnl This macro should be used after AC_CONFIG_AUX_DIR.
dnl The value if the file is found will be the first line of the file up to
dnl but not including the first whitespace.
dnl Side effects:
dnl substitutes CCA_BUNDLE_VERSION
dnl --------------------------------------------------------------------
AC_DEFUN([CCA_BUNDLE_TAG],
[
AC_MSG_CHECKING([CCA_BUNDLE_RELEASE])
CCA_BUNDLE_VERSION=developer
cbr_searchpath="$1:$ac_aux_dir:$srcdir:."
cbr_paths=`echo $cbr_searchpath|tr ":" " "`
for rdir in $cbr_paths ; do
	if test -d "$rdir"; then
		f=$rdir/CCA_BUNDLE_RELEASE
		if test -f "$f" ; then
			CCA_BUNDLE_VERSION=`cat $f | sed q`
			for rword in $CCA_BUNDLE_VERSION ; do
				CCA_BUNDLE_VERSION=$rword
				break
			done
			break
		fi
		f=$rdir/RELEASE
		if test -f "$f" ; then
			CCA_BUNDLE_VERSION=`cat $f | sed q`
			for rword in $CCA_BUNDLE_VERSION ; do
				CCA_BUNDLE_VERSION=$rword
				break
			done
			break
		fi
	fi
done
AC_SUBST(CCA_BUNDLE_VERSION)
AC_MSG_RESULT([ $CCA_BUNDLE_VERSION])
]
)

