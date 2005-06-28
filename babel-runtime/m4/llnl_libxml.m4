# LLNL_FIND_LIBXML
#
#
AC_DEFUN([VERSION_TO_NUMBER],
[`$1 | sed -e 's/libxml //' | awk 'BEGIN { FS = "."; } { printf "%d", ([$]1 * 1000 + [$]2) * 1000 + [$]3;}'`])

AC_DEFUN([LLNL_LIBXML_CONFIG],
[LIBXML_REQUIRED_VERSION=2.4.0

 AC_ARG_WITH([libxml2],
	[AS_HELP_STRING(--with-libxml2@<:@=DIR@:>@,use libxml2 in @<:@DIR@:>@ (YES if found))],,[withval=maybe])

dnl if the user explicitly asked for no libxml2 (Babel requires it)
if test "$withval" != "no"; then
    dnl find xml2-config program
    XML2_CONFIG="no"
    if test "$withval" != "yes" && test "$withval" != "maybe" ; then
	XML2_CONFIG_PATH="$withval/bin"
	AC_PATH_PROG(XML2_CONFIG, xml2-config,"no", $XML2_CONFIG_PATH)
    else
	XML2_CONFIG_PATH=$PATH
	AC_PATH_PROG(XML2_CONFIG, xml2-config,"no", $XML2_CONFIG_PATH)
    fi

    dnl we can't do anything without xml2-config
    if test "$XML2_CONFIG" = "no"; then
	withval="no"
    else
	withval=`$XML2_CONFIG --prefix`
    fi

    dnl if withval still maybe then we have failed
    if test "$withval" = "maybe"; then
	withval="no"
    fi
fi

if test "$withval" = "no"; then
    XML2_CONFIG="no"
    AC_MSG_WARN(You need libxml2 $LIBXML_REQUIRED_VERSION (or later) for dynamic loading support in Babel)
else
    AC_SUBST(LIBXML_REQUIRED_VERSION)
    AC_MSG_CHECKING(for libxml libraries >= $LIBXML_REQUIRED_VERSION)


   dnl
   dnl test version and init our variables
   dnl

   vers=VERSION_TO_NUMBER([$XML2_CONFIG --version])
   XML2_VERSION=`$XML2_CONFIG --version`

   if test "$vers" -ge VERSION_TO_NUMBER([echo $LIBXML_REQUIRED_VERSION]); then
	AC_MSG_RESULT(found version $XML2_VERSION)
	if $XML2_CONFIG --libtool-libs | grep "^Usage:" 2>&1 > /dev/null; then
	    LIBXML2_LIB="`$XML2_CONFIG --libs`"
	else
	    LIBXML2_LIB="`$XML2_CONFIG --libtool-libs`"
	fi
	
	LIBXML2_CFLAGS="`$XML2_CONFIG --cflags`"
	AC_DEFINE(HAVE_LIBXML2,[1],[Libxml2 support included])
	AC_DEFINE_UNQUOTED(LIBXML2_VERSION, $vers, [Version of libxml2 installed])

	AC_SUBST(LIBXML2_LIB)
        AC_SUBST(LIBXML2_CFLAGS)
   else
	AC_MSG_WARN(You need libxml2 $LIBXML_REQUIRED_VERSION (or later) for dynamic loading support in Babel)
	XML2_CONFIG="no"
	unset XML2_VERSION
   fi

fi
])
