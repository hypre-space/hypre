## Process this file with automake to create Makefile.in

include babel.make

if STATICONLY
 lib_LIBRARIES = libbHYPRE.a
 libbHYPRE_a_SOURCES = $(FILES)
 OBJECTS = libbHYPRE_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libbHYPRE.la
 libbHYPRE_la_SOURCES = $(FILES)
 OBJECTS = libbHYPRE_la_OBJECTS
 libbHYPRE_la_LDFLAGS = -avoid-version
endif !STATICONLY

BABEL_RUNTIME_INCLUDES=\
 -I$(top_srcdir)/babel-runtime/sidl\
 -I$(top_builddir)/babel-runtime/sidl

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/seq_mv\
 ${BABEL_RUNTIME_INCLUDES}

EXTRA_DIST = 

include_HEADERS = ${STUBHDRS} ${IORHDRS} ${IMPLHDRS}

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS} \
 bHYPRE_Extensions.c

$(OBJECTS): $(include_HEADERS)

