include babel.make
# ...defines: IORHDRS IORSRCS STUBHDRS STUBSRCS

if STATICONLY
 lib_LIBRARIES = libbHYPREClient-C.a
 libbHYPREClient_C_a_SOURCES = $(FILES)
 OBJECTS = libbHYPREClient_C_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libbHYPREClient-C.la
 libbHYPREClient_C_la_SOURCES = $(FILES)
 OBJECTS = libbHYPREClient_C_la_OBJECTS
 libbHYPREClient_C_la_LDFLAGS = -avoid-version
endif !STATICONLY

BABEL_RUNTIME_INCLUDES=\
 -I$(top_srcdir)/babel-runtime/sidl\
 -I$(top_builddir)/babel-runtime/sidl

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/seq_mv\
 ${BABEL_RUNTIME_INCLUDES}

EXTRA_DIST = 

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS}
