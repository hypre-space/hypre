## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libbHYPRE.la
libbHYPRE_la_SOURCES = $(FILES)
OBJECTS = libbHYPRE_la_OBJECTS
libbHYPRE_la_LDFLAGS = -avoid-version

C_COMPILE_FLAGS=@CFLAGS@
# new configure won't fill in: CINCLUDES=@INCLUDES@ @MPIINCLUDE@ @BABEL_HYPRE_INCLUDES@
CINCLUDES=-I$(includedir)
BABEL_RUNTIME_INCLUDES=-I$(top_builddir)/babel-runtime/sidl

AM_CPPFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/seq_mv\
 ${BABEL_RUNTIME_INCLUDES}\
 ${CINCLUDES}\
 ${CDEFS}

EXTRA_DIST = 

include babel.make

include_HEADERS = ${STUBHDRS} ${IORHDRS} ${IMPLHDRS}

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS} \
 bHYPRE_Extensions.c

$(OBJECTS): $(include_HEADERS)

