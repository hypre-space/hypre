## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libbHYPRE.la
libbHYPRE_la_SOURCES = $(FILES)
libbHYPRE_la_LDFLAGS = -avoid-version

C_COMPILE_FLAGS=@CFLAGS@
# new configure won't fill in: CINCLUDES=@INCLUDES@ @MPIINCLUDE@ @BABEL_HYPRE_INCLUDES@
CINCLUDES=-I$(srcdir)/../../hypre/include
BABEL_RUNTIME_INCLUDES=-I$(srcdir)/../../babel-runtime/sidl

AM_CPPFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../seq_mv\
 ${BABEL_RUNTIME_INCLUDES}\
 ${CINCLUDES}\
 ${CDEFS}

EXTRA_DIST = 

include babel.make

include_HEADERS = ${STUBHDRS} ${IORHDRS} ${IMPLHDRS}

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS} \
 bHYPRE_Extensions.c

