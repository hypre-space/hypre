include babel.make
# ...defines: IORHDRS IORSRCS STUBHDRS STUBSRCS

lib_LTLIBRARIES = libbHYPREClient-F.la
libbHYPREClient_F_la_SOURCES = $(FILES)
OBJECTS = libbHYPREClient_F_la_OBJECTS
libbHYPREClient_F_la_LDFLAGS = -avoid-version

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

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS}
