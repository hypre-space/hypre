AUTOMAKE_OPTIONS = foreign no-dependencies

BABELDIR = @babeldir@

check_PROGRAMS = testum

testum_SOURCES = testum.c

testum_LDADD =  -L$(BABELDIR)/lib -L../../hypre/lib -L../Hypre -lhypre_class -lbabel -lHYPRE_ls -lHYPRE_mv -lHYPRE_utilities

HYPRE_INCLUDES = -I../../hypre/include -I../../struct_mv
BABEL_INCLUDES  = -I${BABELDIR}/include -I${BABELDIR}/include/babel
INCLUDES = -I.. ${HYPRE_INCLUDES} ${BABEL_INCLUDES} -I../Hypre

clean-local: 
	rm -f testum.out testuv.out
