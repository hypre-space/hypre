
include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../../\
 -I$(srcdir)/../../utilities\
 ${CINCLUDES}

HEADERS =\
 Cnames.h\
 colamd.h\
 dsp_defs.h\
 superlu_util.h\
 supermatrix.h

FILES =\
 colamd.c \
 dcolumn_bmod.c \
 dcolumn_dfs.c \
 dcopy_to_ucol.c \
 dgscon.c \
 dgsequ.c \
 dgsrfs.c \
 dgssv.c \
 dgssvx.c \
 dgstrf.c \
 dgstrs.c \
 dlacon.c \
 dlangs.c \
 dlaqgs.c \
 dmemory.c \
 dpanel_bmod.c \
 dpanel_dfs.c \
 dpivotgrowth.c \
 dpivotL.c \
 dpruneL.c \
 dreadhb.c \
 dsnode_bmod.c \
 dsnode_dfs.c \
 dsp_blas2.c \
 dsp_blas3.c \
 dutil.c \
 get_perm_c.c \
 mmd.c \
 relax_snode.c \
 sp_coletree.c \
 sp_ienv.c \
 sp_preorder.c \
 superlu_memory.c \
 superlu_timer.c \
 superlu_util.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_superlu${HYPRE_LIB_SUFFIX}
	cp -fp *.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_superlu${HYPRE_LIB_SUFFIX}
	cp -f *.h $(HYPRE_INC_INSTALL)
	cp -f libHYPRE* $(HYPRE_LIB_INSTALL)
	@echo  " "

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_superlu.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_superlu.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS}

${OBJS}: ${HEADERS}
