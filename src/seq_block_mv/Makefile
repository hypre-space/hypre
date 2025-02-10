# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 _hypre_seq_block_mv.h

FILES =\
 dense_block_matrix.c\
 dense_block_matmult.c

# CUFILES =\

COBJS = ${FILES:.c=.o}
#CUOBJS = ${CUFILES:.c=.obj}
#OBJS = ${COBJS} ${CUOBJS}
OBJS = ${COBJS}

SONAME = libHYPRE_seq_block_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all:
	$(MAKE) lib

lib: libHYPRE_seq_block_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/_hypre_seq_block_mv.h $(HYPRE_BUILD_DIR)/include

install: libHYPRE_seq_block_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/_hypre_seq_block_mv.h $(HYPRE_INC_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_seq_block_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_seq_block_mv.so libHYPRE_seq_block_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
