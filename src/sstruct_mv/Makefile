# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../seq_block_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../IJ_mv\
 ${CINCLUDES}

HEADERS =\
 HYPRE_sstruct_mv.h\
 _hypre_sstruct_mv.h\
 sstruct_graph.h\
 sstruct_grid.h\
 sstruct_matrix.h\
 sstruct_stencil.h\
 sstruct_vector.h

FILES =\

CUFILES =\
 sstruct_matrix.c\
 sstruct_vector.c

#######################
# Multiprecision files
#######################
MuP_FILES = \
 F90_HYPRE_sstruct_graph.c\
 F90_HYPRE_sstruct_grid.c\
 F90_HYPRE_sstruct_matrix.c\
 F90_HYPRE_sstruct_stencil.c\
 F90_HYPRE_sstruct_vector.c\
 sstruct_axpy.c\
 sstruct_copy.c\
 sstruct_graph.c\
 sstruct_grid.c\
 sstruct_innerprod.c\
 sstruct_matvec.c\
 sstruct_scale.c\
 sstruct_stencil.c\
 HYPRE_sstruct_graph.c\
 HYPRE_sstruct_grid.c\
 HYPRE_sstruct_matrix.c\
 HYPRE_sstruct_stencil.c\
 HYPRE_sstruct_vector.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
   MuP_COBJS = ${MuP_FILES:.c=.o}
else
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
endif

SONAME = libHYPRE_sstruct_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_sstruct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_sstruct_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/sstruct_mv_mup_func.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_sstruct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_sstruct_mv.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/sstruct_mv_mup_func.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean: clean_mup
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

ifeq (${MP_BUILD}, 1) 
clean_mup:
	rm -rf ${MuP_OBJDIR}	
else
clean_mup:
endif

##################################################################
# Build Rules
##################################################################

ifeq (${MP_BUILD}, 1) 

MuP_OBJDIR = MuP_obj.dir
MuP_OBJDIR_single = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_flt}
MuP_OBJDIR_double = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_dbl}
MuP_OBJDIR_longdouble = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_ldbl}
#
ADD_OBJS_single = ${addprefix ${MuP_OBJDIR_single}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_single}: CFLAGS += -DMP_BUILD_SINGLE=1
${MuP_OBJDIR_single}/%.o ${MuP_OBJDIR_single}/%.obj: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_double = ${addprefix ${MuP_OBJDIR_double}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_double}: CFLAGS += -DMP_BUILD_DOUBLE=1
${MuP_OBJDIR_double}/%.o ${MuP_OBJDIR_double}/%.obj: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_longdouble = ${addprefix ${MuP_OBJDIR_longdouble}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_longdouble}: CFLAGS += -DMP_BUILD_LONGDOUBLE=1
${MuP_OBJDIR_longdouble}/%.o ${MuP_OBJDIR_longdouble}/%.obj: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<

OBJS = ${COBJS} ${CUOBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

else
OBJS = ${COBJS} ${CUOBJS}
endif
#
libHYPRE_sstruct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_sstruct_mv.so libHYPRE_sstruct_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
