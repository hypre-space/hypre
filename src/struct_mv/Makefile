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
 ${CINCLUDES}

HEADERS =\
 HYPRE_struct_mv.h\
 _hypre_struct_mv.h\
 _hypre_struct_mv.hpp\
 assumed_part.h\
 box.h\
 box_manager.h\
 computation.h\
 struct_communication.h\
 struct_grid.h\
 struct_matrix.h\
 struct_stencil.h\
 struct_vector.h

FILES =\
 hypre_struct_mv_mp.c\
 HYPRE_struct_mv_mp.c

CUFILES =\
 struct_axpy.c\
 struct_communication.c\
 struct_copy.c\
 struct_innerprod.c\
 struct_matrix.c\
 struct_matvec.c\
 struct_scale.c\
 struct_vector.c

#######################
# Multiprecision files
#######################
MuP_FILES = \
 assumed_part.c\
 box_algebra.c\
 box_boundary.c\
 box.c\
 box_manager.c\
 communication_info.c\
 computation.c\
 F90_HYPRE_struct_grid.c\
 F90_HYPRE_struct_matrix.c\
 F90_HYPRE_struct_stencil.c\
 F90_HYPRE_struct_vector.c\
 project.c\
 struct_grid.c\
 struct_io.c\
 struct_matrix_mask.c\
 struct_stencil.c\
 HYPRE_struct_grid.c\
 HYPRE_struct_matrix.c\
 HYPRE_struct_stencil.c\
 HYPRE_struct_vector.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
   MuP_COBJS = ${MuP_FILES:.c=.o}
else
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
endif

SONAME = libHYPRE_struct_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_struct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_struct_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_struct_mv.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/struct_mv_mup_func.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/HYPRE_struct_mv_mp.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_struct_mv_mp.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_struct_mv_mup.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_struct_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_struct_mv.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/struct_mv_mup_func.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/hypre_struct_mv_mp.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/HYPRE_struct_mv_mp.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/struct_mv_mup_mup.h $(HYPRE_INC_INSTALL)
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

OBJS = ${COBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

else
OBJS = ${COBJS} ${CUOBJS}
endif
#

libHYPRE_struct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_struct_mv.so libHYPRE_struct_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
