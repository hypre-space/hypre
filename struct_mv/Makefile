#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 assumed_part.h\
 box.h\
 box_neighbors.h\
 box_pthreads.h\
 computation.h\
 headers.h\
 hypre_box_smp_forloop.h\
 HYPRE_struct_mv.h\
 struct_communication.h\
 struct_grid.h\
 struct_matrix.h\
 struct_mv.h\
 struct_stencil.h\
 struct_vector.h

FILES =\
 assumed_part.c\
 box_algebra.c\
 box_alloc.c\
 box_boundary.c\
 box.c\
 box_neighbors.c\
 communication_info.c\
 computation.c\
 F90_HYPRE_struct_grid.c\
 F90_HYPRE_struct_matrix.c\
 F90_HYPRE_struct_stencil.c\
 F90_HYPRE_struct_vector.c\
 grow.c\
 HYPRE_struct_grid.c\
 HYPRE_struct_matrix.c\
 HYPRE_struct_stencil.c\
 HYPRE_struct_vector.c\
 new_assemble.c\
 new_box_neighbors.c\
 project.c\
 struct_axpy.c\
 struct_communication.c\
 struct_copy.c\
 struct_grid.c\
 struct_innerprod.c\
 struct_io.c\
 struct_matrix.c\
 struct_matrix_mask.c\
 struct_matvec.c\
 struct_overlap_innerprod.c\
 struct_scale.c\
 struct_stencil.c\
 struct_vector.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_struct_mv.a
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/struct_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib
	@\
	for file in ${OBJS}; \
	do \
	   (cp -fp $$file ${HYPRE_SRC_TOP_DIR}/lib); \
	done

install: libHYPRE_struct_mv.a
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/struct_mv.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_struct_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
