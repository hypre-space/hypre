#BHEADER***********************************************************************
# (c) 1999   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.1 $
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS =\
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 ${CINCLUDES}

HEADERS =\
 aux_parcsr_matrix.h\
 aux_par_vector.h\
 headers.h\
 HYPRE_IJ_mv.h\
 IJ_matrix.h\
 IJ_mv.h\
 IJ_vector.h

EXTRA_FILES =\
 IJMatrix_isis.c\
 IJMatrix_petsc.c

FILES =\
 aux_parcsr_matrix.c\
 aux_par_vector.c\
 F90_HYPRE_IJMatrix.c\
 F90_HYPRE_IJVector.c\
 F90_IJMatrix.c\
 HYPRE_IJMatrix.c\
 HYPRE_IJVector.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector.c\
 IJVector_parcsr.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_IJ_mv.a
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp $(srcdir)/IJ_mv.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp libHYPRE_*.a $(HYPRE_BUILD_DIR)/lib/.

install: libHYPRE_IJ_mv.a
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)/.
	cp -fp $(srcdir)/IJ_mv.h $(HYPRE_INC_INSTALL)/.
	cp -fp libHYPRE_*.a $(HYPRE_LIB_INSTALL)/.

clean:
	rm -f *.o lib*.a

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_IJ_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
