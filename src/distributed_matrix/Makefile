#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.1 $
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}

EXTRA_HEADERS = internal_protos.h

HEADERS =\
 distributed_matrix.h\
 HYPRE_distributed_matrix_mv.h\
 HYPRE_distributed_matrix_protos.h\
 HYPRE_distributed_matrix_types.h

FILES =\
 distributed_matrix.c\
 HYPRE_distributed_matrix.c\
 distributed_matrix_ISIS.c\
 distributed_matrix_PETSc.c\
 distributed_matrix_parcsr.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_DistributedMatrix.a 
	cp -fp $(srcdir)/distributed_matrix.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp lib*.a $(HYPRE_BUILD_DIR)/lib/.

install: libHYPRE_DistributedMatrix.a 
	cp -fp $(srcdir)/distributed_matrix.h $(HYPRE_INC_INSTALL)/.
	cp -fp lib*.a $(HYPRE_LIB_INSTALL)/.
clean:
	rm -f *.o lib*.a

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_DistributedMatrix.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
