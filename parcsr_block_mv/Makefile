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
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../parcsr_ls\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../krylov\
 ${CINCLUDES}
 
LFLAGS = \
  -L. -lHYPRE_parcsr_block_mv -L../hypre/lib -lHYPRE_krylov \
      -lHYPRE_parcsr_ls -lHYPRE_ParaSails -lHYPRE_Euclid \
      -lHYPRE_DistributedMatrixPilutSolver -lHYPRE_IJ_mv \
      -lHYPRE_parcsr_mv -lHYPRE_seq_mv -lHYPRE_DistributedMatrix \
      -lHYPRE_MatrixMatrix -lHYPRE_utilities -lm
# -L. -lHYPRE_parcsr_block_mv -L../hypre/lib -lHYPRE_LSI -lHYPRE_superlu -lm

HEADERS =\
 csr_block_matrix.h\
 par_csr_block_matrix.h

FILES =\
 csr_block_matrix.c\
 csr_block_matvec.c\
 par_csr_block_matrix.c\
 par_csr_block_matvec.c\
 par_csr_block_comm.c\
 par_csr_block_matop.c\
 par_csr_block_rap.c\
 par_csr_block_rap_communication.c\
 par_csr_block_interp.c\
 par_csr_block_relax.c\
 par_block_nodal_systems.c

EXTRA_FILES =

DRIVER_FILES =

OBJS = ${FILES:.c=.o}
DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_block_mv.a 
	cp -fp $(srcdir)/par_csr_block_matrix.h $(HYPRE_BUILD_DIR)/include	
	cp -fp $(srcdir)/csr_block_matrix.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE_*.a $(HYPRE_BUILD_DIR)/lib/.
 
mytest: mytest.o libHYPRE_parcsr_block_mv.a 
	@echo  "Linking" $@ "... "
	${CC} -o mytest mytest.o ${LFLAGS} 
 

driver: driver.o libHYPRE_parcsr_block_mv.a 
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 
 
driver_matvec: driver_matvec.o
	@echo  "Linking" $@ "... "
	${CC} -o driver_matvec driver_matvec.o ${LFLAGS} 
 
driver_matmul: driver_matmul.o libHYPRE_parcsr_mv.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_matmul driver_matmul.o ${LFLAGS} 

driver_boolmatmul: driver_boolmatmul.o libHYPRE_parcsr_mv.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_boolmatmul driver_boolmatmul.o ${LFLAGS} 

driver_aat: driver_aat.o libHYPRE_parcsr_mv.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_aat driver_aat.o ${LFLAGS} 

driver_boolaat: driver_boolaat.o libHYPRE_parcsr_mv.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_boolaat driver_boolaat.o ${LFLAGS} 

install: libHYPRE_parcsr_block_mv.a
	cp -fp $(srcdir)/par_csr_block_matrix.h $(HYPRE_INC_INSTALL)/.
	cp -fp $(srcdir)/csr_block_matrix.h $(HYPRE_INC_INSTALL)/.
	cp -fp libHYPRE_*.a $(HYPRE_LIB_INSTALL)/.
 
clean:
	rm -f *.o lib*.a
	rm -f driver driver_matvec driver_matmul

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_block_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}

