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
 -I$(srcdir)/../blas\
 -I$(srcdir)/../lapack\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../seq_block_mv\
 ${CINCLUDES}

HEADERS =\
 HYPRE_parcsr_mv.h\
 _hypre_parcsr_mv.h\
 new_commpkg.h\
 numbers.h\
 par_chord_matrix.h\
 par_csr_communication.h\
 par_csr_matrix.h\
 par_vector.h

FILES =\
 communicationT.c\
 F90_HYPRE_parcsr_matrix.c\
 F90_HYPRE_parcsr_vector.c\
 F90_parcsr_matrix.c\
 F90_par_vector.c\
 HYPRE_parcsr_matrix.c\
 HYPRE_parcsr_vector.c\
 gen_fffc.c\
 new_commpkg.c\
 numbers.c\
 par_csr_aat.c\
 par_csr_assumed_part.c\
 par_csr_bool_matop.c\
 par_csr_bool_matrix.c\
 par_csr_communication.c\
 par_csr_matop.c\
 par_csr_matrix.c\
 par_csr_matrix_stats.c\
 par_csr_matvec.c\
 par_csr_matop_marked.c\
 par_csr_triplemat.c\
 par_make_system.c\
 par_vector.c\
 par_vector_batched.c

CUFILES =\
 par_csr_fffc_device.c\
 par_csr_matop_device.c\
 par_csr_matvec_device.c\
 par_csr_triplemat_device.c\
 par_vector_device.c

EXTRA_FILES =\
 parchord_to_parcsr.c

DRIVER_FILES =\
 driver_aat.c\
 driver_boolaat.c\
 driver_boolmatmul.c\
 driver.c\
 driver_matmul.c\
 driver_mat_multivec.c\
 driver_matvec.c\
 driver_multivec.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

SONAME = libHYPRE_parcsr_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_parcsr_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

driver: driver.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

driver_matvec: driver_matvec.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_matvec driver_matvec.o ${LFLAGS}

driver_matmul: driver_matmul.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_matmul driver_matmul.o ${LFLAGS}

driver_boolmatmul: driver_boolmatmul.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_boolmatmul driver_boolmatmul.o ${LFLAGS}

driver_aat: driver_aat.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_aat driver_aat.o ${LFLAGS}

driver_boolaat: driver_boolaat.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver_boolaat driver_boolaat.o ${LFLAGS}

install: libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_parcsr_mv.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o *.obj libHYPRE*
	rm -f driver driver_matvec driver_matmul
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_parcsr_mv.so libHYPRE_parcsr_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
