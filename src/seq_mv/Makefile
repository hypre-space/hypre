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
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 csr_matrix.h\
 HYPRE_seq_mv.h\
 mapped_matrix.h\
 multiblock_matrix.h\
 seq_mv.h\
 seq_mv.hpp\
 vector.h

FILES =\
 csr_matop.c\
 csr_matrix.c\
 csr_matvec.c\
 genpart.c\
 HYPRE_csr_matrix.c\
 HYPRE_mapped_matrix.c\
 HYPRE_multiblock_matrix.c\
 HYPRE_vector.c\
 mapped_matrix.c\
 multiblock_matrix.c\
 vector.c\
 vector_batched.c

CUFILES =\
 csr_matop_device.c\
 csr_matrix_cuda_utils.c\
 csr_matvec_device.c\
 csr_matvec_oomp.c\
 csr_spadd_device.c\
 csr_spgemm_device.c\
 csr_spgemm_device_cusparse.c\
 csr_spgemm_device_numblocks.c\
 csr_spgemm_device_numer.c\
 csr_spgemm_device_numer1.c\
 csr_spgemm_device_numer2.c\
 csr_spgemm_device_numer3.c\
 csr_spgemm_device_numer4.c\
 csr_spgemm_device_numer5.c\
 csr_spgemm_device_numer6.c\
 csr_spgemm_device_numer7.c\
 csr_spgemm_device_numer8.c\
 csr_spgemm_device_numer9.c\
 csr_spgemm_device_numer10.c\
 csr_spgemm_device_onemklsparse.c\
 csr_spgemm_device_rocsparse.c \
 csr_spgemm_device_rowest.c\
 csr_spgemm_device_symbl.c\
 csr_spgemm_device_symbl1.c\
 csr_spgemm_device_symbl2.c\
 csr_spgemm_device_symbl3.c\
 csr_spgemm_device_symbl4.c\
 csr_spgemm_device_symbl5.c\
 csr_spgemm_device_symbl6.c\
 csr_spgemm_device_symbl7.c\
 csr_spgemm_device_symbl8.c\
 csr_spgemm_device_symbl9.c\
 csr_spgemm_device_symbl10.c\
 csr_spgemm_device_util.c\
 csr_spmv_device.c\
 csr_sptrans_device.c\
 vector_device.c

COBJS = ${FILES:.c=.o}
CUOBJS = ${CUFILES:.c=.obj}
OBJS = ${COBJS} ${CUOBJS}

SONAME = libHYPRE_seq_mv-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all:
	$(MAKE) autogen
	$(MAKE) lib

lib: libHYPRE_seq_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/seq_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_seq_mv${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/seq_mv.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean: deleteautogen
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

autogen:
	@for number in 1 2 3 4 5 6 7 8 9 10; do \
	   fn=csr_spgemm_device_numer$$number.c; \
	   if [ ! -f $$fn ]; then \
	      echo "#define HYPRE_SPGEMM_BIN $$number" > $$fn; \
	      cat csr_spgemm_device_numer.in >> $$fn; \
	   fi \
	   done
	@for number in 1 2 3 4 5 6 7 8 9 10; do \
	   fn=csr_spgemm_device_symbl$$number.c; \
	   if [ ! -f $$fn ]; then \
	      echo "#define HYPRE_SPGEMM_BIN $$number" > $$fn; \
	      cat csr_spgemm_device_symbl.in >> $$fn; \
	   fi \
	   done

deleteautogen:
	rm -f csr_spgemm_device_numer[1-9].c
	rm -f csr_spgemm_device_numer10.c
	rm -f csr_spgemm_device_symbl[1-9].c
	rm -f csr_spgemm_device_symbl10.c

##################################################################
# Rules
##################################################################

libHYPRE_seq_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_seq_mv.so libHYPRE_seq_mv.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
