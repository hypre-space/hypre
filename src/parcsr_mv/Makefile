#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.22 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}

HEADERS =\
 headers.h\
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
 new_commpkg.c\
 numbers.c\
 par_csr_aat.c\
 par_csr_assumed_part.c\
 par_csr_bool_matop.c\
 par_csr_bool_matrix.c\
 par_csr_communication.c\
 par_csr_matop.c\
 par_csr_matrix.c\
 par_csr_matop_marked.c\
 par_csr_matvec.c\
 par_vector.c\
 par_make_system.c

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

OBJS = ${FILES:.c=.o}
DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

SONAME = libHYPRE_parcsr_mv-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/_hypre_parcsr_mv.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

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
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/_hypre_parcsr_mv.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
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

libHYPRE_parcsr_mv.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
