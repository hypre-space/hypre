#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 -I..\
 -I../utilities\
 -I../seq_mv\
 ${CINCLUDES}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS}

LFLAGS =\
 -L.\
 -L../seq_mv\
 -L../utilities\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

HEADERS =\
 par_chord_matrix.h

FILES =\
 par_chord_mm.c\
 par_chord_matrix.c\
 compute_outprocessors.c

DRIVER_FILES =\
 driver.c

OBJS = ${FILES:.c=.o}
DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@cp -fpPR HYPRE_*.h parcsr_matrix_vector.h $(HYPRE_BUILD_DIR)/include/.
#	@cp -fpPR libHYPRE_* $(HYPRE_BUILD_DIR)/lib/.

driver: driver.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 

install: all
	@cp -fpPR HYPRE_*.h parcsr_matrix_vector.h $(HYPRE_INC_INSTALL)/.
#	@cp -fpPR libHYPRE_* $(HYPRE_LIB_INSTALL)/.

clean:
	@rm -f *.o lib* driver

veryclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR}  $@ ${OBJS}
	${RANLIB} $@

libHYPRE_parcsr_mv.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS}

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
