#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.3 $
#EHEADER**********************************************************************










include ../../config/Makefile.config
include babel.make

IORSRCS_NONNULL = $(filter-out bHYPRE_SStructVariable_IOR.c, ${IORSRCS})
# ... ignore zero-length file because it will confuse the gcc/ar interaction

CINCLUDES=${INCLUDES} ${MPIINCLUDE}
CDEFS = 
 
BABEL_RUNTIME_INCLUDES=\
  -I$(HYPRE_SRC_TOP_DIR)/babel-runtime/sidl

C_COMPILE_FLAGS=\
 -I..\
 -I$(HYPRE_SRC_TOP_DIR)\
 -I$(HYPRE_SRC_TOP_DIR)/utilities\
 -I$(HYPRE_SRC_TOP_DIR)/seq_mv\
 -I$(HYPRE_SRC_TOP_DIR)/krylov\
 -I$(HYPRE_SRC_TOP_DIR)/IJ_mv\
 -I$(HYPRE_SRC_TOP_DIR)/parcsr_mv\
 -I$(HYPRE_SRC_TOP_DIR)/parcsr_ls\
 ${BABEL_RUNTIME_INCLUDES}\
 ${CINCLUDES}\
 ${CDEFS}
 
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS}
 
LFLAGS =\
 -L.\
 -L../seq_mv\
 -L../utilities\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${LIBFLAGS} ${LDLIBFLAGS}

HEADERS = ${STUBHDRS} ${IORHDRS} ${IMPLHDRS}

FILES = ${IMPLSRCS} ${IORSRCS_NONNULL} ${SKELSRCS} ${STUBSRCS} 

OBJS = $(IORSRCS:.c=.o) $(TYPEMODULESRCS:.F90=.o) \
	$(SKELSRCS:.c=.o) $(STUBMODULESRCS:.F90=.o) $(STUBSRCS:.c=.o) \
	$(IMPLMODULESRCS:.F90=.o) $(IMPLSRCS:.F90=.o) 

MORE_ARRAYMODULESRCS = sidl_array_type.F90
SIDL_OBJS = $(BASICMODULESRC:.F90=.o) $(MORE_ARRAYMODULESRCS:.F90=.o) \
 $(ARRAYMODULESRCS:.F90=.o)

##################################################################
# Targets
##################################################################

all: libbHYPREClient-F90.a
	cp -fp $(srcdir)/*.h ${HYPRE_BUILD_DIR}/include/.
	cp -fp libbHYPRE* ${HYPRE_BUILD_DIR}/lib/.
 
install: all
	cp -fp $(srcdir)/*.h ${HYPRE_INC_INSTALL}/.
	cp -fp libbHYPRE* ${HYPRE_LIB_INSTALL}/.
 
clean:
	rm -f *.o libbHYPRE*

distclean: clean
	rm -f *.mod

##################################################################
# Rules
##################################################################

# This will add to what's in Makefile.config.  A configure system with
# full Fortran 90 support would do this in Makefile.config...
.SUFFIXES: .F90

.F90.o:
	$(CC) -E -traditional -P -o $*.tmp -x c  $<
	sed -e 's/^#pragma.*$$//' < $*.tmp > $*.f90
	$(FC) -c -o $@  $*.f90
	rm -f $*.f90 $*.tmp

libbHYPREClient-F90.a: ${SIDL_OBJS} ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

