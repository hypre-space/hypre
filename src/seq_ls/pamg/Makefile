#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.10 $
#EHEADER**********************************************************************


include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
LIBFLAGS = ${LIBDIRS} ${LIBS}

C_COMPILE_FLAGS =\
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../seq_mv\
 ${CINCLUDES}
 
LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L.\
 -L../../seq_mv\
 -L../../utilities\
 -L../../krylov\
 -lHYPRE_amg_ls\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 -lHYPRE_krylov\
 ${MPILIBFLAGS} ${LIBS}

HEADERS =\
 headers.h\
 amg.h\
 pcg.h\
 bcsr_matrix.h\
 bcsr_matrix_dense_block.h\
 pamg.h\
 HYPRE_ls.h

FILES =\
 amg.c\
 amgstats.c\
 amg_setup.c\
 amg_solve.c\
 Atrunc.c\
 bcsr_interp.c\
 bcsr_matrix.c\
 bcsr_matrix_dense_block.c\
 bcsr_relax.c\
 cg_fun.c\
 coarsen.c\
 coarsenCR.c\
 cycle.c\
 difconv.c\
 indepset.c\
 interp.c\
 interpCR.c\
 interpRBM.c\
 inx_part_of_u_interp.c\
 laplace.c\
 laplace_9pt.c\
 laplace_27pt.c\
 pcg.c\
 random.c\
 rap.c\
 relax.c\
 schwarz.c\
 scaled_matnorm.c\
 strength.c\
 SPamg-pcg.c\
 stencil_matrix.c\
 trunc.c\
 HYPRE_amg.c\
 HYPRE_csr_pcg.c\
 HYPRE_csr_gmres.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_amg_ls-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

lib: libHYPRE_amg_ls${HYPRE_LIB_SUFFIX}

all: libHYPRE_amg_ls${HYPRE_LIB_SUFFIX} driver

install: all

clean:
	rm -f *.o lib*

distclean: clean
	rm -f driver

##################################################################
# Rules
##################################################################

driver: driver.o libHYPRE_amg_ls${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

libHYPRE_amg_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_amg_ls.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
