#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.21 $
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 HYPRE_krylov.h\
 krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 flexgmres.h\
 lgmres.h\
 pcg.h\
 HYPRE_lobpcg.h\
 HYPRE_MatvecFunctions.h\
 lobpcg.h

FILES =\
 bicgstab.c\
 cgnr.c\
 gmres.c\
 flexgmres.c\
 lgmres.c\
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_lgmres.c\
 HYPRE_flexgmres.c\
 HYPRE_pcg.c\
 pcg.c\
 HYPRE_lobpcg.c\
 lobpcg.c

OBJS = ${FILES:.c=.o}

SONAME = libHYPRE_krylov-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_krylov${HYPRE_LIB_SUFFIX}
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/krylov.h $(HYPRE_BUILD_DIR)/include
	cp -fpPR $(srcdir)/lobpcg.h $(HYPRE_BUILD_DIR)/include
#	cp -fpPR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: all
	cp -fpPR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/krylov.h $(HYPRE_INC_INSTALL)
	cp -fpPR $(srcdir)/lobpcg.h $(HYPRE_INC_INSTALL)
#	cp -fpPR libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_krylov.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_krylov.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}
