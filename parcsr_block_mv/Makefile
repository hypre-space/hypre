#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer and the GNU Lesser General Public License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************


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
 par_csr_block_rap.c\
 par_csr_block_rap_communication.c\
 par_csr_block_interp.c\
 par_csr_block_relax.c\
 par_block_nodal_systems.c

EXTRA_FILES =

DRIVER_FILES =

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}
DRIVER_OBJS = ${DRIVER_FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_block_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/par_csr_block_matrix.h $(HYPRE_BUILD_DIR)/include	
	cp -fp $(srcdir)/csr_block_matrix.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE_* $(HYPRE_BUILD_DIR)/lib/.
 
driver: driver.o libHYPRE_parcsr_block_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 
 
driver_matvec: driver_matvec.o
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

install: libHYPRE_parcsr_block_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/par_csr_block_matrix.h $(HYPRE_INC_INSTALL)/.
	cp -fp $(srcdir)/csr_block_matrix.h $(HYPRE_INC_INSTALL)/.
	cp -fp libHYPRE_* $(HYPRE_LIB_INSTALL)/.
 
clean:
	rm -f *.o *.lo libHYPRE_*
	rm -f driver driver_matvec driver_matmul

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_block_mv${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}

