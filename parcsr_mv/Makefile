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
 -I$(srcdir)/../seq_mv\
 ${CINCLUDES}
 
HEADERS =\
 headers.h\
 HYPRE_parcsr_mv.h\
 new_commpkg.h\
 numbers.h\
 par_chord_matrix.h\
 par_csr_communication.h\
 par_csr_matrix.h\
 parcsr_mv.h\
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
 par_vector.c

EXTRA_FILES =\
 parchord_to_parcsr.c\

DRIVER_FILES =\
 driver_aat.c\
 driver_boolaat.c\
 driver_boolmatmul.c\
 driver.c\
 driver_matmul.c\
 driver_mat_multivec.c\
 driver_matvec.c\
 driver_multivec.c

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}
DRIVER_OBJS = ${DRIVER_FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/parcsr_mv.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib
 
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
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/parcsr_mv.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)
 
clean:
	rm -f *.o *.lo libHYPRE*
	rm -f driver driver_matvec driver_matmul
	rm -rf pchdir tca.map *inslog* .libs

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
