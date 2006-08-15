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

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}
DRIVER_OBJS = ${DRIVER_FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@cp -f HYPRE_*.h parcsr_matrix_vector.h $(HYPRE_BUILD_DIR)/include/.
	@cp -f libHYPRE_* $(HYPRE_BUILD_DIR)/lib/.
 
driver: driver.o libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 
 

install: all
	@cp -f HYPRE_*.h parcsr_matrix_vector.h $(HYPRE_INC_INSTALL)/.
	@cp -f libHYPRE_* $(HYPRE_LIB_INSTALL)/.
 
clean:
	@rm -f *.o *.lo lib* driver

veryclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_mv${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR}  $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
