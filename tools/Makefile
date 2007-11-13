#BHEADER**********************************************************************
# Copyright (c) 2007,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${HYPREDEFS} -DHYPRE_TIMING

C_COMPILE_FLAGS =\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../struct_mv\
 ${CINCLUDES} ${CDEFS}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS}

LFLAGS =\
 -L../struct_mv\
 -L../utilities\
 -lHYPRE_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

##################################################################
# Targets
##################################################################

all: 

install: all
	cp -f V[1-9]* $$HYPRE_INSTALL_DIR/bin/.

clean:
	rm -f *.o

veryclean: clean
	rm -f one_to_many
	rm -f one_to_many_vector
	rm -f create_2d_laplacian 
	rm -f create_3d_laplacian
	rm -f Makefile

##################################################################
# Rules
##################################################################

one_to_many: one_to_many.o
	@echo  "Linking" $@ "... "
	${CC} -o one_to_many one_to_many.o ${LFLAGS}

one_to_many_vector: one_to_many_vector.o
	@echo  "Linking" $@ "... "
	${CC} -o one_to_many_vector one_to_many_vector.o ${LFLAGS}

create_2d_laplacian: create_2d_laplacian.o
	@echo  "Linking" $@ "... "
	${CC} -o create_2d_laplacian create_2d_laplacian.o ${LFLAGS}

create_3d_laplacian: create_3d_laplacian.o
	@echo  "Linking" $@ "... "
	${CC} -o create_3d_laplacian create_3d_laplacian.o ${LFLAGS}
