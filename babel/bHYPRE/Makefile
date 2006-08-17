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
 -I$(HYPRE_SRC_TOP_DIR)/hypre/include\
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

FILES = ${IMPLSRCS} ${IORSRCS_NONNULL} ${SKELSRCS} ${STUBSRCS} \
 bHYPRE_Extensions.c

OBJS = ${FILES:.c=${HYPRE_OBJ_SUFFIX}}

##################################################################
# Targets
##################################################################

all: libbHYPRE${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/*.h ${HYPRE_BUILD_DIR}/include
	cp -fp libbHYPRE* ${HYPRE_BUILD_DIR}/lib
 
install: all
	cp -fp $(srcdir)/*.h ${HYPRE_INC_INSTALL}
	cp -fp libbHYPRE* ${HYPRE_LIB_INSTALL}
 
clean:
	rm -f *.o *.lo libbHYPRE*

distclean: clean

##################################################################
# Rules
##################################################################

libbHYPRE.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libbHYPRE.so: ${OBJS}
	@echo  "Building $@ ... "
	${LINK_CC} -o $@ ${OBJS} -L${HYPRE_LIB_INSTALL} -lHYPRE -lsidl-0.10.12


${OBJS}: ${HEADERS}
