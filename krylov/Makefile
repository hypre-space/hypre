#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
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
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 all_krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 krylov.h\
 pcg.h\
 HYPRE_lobpcg.h\
 HYPRE_MatvecFunctions.h\
 lobpcg.h

FILES =\
 bicgstab.c\
 cgnr.c\
 gmres.c\
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_pcg.c\
 pcg.c\
 HYPRE_lobpcg.c\
 lobpcg.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_krylov${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/krylov.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/HYPRE_MatvecFunctions.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/*lobpcg*.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: all
	cp -fp $(srcdir)/krylov.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/HYPRE_MatvecFunctions.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/*lobpcg*.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

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
	${BUILD_CC_SHARED} -o $@ ${OBJS}

${OBJS}: ${HEADERS}
