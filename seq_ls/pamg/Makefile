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

# Generated automatically from Makefile.in by configure.

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
	${BUILD_CC_SHARED} -o $@ ${OBJS}

${OBJS}: ${HEADERS}
