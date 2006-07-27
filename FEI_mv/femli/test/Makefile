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


include ../../../config/Makefile.config


CC= mpicc

C_COMPILE_FLAGS=-O2
LD_LINK_FLAGS=
CINCLUDES= 
CDEFS = -DHYPRE_TIMING
 
CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../seq_mv\
 ${CINCLUDES}\
 ${CDEFS}
 
MPILIBFLAGS =     
LIBFLAGS =   -lm -lg2c
LDLIBFLAGS =   @LIBM@
LAPACKLIBFLAGS =  -L/usr/lib -L/usr/local/lib -llapack
BLASLIBFLAGS =  ${BLASLIBDIRS} ${BLASLIBS}


##################################################################
# Targets
##################################################################

all: driver

install:

clean:
	@rm -f *.o

distclean: clean

##################################################################
# Rules
##################################################################

driver: driver.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}
	@echo  " "
