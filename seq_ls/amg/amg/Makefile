#BHEADER**********************************************************************
# Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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





.SUFFIXES:
.SUFFIXES: .c .f .${AMG_ARCH}.o

AMG_OBJS1 = ${AMG_FILES:.c=.${AMG_ARCH}.o}
AMG_OBJS  = ${AMG_OBJS1:.f=.${AMG_ARCH}.o}

##################################################################
# Main rules
##################################################################

${AMG_LIBRARY}.${AMG_ARCH}.a: ${AMG_OBJS}
	@echo  "Building ${AMG_LIBRARY}.${AMG_ARCH}.a ... "
	@${AMG_AR} ${AMG_LIBRARY}.${AMG_ARCH}.a ${AMG_OBJS}
	@${AMG_RANLIB} ${AMG_LIBRARY}.${AMG_ARCH}.a

${AMG_OBJS}: ${AMG_HEADERS}

##################################################################
# Generic rules
##################################################################

.c.${AMG_ARCH}.o:
	@echo "Making (c) " $@
	@${AMG_CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${AMG_F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

install:
	@${AMG_INSTALL} ${AMG_LIBRARY}.${AMG_ARCH}.a \
                       ${AMG_LIB_DIR}/lib${AMG_LIBRARY}.a
	@${AMG_INSTALL} amg.h vector.h matrix.h ${AMG_INCLUDE_DIR}

veryclean: clean
	@rm -f ${AMG_LIBRARY}.${AMG_ARCH}.a

clean:
	@rm -f *.${AMG_ARCH}.o

