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


###  Internal LAPACK routines may be compiled in this sub-directory.
###  Generally, the Makefile in the utilities sub-directory controls
###  the inclusion of these routines based on the configure options.
###

include ../config/Makefile.config

LAPACK_HEADERS =  f2c.h hypre_lapack.h
LAPACK_FILES = \
dbdsqr.c\
dgebd2.c\
dgebrd.c\
dgelq2.c\
dgelqf.c\
dgels.c\
dgeqr2.c\
dgeqrf.c\
dgesvd.c\
dgetrf.c\
dgetrs.c\
dgetf2.c\
dlabad.c\
dlabrd.c\
dlacpy.c\
dlae2.c\
dlaev2.c\
dlamch.c\
dlange.c\
dlanst.c\
dlansy.c\
dlapy2.c\
dlarfb.c\
dlarf.c\
dlarfg.c\
dlarft.c\
dlartg.c\
dlas2.c\
dlascl.c\
dlaset.c\
dlasq1.c\
dlasq2.c\
dlasq3.c\
dlasq4.c\
dlasq5.c\
dlasq6.c\
dlasr.c\
dlasrt.c\
dlassq.c\
dlaswp.c\
dlasv2.c\
dlatrd.c\
dorg2l.c\
dorg2r.c\
dorgbr.c\
dorgl2.c\
dorglq.c\
dorgql.c\
dorgqr.c\
dorgtr.c\
dorm2r.c\
dormbr.c\
dorml2.c\
dormlq.c\
dormqr.c\
dpotf2.c\
dpotrf.c\
dpotrs.c\
dsteqr.c\
dsterf.c\
dsyev.c\
dsygs2.c\
dsygst.c\
dsygv.c\
dsytd2.c\
dsytrd.c\
ieeeck.c\
ilaenv.c\
lapack_utils.c\
lsame.c\
xerbla.c

OBJS = ${LAPACK_FILES:.c=${HYPRE_OBJ_SUFFIX}}

#################################################################
# Targets
#################################################################

all: ${OBJS}
	cp -fp *${HYPRE_OBJ_SUFFIX} ${HYPRE_SRC_TOP_DIR}/utilities
	cp -fp *.h ${HYPRE_BUILD_DIR}/include

install: all
	cp -fp *.h ${HYPRE_INC_INSTALL}/include

clean:
	rm -rf *.o *.lo .libs

distclean: clean

#################################################################
# Rules
#################################################################

${OBJS}: ${LAPACK_HEADERS}
