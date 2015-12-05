#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.26 $
#EHEADER**********************************************************************


###  the inclusion of these routines based on the configure options.

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I.. -I../utilities

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

OBJS = ${LAPACK_FILES:.c=.o}

#################################################################
# Targets
#################################################################

all: ${OBJS} dlamch.o
	cp -fp *.o ${HYPRE_SRC_TOP_DIR}/utilities

install: all
	cp -fp *.h ${HYPRE_INC_INSTALL}/include

clean:
	rm -rf *.o

distclean: clean

#################################################################
# Rules
#################################################################

# compile without optimization
dlamch.o : dlamch.c ${LAPACK_HEADERS}
	${CC} ${CFLAGS} -O0 -c dlamch.c

${OBJS}: ${LAPACK_HEADERS}
