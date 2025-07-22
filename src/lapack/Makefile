# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

###  the inclusion of these routines based on the configure options.

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I.. -I../blas -I../utilities

HEADERS =\
 f2c.h\
 hypre_lapack.h

FILES =\
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
 dgetri.c\
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
 dtrtri.c\
 dtrti2.c\
 ieeeck.c\
 ilaenv.c\
 lsame.c\
 xerbla.c

# Mixed precision files
MP_FILES = \
 mup_fixed.c\
 mup_functions.c\
 mup_pre.c

OBJS = ${FILES:.c=.o}

ifeq (${MP_BUILD}, 1) 

COBJS_single      = ${FILES:.c=.o_flt}
COBJS_double      = ${FILES:.c=.o_dbl}
COBJS_longdouble  = ${FILES:.c=.o_ldbl}
MP_COBJS          = ${MP_FILES:.c=.o}

OBJS = ${COBJS_single} ${COBJS_double} ${COBJS_longdouble} ${MP_COBJS}

endif

#################################################################
# Targets
#################################################################

all: ${OBJS}
	cp -fR $(srcdir)/_hypre_lapack.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_BUILD_DIR)/include

install: all
	cp -fR $(srcdir)/_hypre_lapack.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_INC_INSTALL)

clean:
	rm -rf *.o
	rm -f *.o_flt *.o_dbl *.o_ldbl

distclean: clean

#################################################################
# Rules
#################################################################

${OBJS}: ${HEADERS}
