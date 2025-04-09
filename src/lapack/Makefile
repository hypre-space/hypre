# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

###  the inclusion of these routines based on the configure options.

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I.. -I../blas -I../utilities

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
 dgetri.c\
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
 dtrtri.c\
 dtrti2.c\
 ieeeck.c\
 ilaenv.c\
 lsame.c\
 xerbla.c

OBJS = ${LAPACK_FILES:.c=.o}

#################################################################
# Targets
#################################################################

all: lapack_OBJS dlamch_obj

install: all
	cp -fR $(srcdir)/_hypre_lapack.h $(HYPRE_INC_INSTALL)

clean: clean_mup
	rm -rf *.o

distclean: clean

ifeq (${MP_BUILD}, 1) 
clean_mup:
	rm -rf ${MuP_OBJDIR}	
else
clean_mup:
endif

##################################################################
# Build Rules
##################################################################

ifeq (${MP_BUILD}, 1) 

MuP_OBJDIR = MuP_obj.dir
MuP_OBJDIR_single = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_flt}
MuP_OBJDIR_double = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_dbl}
MuP_OBJDIR_longdouble = ${MuP_OBJDIR}/${join ${shell basename ${CURDIR}},_ldbl}
#
#MKDIR_P = mkdir -p
#mkmpdir: 
#	${MKDIR_P} ${MuP_OBJDIR}
#
ADD_OBJS_single = ${addprefix ${MuP_OBJDIR_single}/, ${notdir ${OBJS}}}
${ADD_OBJS_single}: CFLAGS += -DMP_BUILD_SINGLE=1
${MuP_OBJDIR_single}/%.o: %.c ${LAPACK_HEADERS}
#	@echo  "MuP_OBJDIR is ${ADD_OBJS_single} ... $%"
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
dlamch_single = ${MuP_OBJDIR_single}/dlamch.o
${dlamch_single}: dlamch.c ${LAPACK_HEADERS}
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -DMP_BUILD_SINGLE=1 -c -o $@ dlamch.c
#
ADD_OBJS_double = ${addprefix ${MuP_OBJDIR_double}/, ${notdir ${OBJS}}}
${ADD_OBJS_double}: CFLAGS += -DMP_BUILD_DOUBLE=1
${MuP_OBJDIR_double}/%.o: %.c ${LAPACK_HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
dlamch_double = ${MuP_OBJDIR_double}/dlamch.o
${dlamch_double}: dlamch.c ${LAPACK_HEADERS}
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -DMP_BUILD_DOUBLE=1 -c -o $@ dlamch.c
#
ADD_OBJS_longdouble = ${addprefix ${MuP_OBJDIR_longdouble}/, ${notdir ${OBJS}}}
${ADD_OBJS_longdouble}: CFLAGS += -DMP_BUILD_LONGDOUBLE=1
${MuP_OBJDIR_longdouble}/%.o: %.c ${LAPACK_HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
dlamch_longdouble = ${MuP_OBJDIR_longdouble}/dlamch.o
${dlamch_longdouble}: dlamch.c ${LAPACK_HEADERS}
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -DMP_BUILD_LONGDOUBLE=1 -c -o $@ dlamch.c
#
# multiprecision object files
lapack_OBJS: ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

# compile without optimization
dlamch_obj: ${dlamch_single} ${dlamch_double} ${dlamch_longdouble}

else
# compile without optimization
dlamch.o : dlamch.c ${LAPACK_HEADERS}
	${CC} ${CFLAGS} -c dlamch.c

dlamch_obj: dlamch.o

${OBJS}: ${LAPACK_HEADERS}
lapack_OBJS: ${OBJS}
endif

