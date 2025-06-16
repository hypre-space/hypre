# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

###  Internal BLAS routines may be compiled in this sub-directory
###  Generally, the Makefile in the utilities subdirectory controls
###  the inclusion of these routines based on the configure options.
###

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I.. -I../utilities

BLAS_HEADERS =  f2c.h  hypre_blas.h
BLAS_FILES = \
 dasum.c\
 daxpy.c\
 dcopy.c\
 ddot.c\
 dgemm.c\
 dgemv.c\
 dger.c\
 dnrm2.c\
 drot.c\
 dscal.c\
 dswap.c\
 dsymm.c\
 dsymv.c\
 dsyr2.c\
 dsyr2k.c\
 dsyrk.c\
 dtrmm.c\
 dtrmv.c\
 dtrsm.c\
 dtrsv.c\
 f2c.c\
 idamax.c\
 lsame.c\
 xerbla.c

OBJS = ${BLAS_FILES:.c=.o}

##################################################################
# Targets
##################################################################

#all: ${OBJS}

all: blas_OBJS

install: all
	cp -fR $(srcdir)/_hypre_blas.h $(HYPRE_INC_INSTALL)

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
${MuP_OBJDIR_single}/%.o: %.c ${BLAS_HEADERS}
#	@echo  "MuP_OBJDIR is ${ADD_OBJS_single} ... $%"
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_double = ${addprefix ${MuP_OBJDIR_double}/, ${notdir ${OBJS}}}
${ADD_OBJS_double}: CFLAGS += -DMP_BUILD_DOUBLE=1
${MuP_OBJDIR_double}/%.o: %.c ${BLAS_HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_longdouble = ${addprefix ${MuP_OBJDIR_longdouble}/, ${notdir ${OBJS}}}
${ADD_OBJS_longdouble}: CFLAGS += -DMP_BUILD_LONGDOUBLE=1
${MuP_OBJDIR_longdouble}/%.o: %.c ${BLAS_HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
# multiprecision object files
${OBJS}: CFLAGS += -DMP_BUILD_DEFAULT=1
blas_OBJS: ${OBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}
#
else
${OBJS}: ${BLAS_HEADERS}
blas_OBJS: ${OBJS}
endif

