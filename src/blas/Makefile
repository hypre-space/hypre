# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

###  Internal BLAS routines may be compiled in this sub-directory
###  Generally, the Makefile in the utilities subdirectory controls
###  the inclusion of these routines based on the configure options.
###

include ../config/Makefile.config

C_COMPILE_FLAGS = ${INCLUDES} -I. -I.. -I../utilities

BLAS_HEADERS =  f2c.h  hypre_blas.h
BLAS_FILES = \
 f2c.c\
 xerbla.c

#######################
# Multiprecision files
#######################
BLAS_MuP_FILES = \
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
 lsame.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${BLAS_FILES:.c=.o}
   MuP_COBJS = ${BLAS_MuP_FILES:.c=.o} 
else
   COBJS = ${BLAS_FILES:.c=.o} ${BLAS_MuP_FILES:.c=.o} 
endif

OBJS = ${COBJS}

##################################################################
# Targets
##################################################################

all: ${OBJS} MuP_OBJS

install: all
	cp -fR $(srcdir)/_hypre_blas.h $(HYPRE_INC_INSTALL)

clean: clean_mup
	@rm -rf *.o

distclean: clean

ifeq (${MP_BUILD}, 1) 
clean_mup:
	rm -rf ${MuP_OBJDIR}	
else
clean_mup:
endif

##################################################################
# Rules
##################################################################

${OBJS}: ${BLAS_HEADERS}

##################################################################
# Multiprecision build Rules
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
ADD_OBJS_single = ${addprefix ${MuP_OBJDIR_single}/, ${notdir ${MuP_COBJS}}}
${ADD_OBJS_single}: CFLAGS += -DMP_BUILD_SINGLE=1
${MuP_OBJDIR_single}/%.o: %.c
#	@echo  "MuP_OBJDIR is ${ADD_OBJS_single} ... $%"
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_double = ${addprefix ${MuP_OBJDIR_double}/, ${notdir ${MuP_COBJS}}}
${ADD_OBJS_double}: CFLAGS += -DMP_BUILD_DOUBLE=1
${MuP_OBJDIR_double}/%.o: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_longdouble = ${addprefix ${MuP_OBJDIR_longdouble}/, ${notdir ${MuP_COBJS}}}
${ADD_OBJS_longdouble}: CFLAGS += -DMP_BUILD_LONGDOUBLE=1
${MuP_OBJDIR_longdouble}/%.o: %.c
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
# multiprecision object files
MuP_OBJS: ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}
#
else
MuP_OBJS:
endif

	
