# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../blas\
 -I$(srcdir)/../lapack\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 HYPRE_krylov.h\
 krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 cogmres.h\
 flexgmres.h\
 lgmres.h\
 pcg.h\
 HYPRE_lobpcg.h\
 HYPRE_MatvecFunctions.h\
 lobpcg.h

FILES =\

#######################
# Multiprecision files
#######################
MuP_FILES = \
 bicgstab.c\
 cgnr.c\
 gmres.c\
 cogmres.c\
 flexgmres.c\
 lgmres.c\
 pcg.c\
 lobpcg.c\
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_cogmres.c\
 HYPRE_lgmres.c\
 HYPRE_flexgmres.c\
 HYPRE_pcg.c\
 HYPRE_lobpcg.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   MuP_COBJS = ${MuP_FILES:.c=.o}
else
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
endif

SONAME = libHYPRE_krylov-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_krylov${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/krylov.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/lobpcg.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_krylov_mup_def.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_krylov_mup_undef.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_krylov_mup.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: all
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/krylov.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/lobpcg.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_krylov_mup_def.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_krylov_mup_undef.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/hypre_krylov_mup.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean: clean_mup
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

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
ADD_OBJS_single = ${addprefix ${MuP_OBJDIR_single}/, ${notdir ${MuP_COBJS}}}
${ADD_OBJS_single}: CFLAGS += -DMP_BUILD_SINGLE=1
${MuP_OBJDIR_single}/%.o: %.c
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
${COBJS}: CFLAGS += -DMP_BUILD_FIXED=1

OBJS = ${COBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

else
OBJS = ${COBJS}
endif
#
libHYPRE_krylov.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_krylov.so libHYPRE_krylov.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@

${OBJS}: ${HEADERS}
