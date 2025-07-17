# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${TIMERDEFS}

C_COMPILE_FLAGS =\
   -I..\
   -I$(srcdir)/..\
   -I$(srcdir)/../struct_mv\
   -I$(srcdir)\
   ${CINCLUDES}\
   ${CDEFS}

CXX_COMPILE_FLAGS = ${C_COMPILE_FLAGS}

HEADERS =\
 amg_linklist.h\
 exchange_data.h\
 fortran.h\
 fortran_matrix.h\
 general.h\
 int_array.h\
 memory.h\
 memory_tracker.h\
 multiprecision.h\
 HYPRE_utilities.h\
 mpistubs.h\
 threading.h\
 timing.h\
 magma.h\
 _hypre_onedpl.hpp\
 _hypre_utilities.h\
 _hypre_utilities.hpp\
 _hypre_mup_def.h

FILES =\
 
#######################
# Multiprecision files
#######################
MuP_FILES = \
 F90_HYPRE_error.c\
 F90_HYPRE_general.c\
 HYPRE_handle.c\
 HYPRE_version.c\
 amg_linklist.c\
 binsearch.c\
 exchange_data.c\
 fortran_matrix.c\
 ap.c\
 log.c\
 complex.c\
 error.c\
 int_array.c\
 hopscotch_hash.c\
 matrix_stats.c\
 magma.c\
 memory_tracker.c\
 merge_sort.c\
 mmio.c\
 mpi_comm_f2c.c\
 multiprecision.c\
 prefix_sum.c\
 printf.c\
 qsort.c\
 utilities.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 state.c\
 threading.c\
 timer.c\
 timing.c

CUFILES=\
 device_utils.c\
 general.c\
 handle.c\
 int_array_device.c\
 memory.c\
 omp_device.c\
 nvtx.c\
 stl_ops.c

ifeq (${MP_BUILD}, 1) 
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
   MuP_COBJS = ${MuP_FILES:.c=.o}
else
   COBJS = ${FILES:.c=.o} ${MuP_FILES:.c=.o}
   CUOBJS = ${CUFILES:.c=.obj}
endif

#COBJS = ${FILES:.c=.o}
#CUOBJS = ${CUFILES:.c=.obj}
#OBJS = ${COBJS} ${CUOBJS}

SONAME = libHYPRE_utilities-${HYPRE_RELEASE_VERSION}${HYPRE_LIB_SUFFIX}

##################################################################
# Targets
##################################################################

all: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_onedpl.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_utilities.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_utilities.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_mup_def.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/fortran*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/utilities_mup_func.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_utilities_mup.h $(HYPRE_BUILD_DIR)/include
#	cp -fR libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_utilities${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_utilities.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_mup_def.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/fortran*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/utilities_mup_func.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/hypre_utilities_mup.h $(HYPRE_INC_INSTALL)
#	cp -fR libHYPRE* $(HYPRE_LIB_INSTALL)

clean: clean_mup
	rm -f *.o *.obj libHYPRE*
	rm -rf pchdir tca.map *inslog*

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

#libHYPRE_utilities.a: ${OBJS}
#	@echo  "Building $@ ... "
#	${AR} $@ ${OBJS}
#	${RANLIB} $@

#libHYPRE_utilities.so libHYPRE_utilities.dylib: ${OBJS}
#	@echo  "Building $@ ... "
#	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
#	ln -s -f ${SONAME} $@

#${OBJS}: ${HEADERS}

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
ADD_OBJS_single = ${addprefix ${MuP_OBJDIR_single}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_single}: CFLAGS += -DMP_BUILD_SINGLE=1
${MuP_OBJDIR_single}/%.o ${MuP_OBJDIR_single}/%.obj: %.c ${HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_double = ${addprefix ${MuP_OBJDIR_double}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_double}: CFLAGS += -DMP_BUILD_DOUBLE=1
${MuP_OBJDIR_double}/%.o ${MuP_OBJDIR_double}/%.obj: %.c ${HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
ADD_OBJS_longdouble = ${addprefix ${MuP_OBJDIR_longdouble}/, ${notdir ${MuP_COBJS} ${CUOBJS}}}
${ADD_OBJS_longdouble}: CFLAGS += -DMP_BUILD_LONGDOUBLE=1
${MuP_OBJDIR_longdouble}/%.o ${MuP_OBJDIR_longdouble}/%.obj: %.c ${HEADERS}
	@echo  "processing C object $@ ..."
	@mkdir -p ${@D}
	${CC} ${CFLAGS} -c -o $@ $<
#
#.SECONDEXPANSION:			
#MuPlibs = libHYPRE_utilities_single.a libHYPRE_utilities_double.a libHYPRE_utilities_longdouble.a
#${MuPlibs}: libHYPRE_utilities_%.a: $${ADD_OBJS_%}
#	@echo  "Building $@ ... "
#	ar -rcu $@ $^
#	${RANLIB} $@

# multiprecision object libraries
#libHYPRE_utilities.a: ${MuPlibs}

#UTILITIESLIBS = ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

${COBJS}: CFLAGS += -DMP_BUILD_DEFAULT=1
${CUOBJS}: CFLAGS += -DMP_BUILD_DEFAULT=1
OBJS = ${COBJS} ${CUOBJS} ${ADD_OBJS_single} ${ADD_OBJS_double} ${ADD_OBJS_longdouble}

else
OBJS = ${COBJS} ${CUOBJS}
endif

libHYPRE_utilities.a: ${OBJS}
	@echo "Building $@ ..."
	rm -f $@
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

libHYPRE_utilities.so libHYPRE_utilities.dylib: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s -f ${SONAME} $@


${OBJS}: ${HEADERS}
