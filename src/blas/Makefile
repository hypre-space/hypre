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

HEADERS =\
 f2c.h\
 hypre_blas.h

FILES =\
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

##################################################################
# Targets
##################################################################

all: ${OBJS}
	cp -fR $(srcdir)/_hypre_blas.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_BUILD_DIR)/include

install: all
	cp -fR $(srcdir)/_hypre_blas.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/_hypre_*.h $(HYPRE_INC_INSTALL)

clean:
	rm -rf *.o
	rm -f *.o_flt *.o_dbl *.o_ldbl

distclean: clean

##################################################################
# Rules
##################################################################

${OBJS}: ${HEADERS}
