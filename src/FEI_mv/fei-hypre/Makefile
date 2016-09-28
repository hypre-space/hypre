#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************


include ../../config/Makefile.config

BASE_DIR = ${HYPRE_FEI_BASE_DIR}

SUPERLU_INCLUDE = -I$(srcdir)/../SuperLU/SRC
SUPERLU_LIB     = -L$(srcdir)/../SuperLU

CINCLUDES=${INCLUDES} ${MPIINCLUDE} -I../ml/src/Include
CXXINCLUDES=${INCLUDES} ${MPIINCLUDE} -I../ml/src/Include
CDEFS = -DHAVE_SUPERLU -DBOOL_NOT_SUPPORTED -DHAVE_MLI
CXXDEFS = -DHAVE_SUPERLU -DBOOL_NOT_SUPPORTED -DMPICH_SKIP_MPICXX -DHAVE_MLI

C_COMPILE_FLAGS = \
 ${CDEFS}\
 -I../..\
 -I$(BASE_DIR)\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../IJ_mv\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../multivector\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../parcsr_ls\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../distributed_ls\
 -I$(srcdir)/../fei-base\
 ${SUPERLU_INCLUDE}\
 ${CINCLUDES}

CXX_COMPILE_FLAGS = \
 ${CXXDEFS}\
 -I../..\
 -I$(BASE_DIR)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../lapack\
 -I$(srcdir)/../../IJ_mv\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../multivector\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../parcsr_ls\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../distributed_ls\
 -I$(srcdir)/../femli\
 -I$(srcdir)/../fei-base\
 ${SUPERLU_INCLUDE}\
 ${CXXINCLUDES}

HEADERS =\
 cfei-hypre.h\
 cfei_hypre.h\
 hypre_cfei.h\
 HYPRE_LinSysCore.h\
 HYPRE_FEI_includes.h \
 $(BASE_DIR)/fei_Data.hpp  \
 $(BASE_DIR)/fei_Lookup.hpp  \
 $(BASE_DIR)/fei_LinearSystemCore.hpp \
 $(BASE_DIR)/cfei.h\
 $(BASE_DIR)/fei_bool.h\
 $(BASE_DIR)/fei_defs.h\
 $(BASE_DIR)/fei_mpi.h\
 $(BASE_DIR)/fei_version.h\
 LLNL_FEI_Impl.h\
 LLNL_FEI_Fei.h\
 LLNL_FEI_LSCore.h\
 LLNL_FEI_Solver.h\
 LLNL_FEI_Matrix.h\
 FEI_HYPRE_Impl.h\
 FEI_HYPRE_include.h\
 HYPRE_Builder.h\
 HYPRE_LSI_Dsuperlu.h\
 HYPRE_LSI_UZAWA.h\
 HYPRE_LSI_Uzawa_c.h\
 HYPRE_LSI_blkprec.h\
 HYPRE_LSI_block.h\
 HYPRE_LSI_ddict.h\
 HYPRE_LSI_ddilut.h\
 HYPRE_LSI_mli.h\
 HYPRE_LSI_poly.h\
 HYPRE_LSI_schur.h\
 HYPRE_LSI_schwarz.h\
 HYPRE_MHMatrix.h\
 HYPRE_MLMatrix.h\
 HYPRE_MLMaxwell.h\
 HYPRE_SlideReduction.h\
 HYPRE_parcsr_TFQmr.h\
 HYPRE_parcsr_bicgs.h\
 HYPRE_parcsr_bicgstabl.h\
 HYPRE_parcsr_fgmres.h\
 HYPRE_parcsr_lsicg.h\
 HYPRE_parcsr_symqmr.h\
 fei_mv.h

FILESC = \
 HYPRE_LSI_Dsuperlu.c\
 HYPRE_LSI_ddict.c\
 HYPRE_LSI_ddilut.c\
 HYPRE_LSI_ml.c\
 HYPRE_LSI_mlmaxwell.c\
 HYPRE_LSI_poly.c\
 HYPRE_LSI_schwarz.c\
 HYPRE_parcsr_TFQmr.c\
 HYPRE_parcsr_bicgs.c\
 HYPRE_parcsr_bicgstabl.c\
 HYPRE_parcsr_fgmres.c\
 HYPRE_parcsr_lsicg.c\
 HYPRE_parcsr_symqmr.c\
 HYPRE_parcsr_maxwell.c\
 SymQMR.c\
 TFQmr.c\
 bicgs.c\
 bicgstabl.c\
 fgmres.c\
 hypre_lsi_amge.c\
 hypre_lsi_ddamg.c\
 hypre_lsi_misc.c\
 lsicg.c

FILESCXX = \
 FEI_HYPRE_Impl.cxx\
 LLNL_FEI_Impl.cxx\
 LLNL_FEI_Fei.cxx\
 LLNL_FEI_LSCore.cxx\
 LLNL_FEI_Solver.cxx\
 LLNL_FEI_Matrix.cxx\
 HYPRE_LSC_aux.cxx\
 HYPRE_LSI_UZAWA.cxx\
 HYPRE_LSI_blkprec.cxx\
 HYPRE_LSI_mli.cxx\
 HYPRE_LinSysCore.cxx\
 HYPRE_SlideReduction.cxx\
 cfei_hypre.cxx\
 hypre_cfei.cxx\
 hypre_schur_reduce.cxx\
 hypre_slide_reduce.cxx\
 HYPRE_fei_mesh.cxx\
 HYPRE_fei_matrix.cxx\
 HYPRE_fei_vector.cxx

OBJSC = ${FILESC:.c=.o}
OBJSCXX = ${FILESCXX:.cxx=.o}
OBJS = ${OBJSC} ${OBJSCXX}

SONAME = libHYPRE_FEI-${HYPRE_RELEASE_VERSION}.so

##################################################################
# Targets
##################################################################

all: libHYPRE_FEI${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/cfei-hypre.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/cfei_hypre.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/hypre_cfei.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/HYPRE_LinSysCore.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/HYPRE_FEI*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/LLNL_FEI*.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_Data.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_Lookup.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_LinearSystemCore.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/cfei.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_bool.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_defs.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_mpi.h $(HYPRE_BUILD_DIR)/include

install: libHYPRE_FEI${HYPRE_LIB_SUFFIX}
	cp -fR $(srcdir)/cfei-hypre.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/cfei_hypre.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/hypre_cfei.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(srcdir)/HYPRE_LinSysCore.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/HYPRE_FEI*.h $(HYPRE_INC_INSTALL)
	cp -fR $(srcdir)/LLNL_FEI*.h $(HYPRE_INC_INSTALL)
	cp -fR $(BASE_DIR)/fei_Data.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_Lookup.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_LinearSystemCore.hpp $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/cfei.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_bool.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_defs.h $(HYPRE_BUILD_DIR)/include
	cp -fR $(BASE_DIR)/fei_mpi.h $(HYPRE_BUILD_DIR)/include
	@echo  ""

clean:
	rm -rf *.o libHYPRE* 
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_FEI.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_FEI.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o ${SONAME} ${OBJS} ${SHARED_SET_SONAME}${SONAME}
	ln -s ${SONAME} $@

${OBJS}: ${HEADERS}

