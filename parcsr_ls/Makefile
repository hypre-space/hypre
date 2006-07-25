#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config
 
## Euclid specific macros taken from distributed_ls/Euclid
EUCLID_CFLAGS = -DMPI_MODE -DMPI_TIMING -DUSING_MPI -DHYPRE_GET_ROW \
                -DHYPRE_MODE -DOPTIMIZED_DH

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS =\
 ${EUCLID_CFLAGS}\
 -I..\
 -I../distributed_ls/Euclid\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../multivector\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../krylov\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 -I$(srcdir)/../distributed_matrix\
 -I$(srcdir)/../matrix_matrix\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../parcsr_block_mv\
 ${CINCLUDES}
 
 
HEADERS =\
 Common.h\
 headers.h\
 HYPRE_parcsr_int.h\
 HYPRE_parcsr_ls.h\
 par_amg.h\
 parcsr_ls.h\
 schwarz.h\
 block_tridiag.h

FILES =\
 amg_hybrid.c\
 F90_hypre_laplace.c\
 F90_HYPRE_parcsr_amg.c\
 F90_HYPRE_parcsr_bicgstab.c\
 F90_HYPRE_parcsr_block.c\
 F90_HYPRE_parcsr_cgnr.c\
 F90_HYPRE_parcsr_Euclid.c\
 F90_HYPRE_parcsr_gmres.c\
 F90_HYPRE_parcsr_hybrid.c\
 F90_HYPRE_parcsr_int.c\
 F90_HYPRE_parcsr_ParaSails.c\
 F90_HYPRE_parcsr_pcg.c\
 F90_HYPRE_parcsr_pilut.c\
 F90_HYPRE_parcsr_schwarz.c\
 HYPRE_parcsr_amg.c\
 HYPRE_parcsr_bicgstab.c\
 HYPRE_parcsr_block.c\
 HYPRE_parcsr_cgnr.c\
 HYPRE_parcsr_Euclid.c\
 HYPRE_parcsr_gmres.c\
 HYPRE_parcsr_hybrid.c\
 HYPRE_parcsr_int.c\
 HYPRE_parcsr_ParaSails.c\
 HYPRE_parcsr_pcg.c\
 HYPRE_parcsr_pilut.c\
 HYPRE_parcsr_schwarz.c\
 HYPRE_ams.c\
 par_amg.c\
 par_amg_setup.c\
 par_amg_solve.c\
 par_amg_solveT.c\
 par_cg_relax_wt.c\
 par_coarsen.c\
 par_coarse_parms.c\
 par_cr.c\
 par_cycle.c\
 par_difconv.c\
 par_gsmg.c\
 par_indepset.c\
 par_interp.c\
 par_jacobi_interp.c\
 par_multi_interp.c\
 par_laplace_27pt.c\
 par_laplace_9pt.c\
 par_laplace.c\
 par_nodal_systems.c\
 par_rap.c\
 par_rap_communication.c\
 par_rotate_7pt.c\
 par_vardifconv.c\
 par_relax.c\
 par_relax_interface.c\
 par_scaled_matnorm.c\
 par_schwarz.c\
 par_stats.c\
 par_strength.c\
 pcg_par.c\
 schwarz.c\
 block_tridiag.c\
 ams.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_ls${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/parcsr_ls.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib
 
install: libHYPRE_parcsr_ls${HYPRE_LIB_SUFFIX}
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/parcsr_ls.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)
 
clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_parcsr_ls${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@
 
${OBJS}: ${HEADERS}
