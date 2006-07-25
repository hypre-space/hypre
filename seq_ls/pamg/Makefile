# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
LIBFLAGS = ${LIBDIRS} ${LIBS}

C_COMPILE_FLAGS =\
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../krylov\
 -I$(srcdir)/../../seq_mv\
 ${CINCLUDES}
 
LFLAGS =\
 ${LD_LINK_FLAGS}\
 -L.\
 -L../../seq_mv\
 -L../../utilities\
 -L../../krylov\
 -lHYPRE_amg_ls\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 -lHYPRE_krylov\
 ${MPILIBFLAGS} ${LIBS}
 

HEADERS =\
 headers.h\
 amg.h\
 pcg.h\
 bcsr_matrix.h\
 bcsr_matrix_dense_block.h\
 pamg.h\
 HYPRE_ls.h

FILES =\
 amg.c\
 amgstats.c\
 amg_setup.c\
 amg_solve.c\
 Atrunc.c\
 bcsr_interp.c\
 bcsr_matrix.c\
 bcsr_matrix_dense_block.c\
 bcsr_relax.c\
 cg_fun.c\
 coarsen.c\
 coarsenCR.c\
 cycle.c\
 difconv.c\
 indepset.c\
 interp.c\
 interpCR.c\
 interpRBM.c\
 inx_part_of_u_interp.c\
 laplace.c\
 laplace_9pt.c\
 laplace_27pt.c\
 pcg.c\
 random.c\
 rap.c\
 relax.c\
 schwarz.c\
 scaled_matnorm.c\
 strength.c\
 SPamg-pcg.c\
 stencil_matrix.c\
 trunc.c\
 HYPRE_amg.c\
 HYPRE_csr_pcg.c\
 HYPRE_csr_gmres.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

lib: libHYPRE_amg_ls${HYPRE_LIB_SUFFIX}

all: libHYPRE_amg_ls${HYPRE_LIB_SUFFIX} driver

install: all

clean:
	rm -f *.o lib*

distclean: clean
	rm -f driver

##################################################################
# Rules
##################################################################

driver: driver.o libHYPRE_amg_ls${HYPRE_LIB_SUFFIX}
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

libHYPRE_amg_ls${HYPRE_LIB_SUFFIX}: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
