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

CINCLUDES = ${INCLUDES} ${MPIINCLUDES}

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
 -lHYPRE_amg_ls\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 -L../../krylov -lkrylov\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}
 

HEADERS =\
 headers.h\
 amg.h\
 pcg.h\
 pamg.h\
 HYPRE_ls.h

FILES =\
 driver.c\
 amg.c\
 amgstats.c\
 amg_setup.c\
 amg_solve.c \
 cg_fun.c\
 coarsen.c\
 coarsenCR.c\
 cycle.c\
 difconv.c\
 indepset.c\
 interp.c\
 interpCR.c\
 interpRBM.c\
 laplace.c\
 laplace_9pt.c\
 laplace_27pt.c\
 random.c\
 rap.c \
 relax.c\
 schwarz.c\
 scaled_matnorm.c\
 HYPRE_amg.c\
 HYPRE_csr_pcg.c\
 HYPRE_csr_gmres.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_amg_ls.a driver

install: all

clean:
	rm -f *.o lib*

distclean: clean
	rm -f driver

##################################################################
# Rules
##################################################################

driver: driver.o libHYPRE_amg_ls.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

libHYPRE_amg_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
