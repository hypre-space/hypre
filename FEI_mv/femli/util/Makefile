#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../../../config/Makefile.config

CC =  mpicc
F77 = mpif77

C_COMPILE_FLAGS=-O2 
CINCLUDES= 

F_COMPILE_FLAGS=-O
FINCLUDES= 
 
CFLAGS =\
 ${C_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../fei-hypre\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../eigensolvers/krylov\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../distributed_ls\
 -I../../../distributed_matrix\
 -I../../../seq_mv\
 ${CINCLUDES}
 
FFLAGS =\
 ${F_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../fei-hypre\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../eigensolvers/krylov\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../distributed_ls\
 -I../../../distributed_matrix\
 -I../../../seq_mv\
 ${FINCLUDES}

HEADERS =\
 mli_utils.h

FILES =\
 mli_olapdd.c\
 mli_utils.c\
 mli_utils_fortran.f

TEMP = ${FILES:.f=.o}
OBJS = ${TEMP:.c=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS} 

install: all

clean:
	@rm -f *.o

distclean: clean


##################################################################
# Rules
##################################################################

${OBJS}: ${HEADERS}
