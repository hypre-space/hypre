#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../../../config/Makefile.config

CXX =  mpiCC

CXX_COMPILE_FLAGS=-O2 
CXXINCLUDES= 
 
CXXFLAGS =\
 ${CXX_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../seq_mv\
 ${CXXINCLUDES}

HEADERS =\
 mli_solver_arpacksuperlu.h\
 mli_solver_bjacobi.h\
 mli_solver_bsgs.h\
 mli_solver_cg.h\
 mli_solver_chebyshev.h\
 mli_solver_gmres.h\
 mli_solver_gs.h\
 mli_solver.h\
 mli_solver_hschwarz.h\
 mli_solver_hsgs.h\
 mli_solver_jacobi.h\
 mli_solver_kaczmarz.h\
 mli_solver_mli.h\
 mli_solver_mls.h\
 mli_solver_parasails.h\
 mli_solver_seqsuperlu.h\
 mli_solver_sgs.h\
 mli_solver_superlu.h

FILES =\
 mli_solver_arpacksuperlu.cxx\
 mli_solver_bjacobi.cxx\
 mli_solver_bsgs.cxx\
 mli_solver_cg.cxx\
 mli_solver_chebyshev.cxx\
 mli_solver_gmres.cxx\
 mli_solver_gs.cxx\
 mli_solver.cxx\
 mli_solver_hschwarz.cxx\
 mli_solver_hsgs.cxx\
 mli_solver_jacobi.cxx\
 mli_solver_kaczmarz.cxx\
 mli_solver_mli.cxx\
 mli_solver_mls.cxx\
 mli_solver_parasails.cxx\
 mli_solver_seqsuperlu.cxx\
 mli_solver_sgs.cxx\
 mli_solver_superlu.cxx

OBJS = ${FILES:.cxx=.o}

##################################################################
# Targets
##################################################################

all: ${OBJS} 

install: all

clean:
	@rm -f *.o

distclean: clean


##################################################################

##################################################################

${OBJS}: ${HEADERS}
