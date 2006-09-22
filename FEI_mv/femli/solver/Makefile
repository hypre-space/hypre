#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************




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
 mli_solver_amg.h\
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
 mli_solver_amg.cxx\
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
