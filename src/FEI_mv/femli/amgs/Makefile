#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 1.5 $
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
 mli_method.h\
 mli_method_amgcr.h\
 mli_method_amgrs.h\
 mli_method_amgsa.h

FILES =\
 mli_amgsa_calib.cxx\
 mli_amgsa_coarsen1.cxx\
 mli_amgsa_dd_fedata.cxx\
 mli_amgsa_dd_sfei.cxx\
 mli_method.cxx\
 mli_method_amgcr.cxx\
 mli_method_amgrs.cxx\
 mli_method_amgsa.cxx

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
# Rules
##################################################################

${OBJS}: ${HEADERS}
