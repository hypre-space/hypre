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
 mli_vector.h

FILES =\
 mli_vector.cxx

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
