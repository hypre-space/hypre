#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../../../config/Makefile.config


CC= mpicc

C_COMPILE_FLAGS=-O2
LD_LINK_FLAGS=
CINCLUDES= 
CDEFS = -DHYPRE_TIMING
 
CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)/..\
 -I../../../\
 -I../../../utilities\
 -I../../../parcsr_mv\
 -I../../../parcsr_ls\
 -I../../../krylov\
 -I../../../IJ_mv\
 -I../../../seq_mv\
 ${CINCLUDES}\
 ${CDEFS}
 
MPILIBFLAGS =     
LIBFLAGS =   -lm -lg2c
LDLIBFLAGS =   @LIBM@
LAPACKLIBFLAGS =  -L/usr/lib -L/usr/local/lib -llapack
BLASLIBFLAGS =  ${BLASLIBDIRS} ${BLASLIBS}


##################################################################
# Targets
##################################################################

all: driver

install:

clean:
	@rm -f *.o

distclean: clean

##################################################################
# Rules
##################################################################

driver: driver.o
	@echo  "Building" $@ "... "
	${CC} -o $@ $@.o ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}
	@echo  " "
