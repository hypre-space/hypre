# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.0 $
#EHEADER***********************************************************************


.SUFFIXES:
.SUFFIXES: .c .f .o

srcdir = .

default: lib

all: lib

install:all
	@cp -f $(srcdir)/HYPRE_*.h $$HYPRE_INSTALL_DIR/include/.
	@cp -f libHYPRE_*.a $$HYPRE_INSTALL_DIR/lib/.

HEADERS =
 
HYPREFILES =\
 HYPRE_ConvertPETScMatrixToDistributedMatrix.c \
 HYPRE_ConvertParCSRMatrixToDistributedMatrix.c \
 HYPRE_BuildIJMatrixFromDistributedMatrix.c

FILES = $(HYPREFILES) $(LOCALFILES)

OBJS = ${FILES:.c=.o}


CC =  mpicc
CXX =  mpiCC
F77 = mpif77

C_COMPILE_FLAGS= -O 
CXX_COMPILE_FLAGS= -O 
F77_COMPILE_FLAGS= -O -silent 
CINCLUDES= 
CDEFS = 

CFLAGS = \
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 -I$(srcdir)/../matrix_matrix\
 -I$(srcdir)/../distributed_matrix\
 -I$(srcdir)/../IJ_mv\
 -I$(srcdir)/../seq_mv\
 -I$(srcdir)/../parcsr_mv\
 \
 ${C_COMPILE_FLAGS}\
 ${CINCLUDES} ${CDEFS}

CXXFLAGS = \
  \
 ${CXX_COMPILE_FLAGS}\
 -I..\
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}\
 ${COPT}\
 ${CDEBUG} ${CDEFS}

RANLIB = ranlib

##################################################################
# Main rules
##################################################################

lib: $(PROTOS) ${OBJS}
	@echo  "Building $@ ... "
	@ar -rcu libHYPRE_MatrixMatrix.a ${OBJS}
	${RANLIB} libHYPRE_MatrixMatrix.a

${OBJS}: ${HEADERS}

##################################################################
# Targets
##################################################################
clean:
	@rm -f *.o *.dxx 

veryclean: clean
	@rm -f lib*.a


##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

.cc.o:
	${CXX} -o $@ -c ${CFLAGS} $<

.f.o:
	${F77} -o $@ -c ${FFLAGS} $<


