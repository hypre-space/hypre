# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1999   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.1 $
#EHEADER***********************************************************************

.SUFFIXES: .c .f .o

srcdir = .

CC =  mpicc

C_COMPILE_FLAGS= -O 
CXX_COMPILE_FLAGS= -O 
F77_COMPILE_FLAGS= -O -silent 
CINCLUDES= 
 
CFLAGS = \
 ${C_COMPILE_FLAGS}\
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
 ${CINCLUDES}\
 ${CDEFS}

RANLIB = ranlib

MPILIBFLAGS =    
LIBFLAGS =   -lm
LDLIBFLAGS =  
BLASLIBFLAGS = -lHYPRE_blas

LFLAGS =\
 -L.\
 -L../../blas\
 -L../../utilities\
 -L../../distributed_matrix\
 -lHYPRE_utilities\
 -lHYPRE_ParaSails\
 ${BLASLIBFLAGS}\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}


HEADERS =\
 Common.h\
 ConjGrad.h\
 DiagScale.h\
 hypre_ParaSails.h\
 Hash.h\
 LoadBal.h\
 Matrix.h\
 Mem.h\
 Numbering.h\
 OrderStat.h\
 ParaSails.h\
 PrunedRows.h\
 RowPatt.h\
 StoredRows.h

FILES =\
 ConjGrad.c\
 DiagScale.c\
 FlexGmres.c\
 hypre_ParaSails.c\
 Hash.c\
 LoadBal.c\
 Matrix.c\
 Mem.c\
 Numbering.c\
 OrderStat.c\
 ParaSails.c\
 PrunedRows.c\
 RowPatt.c\
 StoredRows.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_ParaSails.a 

driver: driver.o libHYPRE_ParaSails.a
	@echo  "Linking" $@ "... "
	${CC} -o $@ driver.o ${LFLAGS}

convert: convert.o
	${CC} -o $@ convert.o

#install: all
#	@cp -f $(srcdir)/hypre_*.h $$HYPRE_INSTALL_DIR/include/.
#	@cp -f libHYPRE_*.a $$HYPRE_INSTALL_DIR/lib/.

install: all
	@cp -f libHYPRE_*.a $$HYPRE_INSTALL_DIR/lib/.

clean:
	@rm -f *.o

veryclean: clean
	@rm -f libHYPRE_*.a


##################################################################
# Rules
##################################################################

libHYPRE_ParaSails.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}



##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<

.f.o:
	${F77} -o $@ -c ${FFLAGS} $<


