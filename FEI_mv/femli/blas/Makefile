# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 2001   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .f .C .c .o

srcdir = .

F77 = mpif77
CC  = mpicc
CC_COMPILE_FLAGS= -O 
F77_COMPILE_FLAGS= -O 
F77INCLUDES= 
F77DEFS =
RANLIB = ranlib

CFLAGS =\
 ${C_COMPILE_FLAGS}\
 ${CINCLUDES}\
 ${CDEFS}
 
F77FLAGS =\
 ${F77_COMPILE_FLAGS}\
 ${F77INCLUDES}\
 ${F77DEFS}
 
FILES = \
ddot.f    drot.f    dsdot.f  dsymv.f  dtbsv.f   dtrsv.f  izamax.f \
dgbmv.f   drotg.f   dspmv.f  dsyr2.f  dtpmv.f   dzasum.f lsame.f \
dasum.f   dgemm.f  drotm.f   dspr2.f  dsyr2k.f	dtpsv.f  dznrm2.f \
daxpy.f   dgemv.f  drotmg.f  dspr.f   dsyr.f	dtrmm.f  icamax.f  xerbla.f \
dcabs1.f  dger.f   dsbmv.f   dswap.f  dsyrk.f	dtrmv.f  idamax.f \
dcopy.f   dnrm2.f  dscal.f   dsymm.f  dtbmv.f	dtrsm.f  isamax.f\
s_cmp.c s_copy.c   dmyblas2.c

TEMP = ${FILES:.f=.o}
OBJS = ${TEMP:.c=.o}

##################################################################
# Targets
##################################################################

libblas.a: ${OBJS}
	@echo  "Building $@ ... "
	@ar -rcu libblas.a ${OBJS}
	${RANLIB} $@
 
install: all
	@cp -f *.o ../lib
#@cp -f $(srcdir)/*.h $$HYPRE_INSTALL_DIR/include/.
 
clean:
	@rm -f *.o

veryclean: clean

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} $<
.f.o:
	${F77} -o $@ -c ${F77FLAGS} $<

