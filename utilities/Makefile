# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${TIMERDEFS} -DHYPRE_TIMING

C_COMPILE_FLAGS =\
   -I..\
   -I$(srcdir)/..\
   -I$(srcdir)\
   ${CINCLUDES}\
   ${CDEFS}

##################################################################
# Lapack Header & Object Files
##################################################################
LAPACK_HEADERS = hypre_lapack.h f2c.h
LAPACK_FILES = \
dbdsqr.o\
dgebd2.o\
dgebrd.o\
dgelq2.o\
dgelqf.o\
dgels.o\
dgeqr2.o\
dgeqrf.o\
dgesvd.o\
dlabad.o\
dlabrd.o\
dlacpy.o\
dlae2.o\
dlaev2.o\
dlamch.o\
dlange.o\
dlanst.o\
dlansy.o\
dlapy2.o\
dlarfb.o\
dlarf.o\
dlarfg.o\
dlarft.o\
dlartg.o\
dlas2.o\
dlascl.o\
dlaset.o\
dlasq1.o\
dlasq2.o\
dlasq3.o\
dlasq4.o\
dlasq5.o\
dlasq6.o\
dlasr.o\
dlasrt.o\
dlassq.o\
dlasv2.o\
dlatrd.o\
dorg2l.o\
dorg2r.o\
dorgbr.o\
dorgl2.o\
dorglq.o\
dorgql.o\
dorgqr.o\
dorgtr.o\
dorm2r.o\
dormbr.o\
dorml2.o\
dormlq.o\
dormqr.o\
dpotf2.o\
dpotrf.o\
dpotrs.o\
dsteqr.o\
dsterf.o\
dsyev.o\
dsygs2.o\
dsygst.o\
dsygv.o\
dsytd2.o\
dsytrd.o\
ieeeck.o\
ilaenv.o\
lapack_utils.o\
lsame.o\
xerbla.o

##################################################################
# Blas Header & Object Files
##################################################################
BLAS_HEADERS = hypre_blas.h f2c.h
BLAS_FILES = \
blas_utils.o\
dasum.o\
daxpy.o\
dcopy.o\
ddot.o\
dgemm.o\
dgemv.o\
dger.o\
dnrm2.o\
drot.o\
dscal.o\
dswap.o\
dsymm.o\
dsymv.o\
dsyr2.o\
dsyr2k.o\
dsyrk.o\
dtrmm.o\
dtrmv.o\
dtrsm.o\
dtrsv.o\
idamax.o

##################################################################
# Utilities Header Files
##################################################################
HEADERS =\
 amg_linklist.h\
 exchange_data.h\
 fortran.h\
 fortran_matrix.h\
 general.h\
 hypre_memory.h\
 hypre_smp_forloop.h\
 HYPRE_utilities.h\
 mpistubs.h\
 threading.h\
 thread_mpistubs.h\
 timing.h\
 umalloc_local.h\
 utilities.h

##################################################################
# Utilities Files
##################################################################
FILES =\
 amg_linklist.c\
 binsearch.c\
 exchange_data.c\
 fortran_matrix.c\
 hypre_memory.c\
 hypre_qsort.c\
 memory_dmalloc.c\
 mpistubs.c\
 qsplit.c\
 random.c\
 threading.c\
 thread_mpistubs.c\
 timer.c\
 timing.c\
 umalloc_local.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_utilities.a 
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/utilities.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/fortran*.h $(HYPRE_BUILD_DIR)/include
	cp -fp ${BLAS_HEADERS} $(HYPRE_BUILD_DIR)/include
	cp -fp ${LAPACK_HEADERS} $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib
	@\
	for file in ${OBJS}; \
	do \
	   (cp -fp $$file ${HYPRE_SRC_TOP_DIR}/lib); \
	done

install: libHYPRE_utilities.a 
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/utilities.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/fortran*.h $(HYPRE_INC_INSTALL)
	cp -fp ${BLAS_HEADERS} $(HYPRE_INC_INSTALL)
	cp -fp ${LAPACK_HEADERS} $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE* ${BLAS_HEADERS} ${LAPACK_HEADERS} 

distclean: clean

##################################################################
# Rules
##################################################################
 
libHYPRE_utilities.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS} ${LAPACK_FILES} ${BLAS_FILES}
	${RANLIB} $@

${OBJS}: ${HEADERS}

timer.o: timer.c
	${CC} -D_POSIX_SOURCE -o $@ -c ${C_FLAGS} $<
