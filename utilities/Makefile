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
${LAPACKNEED}LAPACK_HEADERS = hypre_lapack.h f2c.h
${LAPACKNEED}LAPACK_FILES = \
${LAPACKNEED}dbdsqr.o\
${LAPACKNEED}dgebd2.o\
${LAPACKNEED}dgebrd.o\
${LAPACKNEED}dgelq2.o\
${LAPACKNEED}dgelqf.o\
${LAPACKNEED}dgels.o\
${LAPACKNEED}dgeqr2.o\
${LAPACKNEED}dgeqrf.o\
${LAPACKNEED}dgesvd.o\
${LAPACKNEED}dlabad.o\
${LAPACKNEED}dlabrd.o\
${LAPACKNEED}dlacpy.o\
${LAPACKNEED}dlae2.o\
${LAPACKNEED}dlaev2.o\
${LAPACKNEED}dlamch.o\
${LAPACKNEED}dlange.o\
${LAPACKNEED}dlanst.o\
${LAPACKNEED}dlansy.o\
${LAPACKNEED}dlapy2.o\
${LAPACKNEED}dlarfb.o\
${LAPACKNEED}dlarf.o\
${LAPACKNEED}dlarfg.o\
${LAPACKNEED}dlarft.o\
${LAPACKNEED}dlartg.o\
${LAPACKNEED}dlas2.o\
${LAPACKNEED}dlascl.o\
${LAPACKNEED}dlaset.o\
${LAPACKNEED}dlasq1.o\
${LAPACKNEED}dlasq2.o\
${LAPACKNEED}dlasq3.o\
${LAPACKNEED}dlasq4.o\
${LAPACKNEED}dlasq5.o\
${LAPACKNEED}dlasq6.o\
${LAPACKNEED}dlasr.o\
${LAPACKNEED}dlasrt.o\
${LAPACKNEED}dlassq.o\
${LAPACKNEED}dlasv2.o\
${LAPACKNEED}dlatrd.o\
${LAPACKNEED}dorg2l.o\
${LAPACKNEED}dorg2r.o\
${LAPACKNEED}dorgbr.o\
${LAPACKNEED}dorgl2.o\
${LAPACKNEED}dorglq.o\
${LAPACKNEED}dorgql.o\
${LAPACKNEED}dorgqr.o\
${LAPACKNEED}dorgtr.o\
${LAPACKNEED}dorm2r.o\
${LAPACKNEED}dormbr.o\
${LAPACKNEED}dorml2.o\
${LAPACKNEED}dormlq.o\
${LAPACKNEED}dormqr.o\
${LAPACKNEED}dpotf2.o\
${LAPACKNEED}dpotrf.o\
${LAPACKNEED}dpotrs.o\
${LAPACKNEED}dsteqr.o\
${LAPACKNEED}dsterf.o\
${LAPACKNEED}dsyev.o\
${LAPACKNEED}dsygs2.o\
${LAPACKNEED}dsygst.o\
${LAPACKNEED}dsygv.o\
${LAPACKNEED}dsytd2.o\
${LAPACKNEED}dsytrd.o\
${LAPACKNEED}ieeeck.o\
${LAPACKNEED}ilaenv.o\
${LAPACKNEED}lapack_utils.o\
${LAPACKNEED}lsame.o\
${LAPACKNEED}xerbla.o

##################################################################
# Blas Header & Object Files
##################################################################
${BLASNEED}BLAS_HEADERS = hypre_blas.h f2c.h
${BLASNEED}BLAS_FILES = \
${BLASNEED}blas_utils.o\
${BLASNEED}dasum.o\
${BLASNEED}daxpy.o\
${BLASNEED}dcopy.o\
${BLASNEED}ddot.o\
${BLASNEED}dgemm.o\
${BLASNEED}dgemv.o\
${BLASNEED}dger.o\
${BLASNEED}dnrm2.o\
${BLASNEED}drot.o\
${BLASNEED}dscal.o\
${BLASNEED}dswap.o\
${BLASNEED}dsymm.o\
${BLASNEED}dsymv.o\
${BLASNEED}dsyr2.o\
${BLASNEED}dsyr2k.o\
${BLASNEED}dsyrk.o\
${BLASNEED}dtrmm.o\
${BLASNEED}dtrmv.o\
${BLASNEED}dtrsm.o\
${BLASNEED}dtrsv.o\
${BLASNEED}idamax.o

##################################################################
# Utilities Header Files
##################################################################
HEADERS =\
 amg_linklist.h\
 exchange_data.h\
 fortran.h\
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
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp $(srcdir)/utilities.h $(HYPRE_BUILD_DIR)/include/.
	${BLASNEED}cp -fp ${BLAS_HEADERS} $(HYPRE_BUILD_DIR)/include/.
	${LAPACKNEED}cp -fp ${LAPACK_HEADERS} $(HYPRE_BUILD_DIR)/include/.
	cp -fp libHYPRE_*.a $(HYPRE_BUILD_DIR)/lib/.

install: libHYPRE_utilities.a 
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)/.
	cp -fp $(srcdir)/utilities.h $(HYPRE_INC_INSTALL)/.
	${BLASNEED}cp -fp ${BLAS_HEADERS} $(HYPRE_INC_INSTALL)/.
	${LAPACKNEED}cp -fp ${LAPACK_HEADERS} $(HYPRE_INC_INSTALL)/.
	cp -fp libHYPRE_*.a $(HYPRE_LIB_INSTALL)/.

clean:
	rm -f *.o lib*.a ${BLAS_HEADERS} ${LAPACK_HEADERS} 

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
