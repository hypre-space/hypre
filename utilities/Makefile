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
UTIL_FILES =\
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

FILES = ${UTIL_FILES} ${HYPRE_BLASFILES} ${HYPRE_LAPACKFILES}

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_utilities.a 
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/utilities.h $(HYPRE_BUILD_DIR)/include
	cp -fp $(srcdir)/fortran*.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib
	cp -fp *.o ${HYPRE_SRC_TOP_DIR}/lib
	cp -fp ${HYPRE_SRC_TOP_DIR}/blas/*.h $(HYPRE_BUILD_DIR)/include
	cp -fp ${HYPRE_SRC_TOP_DIR}/lapack/*.h $(HYPRE_BUILD_DIR)/include

install: libHYPRE_utilities.a 
	cp -fp $(srcdir)/HYPRE_*.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/utilities.h $(HYPRE_INC_INSTALL)
	cp -fp $(srcdir)/fortran*.h $(HYPRE_INC_INSTALL)
	cp -fp libHYPRE* $(HYPRE_LIB_INSTALL)
	cp -fp ${HYPRE_SRC_TOP_DIR}/blas/*.h $(HYPRE_INC_INSTALL)/include
	cp -fp ${HYPRE_SRC_TOP_DIR}/lapack/*.h $(HYPRE_INC_INSTALL)/include

clean:
	rm -f *.o libHYPRE* f2c.h *blas.h *lapack.h 

distclean: clean

##################################################################
# Rules
##################################################################
 
libHYPRE_utilities.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ *.o
	${RANLIB} $@

${OBJS}: ${HEADERS}

timer.o: timer.c
	${CC} -D_POSIX_SOURCE -o $@ -c ${CFLAGS} $<
