#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/utilities\
 -I$(srcdir)/IJ_mv\
 -I$(srcdir)/parcsr_mv\
 -I$(srcdir)/parcsr_ls\
 -I$(srcdir)/seq_mv\
 -I$(srcdir)/krylov\
 ${CINCLUDES}

HEADERS =\
 HYPRE_lobpcg.h\
 lobpcg.h\
 mmio.h

FILES =\
 lobpcg.c\
 lobpcg_matrix.c\
 lobpcg_utilities.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_lobpcg.a 
	@cp -fp $(srcdir)/HYPRE*.h $(HYPRE_BUILD_DIR)/include/.
	@cp -fp $(srcdir)/lobpcg.h $(HYPRE_BUILD__DIR)/include/.
	@cp -fp lib*.a $(HYPRE_BUILD_DIR)/lib/.

install: libHYPRE_lobpcg.a 
	@cp -fp $(srcdir)/HYPRE*.h $(HYPRE_INC_INSTALL)/.
	@cp -fp $(srcdir)/lobpcg.h $(HYPRE_INC_INSTALL)/.
	@cp -fp lib*.a $(HYPRE_LIB_INSTALL)/.

clean:
	@rm -f *.o lib*.a

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_lobpcg.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
