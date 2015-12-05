#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.1 $
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 all_krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 krylov.h\
 pcg.h

FILES =\
 bicgstab.c\
 cgnr.c\
 gmres.c\
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_pcg.c\
 pcg.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_krylov.a 
	cp -fp $(srcdir)/krylov.h $(HYPRE_BUILD_DIR)/include/.
	cp -fp lib*.a $(HYPRE_BUILD_DIR)/lib/.

install: all
	cp -fp $(srcdir)/krylov.h $(HYPRE_INC_INSTALL)/.
	cp -fp lib*.a $(HYPRE_LIB_INSTALL)/.

clean:
	rm -f *.o lib*.a

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_krylov.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
