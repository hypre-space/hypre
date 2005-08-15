#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I.\
 -I$(srcdir)/..\
 -I$(srcdir)/../utilities\
 ${CINCLUDES}

HEADERS =\
 multivector.h\
 temp_multivector.h

FILES =\
 multivector.c\
 temp_multivector.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_multivector.a 
	cp -fp *.h $(HYPRE_BUILD_DIR)/include
	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib
	@\
	for file in ${OBJS}; \
	do \
	   (cp -fp $$file ${HYPRE_SRC_TOP_DIR}/lib); \
	done

install: libHYPRE_multivector.a 
	cp -f *.h $(HYPRE_INC_INSTALL)
	cp -f libHYPRE* $(HYPRE_LIB_INSTALL)

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*

distclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_multivector.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
