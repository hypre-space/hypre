#BHEADER***********************************************************************
# (c) 1999   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.1 $
#EHEADER***********************************************************************

include ../../config/Makefile.config

########Euclid specific macros:
EUCLID_CFLAGS =  -DMPI_MODE -DMPI_TIMING -DUSING_MPI -DHYPRE_GET_ROW \
                 -DHYPRE_MODE -DOPTIMIZED_DH

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

C_COMPILE_FLAGS = \
 ${EUCLID_CFLAGS} \
 -I../..\
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../parcsr_mv\
 -I$(srcdir)/../../seq_mv\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../hypre/include\
 ${CINCLUDES}

HEADERS =\
 blas_dh.h\
 euclid_common.h\
 euclid_config.h\
 Euclid_dh.h\
 euclid.h\
 ExternalRows_dh.h\
 Factor_dh.h\
 getRow_dh.h\
 guards_dh.h\
 Hash_dh.h\
 Hash_i_dh.h\
 ilu_dh.h\
 io_dh.h\
 krylov_dh.h\
 macros_dh.h\
 Mat_dh.h\
 mat_dh_private.h\
 MatGenFD.h\
 Mem_dh.h\
 Numbering_dh.h\
 Parser_dh.h\
 shellSort_dh.h\
 sig_dh.h\
 SortedList_dh.h\
 SortedSet_dh.h\
 SubdomainGraph_dh.h\
 TimeLog_dh.h\
 Timer_dh.h\
 Vec_dh.h

FILES =\
 blas_dh.c\
 Euclid_apply.c\
 Euclid_dh.c\
 ExternalRows_dh.c\
 Factor_dh.c\
 getRow_dh.c\
 globalObjects.c\
 Hash_dh.c\
 Hash_i_dh.c\
 ilu_mpi_bj.c\
 ilu_mpi_pilu.c\
 ilu_seq.c\
 io_dh.c\
 krylov_dh.c\
 Mat_dh.c\
 mat_dh_private.c\
 MatGenFD.c\
 Mem_dh.c\
 Numbering_dh.c\
 Parser_dh.c\
 shellSort_dh.c\
 sig_dh.c\
 SortedList_dh.c\
 SortedSet_dh.c\
 SubdomainGraph_dh.c\
 TimeLog_dh.c\
 Timer_dh.c\
 Vec_dh.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_Euclid.a 
	cp -fp libHYPRE_*.a $(HYPRE_BUILD_DIR)/lib/.

install: libHYPRE_Euclid.a 
	cp -fp libHYPRE_*.a $(HYPRE_LIB_INSTALL)/.

clean:
	rm -f *.o lib*.a

distclean: clean


##################################################################
# Rules
##################################################################

libHYPRE_Euclid.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
