#######################################################################
#
#  This makefile creates a library for distributed SuperLU.
#  The files are organized as follows:
#
#       ALLAUX  -- Auxiliary routines called from all precisions
#       DSLUSRC -- Double precision real serial SuperLU routines
#       DPLUSRC -- Double precision real parallel SuperLU routines
#       ZSLUSRC -- Double precision complex serial SuperLU routines
#       ZPLUSRC -- Double precision complex parallel SuperLU routines
#
#  The library can be set up to include routines for any combination
#  of the two precisions.  To create or add to the library, enter make
#  followed by one or more of the precisions desired.  Some examples:
#       make double
#       make double complex16
#  Alternatively, the command
#       make
#  without any arguments creates a library of all two precisions.
#  The library is called
#       superlu.a
#  and is created at the next higher directory level.
#
#  To remove the object files after the library is created, enter
#       make clean
#
#######################################################################
#include ../make.inc
#
include ../../../config/Makefile.config
CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../../../\
 -I$(srcdir)/../../../utilities\
 ${CINCLUDES}

# Precision independent routines
#
ALLAUX 	= sp_ienv.o etree.o sp_colorder.o get_perm_c.o \
	  mmd.o comm.o memory.o util.o superlu_grid.o \
	  pxerbla.o superlu_timer.o GetDiagU.o mc64ad.o mc21.o symbfact.o \
	  psymbfact.o psymbfact_util.o \
	  get_perm_c_parmetis.o

#### LAPACK auxiliary routines
LAAUX	= lsame.o xerbla.o slamch.o dlamch.o

#
# Routines to permute large entries to diagonal
#
# DLDPERM = dldperm.o mc64ad.o mc21.o #fd05.o

#
# Routines literally taken from SuperLU
#
DSLUSRC	= dlangs.o dgsequ.o dlaqgs.o dutil.o \
	  dmemory.o dmyblas2.o dsp_blas2.o dsp_blas3.o
ZSLUSRC	= dcomplex.o zlangs.o zgsequ.o zlaqgs.o zutil.o \
	  zmemory.o zmyblas2.o dmemory.o zsp_blas2.o zsp_blas3.o

#
# Routines for double precision parallel SuperLU
DPLUSRC = dldperm.o ddistribute.o pdgstrf_irecv.o pdgstrs_Bglobal.o \
	  pdgstrs1.o pdgssvx_ABglobal.o pdgsrfs_ABXglobal.o pdgsmv_AXglobal.o\
	  pdgssvx.o pdgstrs.o pddistribute.o pdlangs.o pdutil.o \
	  pdgsequ.o pdlaqgs.o pdgsrfs.o pdgsmv.o pdgstrs_lsum.o \
	  pdsymbfact_distdata.o

#
# Routines for double complex parallel SuperLU
ZPLUSRC = zldperm.o zdistribute.o pzgssvx_ABglobal.o pzgstrf_irecv.o \
	  pzgstrs1.o pzgstrs_Bglobal.o pzgsrfs_ABXglobal.o pzgsmv_AXglobal.o \
	  pzgssvx.o pzgstrs.o pzdistribute.o pzlangs.o pzutil.o \
	  pzgsequ.o pzlaqgs.o pzgsrfs.o pzgsmv.o pzgstrs_lsum.o \
	  pzsymbfact_distdata.o

#######################################
### changed for hypre
#######################################

FILES =\
 dldperm.c \
 ddistribute.c \
 pdgstrf_irecv.c \
 pdgstrs_Bglobal.c \
 pdgstrs1.c \
 pdgssvx_ABglobal.c \
 pdgsrfs_ABXglobal.c \
 pdgsmv_AXglobal.c \
 pdgssvx.c \
 pdgstrs.c \
 pddistribute.c \
 pdlangs.c \
 pdutil.c \
 pdgsequ.c \
 pdlaqgs.c \
 pdgsrfs.c \
 pdgsmv.c \
 pdgstrs_lsum.c \
 pdsymbfact_distdata.c \
 get_perm_c_parmetis.o \
 sp_ienv.c \
 etree.c \
 sp_colorder.c \
 get_perm_c.c \
 mmd.c \
 comm.c \
 memory.c \
 util.c \
 superlu_grid.c \
 pxerbla.c \
 superlu_timer.c \
 GetDiagU.c \
 symbfact.c \
 psymbfact.c \
 psymbfact_util.c \
 lsame.c \
 xerbla.c \
 dlangs.c \
 dgsequ.c \
 dlaqgs.c \
 dutil.c \
 dmemory.c \
 slamch.c \
 dmyblas2.c \
 dsp_blas2.c \
 dsp_blas3.c \
 hypre_interface.c

FILES2 =\
 mc64ad.f \
 mc21.f

OBJS = ${FILES:.c=.o} ${FILES2:.f=.o}

all: libHYPRE_dsuperlu${HYPRE_LIB_SUFFIX}
	cp -fp *.h $(HYPRE_BUILD_DIR)/include
#	cp -fp libHYPRE* $(HYPRE_BUILD_DIR)/lib

install: libHYPRE_dsuperlu${HYPRE_LIB_SUFFIX}
	cp -f *.h $(HYPRE_INC_INSTALL)
#	cp -f libHYPRE* $(HYPRE_LIB_INSTALL)
	@echo  " "

clean:
	rm -f *.o libHYPRE*
	rm -rf pchdir tca.map *inslog*
distclean: clean

libHYPRE_dsuperlu.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

libHYPRE_dsuperlu.so: ${OBJS}
	@echo  "Building $@ ... "
	${BUILD_CC_SHARED} -o $@ ${OBJS}

#######################################

#all:  double complex16

double: $(DSLUSRC) $(DPLUSRC) $(ALLAUX) $(LAAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(DSLUSRC) $(DPLUSRC) $(ALLAUX) $(LAAUX)
	$(RANLIB) $(DSUPERLULIB)

complex16: $(ZSLUSRC) $(ZPLUSRC) $(ALLAUX) $(LAAUX)
	$(ARCH) $(ARCHFLAGS) $(DSUPERLULIB) \
		$(ZSLUSRC) $(ZPLUSRC) $(ALLAUX) $(LAAUX)
	$(RANLIB) $(DSUPERLULIB)


##################################
# Do not optimize these routines #
##################################
slamch.o: slamch.c ; $(CC) $(CFLAGS) $(NOOPTS) $(CDEFS) -c $<
dlamch.o: dlamch.c ; $(CC) $(CFLAGS) $(NOOPTS) $(CDEFS) -c $<
##################################

#.c.o:
#	$(CC) $(CFLAGS) $(CDEFS) -DUSE_VENDOR_BLAS -c $< $(VERBOSE)

#.f.o:
#	f77 $(FFLAGS) -c $< $(VERBOSE)

#clean:	
#	rm -f *.o $(DSUPERLULIB)


