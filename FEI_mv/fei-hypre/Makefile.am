## Process this file with automake to create Makefile.in

if USE_FEI_BASE_DIR_DEFAULT
  BASE_DIR = $(top_srcdir)/FEI_mv/fei-base
else !USE_FEI_BASE_DIR_DEFAULT
  BASE_DIR = @FEI_BASE_DIR@
endif !USE_FEI_BASE_DIR_DEFAULT

if STATICONLY
 lib_LIBRARIES = libHYPRE_FEI.a
 libHYPRE_FEI_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_FEI_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_FEI.la
 libHYPRE_FEI_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_FEI_la_OBJECTS
 libHYPRE_FEI_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = cfei-hypre.h cfei_hypre.h \
 HYPRE_LinSysCore.h HYPRE_FEI_includes.h \
 $(BASE_DIR)/Lookup.h  \
 $(BASE_DIR)/LinearSystemCore.h \
 $(BASE_DIR)/fei_defs.h\
 snl_LinearSystemCore.h
noinst_HEADERS = \
 FEI_HYPRE_Impl.h\
 FEI_HYPRE_include.h\
 HYPRE_Builder.h\
 HYPRE_FEI_mv.h\
 HYPRE_LSI_Dsuperlu.h\
 HYPRE_LSI_UZAWA.h\
 HYPRE_LSI_Uzawa_c.h\
 HYPRE_LSI_blkprec.h\
 HYPRE_LSI_block.h\
 HYPRE_LSI_ddict.h\
 HYPRE_LSI_ddilut.h\
 HYPRE_LSI_mli.h\
 HYPRE_LSI_poly.h\
 HYPRE_LSI_schur.h\
 HYPRE_LSI_schwarz.h\
 HYPRE_MHMatrix.h\
 HYPRE_SlideReduction.h\
 HYPRE_parcsr_TFQmr.h\
 HYPRE_parcsr_bicgs.h\
 HYPRE_parcsr_bicgstabl.h\
 HYPRE_parcsr_fgmres.h\
 HYPRE_parcsr_lsicg.h\
 HYPRE_parcsr_symqmr.h
dependent_includes = $(include_HEADERS) $(noinst_HEADERS)

AM_CPPFLAGS = -DHAVE_SUPERLU -DBOOL_NOT_SUPPORTED\
 -I$(top_srcdir)\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/distributed_matrix\
 -I$(top_srcdir)/distributed_ls\
 -I$(top_srcdir)/FEI_mv/femli\
 -I$(BASE_DIR)\
 -I$(top_srcdir)/FEI_mv/SuperLU

AM_CXXFLAGS = -DMPICH_SKIP_MPICXX

EXTRA_DIST = \
 driver.cxx
EXTRA_PROGRAMS = driver
driver_SOURCES = driver.cxx
driver_LDADD = \
 -L.\
 -L$(libdir)\
 $(SUPERLU_LIB) $(MLPACK_LIB)\
 -lHYPRE_FEI\
 -lHYPRE_FEI_BASE\
 -lHYPRE_parcsr_ls \
 -lHYPRE_ParaSails \
 -lHYPRE_FEI_isismv \
 -lHYPRE_IJ_mv \
 -lml \
 -lHYPRE_utilities \
 -lHYPRE_parcsr_mv \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_superlu \
 -lHYPRE_seq_mv \
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)

#
FILES = \
 FEI_HYPRE_Impl.cxx\
 HYPRE_LSC_aux.cxx\
 HYPRE_LSI_Dsuperlu.c\
 HYPRE_LSI_UZAWA.cxx\
 HYPRE_LSI_blkprec.cxx\
 HYPRE_LSI_ddict.c\
 HYPRE_LSI_ddilut.c\
 HYPRE_LSI_ml.c\
 HYPRE_LSI_mli.cxx\
 HYPRE_LSI_poly.c\
 HYPRE_LSI_schwarz.c\
 HYPRE_LinSysCore.cxx\
 HYPRE_SlideReduction.cxx\
 HYPRE_parcsr_TFQmr.c\
 HYPRE_parcsr_bicgs.c\
 HYPRE_parcsr_bicgstabl.c\
 HYPRE_parcsr_fgmres.c\
 HYPRE_parcsr_lsicg.c\
 HYPRE_parcsr_symqmr.c\
 SymQMR.c\
 TFQmr.c\
 bicgs.c\
 bicgstabl.c\
 cfei_hypre.cxx\
 fgmres.c\
 hypre_lsi_amge.c\
 hypre_lsi_ddamg.c\
 hypre_lsi_misc.c\
 hypre_schur_reduce.cxx\
 hypre_slide_reduce.cxx\
 lsicg.c

snl_LinearSystemCore.h:
	test -f $(BASE_DIR)/LinearSystemCore.h && $(INSTALL_DATA) $(BASE_DIR)/LinearSystemCore.h snl_LinearSystemCore.h

$(OBJECTS): $(dependent_includes)

