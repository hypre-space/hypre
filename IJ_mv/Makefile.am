## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_IJ_mv.la
libHYPRE_IJ_mv_la_SOURCES = $(FILES)
libHYPRE_IJ_mv_la_LDFLAGS = -avoid-version

include_HEADERS = IJ_mv.h HYPRE_IJ_mv.h

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/struct_mv\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls

EXTRA_DIST = \
 IJMatrix_isis.c\
 IJMatrix_petsc.c
#
FILES = \
 F90_HYPRE_IJMatrix.c\
 F90_HYPRE_IJVector.c\
 F90_IJMatrix.c\
 HYPRE_IJMatrix.c\
 HYPRE_IJVector.c\
 IJMatrix.c\
 IJMatrix_parcsr.c\
 IJVector.c\
 IJVector_parcsr.c\
 aux_parcsr_matrix.c\
 HYPRE_IJ_mv.h\
 IJ_matrix.h\
 IJ_mv.h\
 IJ_vector.h\
 aux_parcsr_matrix.h\
 headers.h

