## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_IJ_mv.a
 libHYPRE_IJ_mv_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_IJ_mv.la
 libHYPRE_IJ_mv_la_SOURCES = $(FILES)
 libHYPRE_IJ_mv_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = IJ_mv.h HYPRE_IJ_mv.h
noinst_HEADERS = \
 IJ_matrix.h\
 IJ_vector.h\
 aux_parcsr_matrix.h\
 headers.h
dependent_includes = $(include_HEADERS) $(noinst_HEADERS)

AM_CPPFLAGS = \
 -I$(top_builddir)/utilities\
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
 aux_parcsr_matrix.c

OBJECTS = $(FILES:.c=.o)

$(OBJECTS): $(dependent_includes)

