## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_MatrixMatrix.a
 libHYPRE_MatrixMatrix_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_MatrixMatrix_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_MatrixMatrix.la
 libHYPRE_MatrixMatrix_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_MatrixMatrix_la_OBJECTS
 libHYPRE_MatrixMatrix_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = HYPRE_matrix_matrix_protos.h
dependent_includes = $(include_HEADERS)

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/distributed_matrix\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv

EXTRA_DIST = HYPRE_BuildIJMatrixFromDistributedMatrix.c internal_protos.h
#
FILES = \
 HYPRE_ConvertPETScMatrixToDistributedMatrix.c\
 HYPRE_ConvertParCSRMatrixToDistributedMatrix.c

$(OBJECTS): $(dependent_includes)

