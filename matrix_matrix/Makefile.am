## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_MatrixMatrix.la
libHYPRE_MatrixMatrix_la_SOURCES = $(FILES)
libHYPRE_MatrixMatrix_la_LDFLAGS = -avoid-version

include_HEADERS = HYPRE_matrix_matrix_protos.h

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/distributed_matrix\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv

EXTRA_DIST = HYPRE_BuildIJMatrixFromDistributedMatrix.c internal_protos.h
#
FILES = \
 HYPRE_ConvertPETScMatrixToDistributedMatrix.c\
 HYPRE_ConvertParCSRMatrixToDistributedMatrix.c\
 HYPRE_matrix_matrix_protos.h

