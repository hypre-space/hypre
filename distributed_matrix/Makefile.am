## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_DistributedMatrix.la
libHYPRE_DistributedMatrix_la_SOURCES = $(FILES)
libHYPRE_DistributedMatrix_la_LDFLAGS = -avoid-version

include_HEADERS = distributed_matrix.h \
 HYPRE_distributed_matrix_mv.h\
 HYPRE_distributed_matrix_protos.h \
 HYPRE_distributed_matrix_types.h

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv

#
FILES = \
 HYPRE_distributed_matrix.c\
 distributed_matrix.c\
 distributed_matrix_ISIS.c\
 distributed_matrix_PETSc.c\
 distributed_matrix_parcsr.c\
 HYPRE_distributed_matrix_mv.h\
 HYPRE_distributed_matrix_protos.h\
 HYPRE_distributed_matrix_types.h\
 distributed_matrix.h\
 internal_protos.h

