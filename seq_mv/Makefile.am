## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_seq_mv.la
libHYPRE_seq_mv_la_SOURCES = $(FILES)
libHYPRE_seq_mv_la_LDFLAGS = -avoid-version

include_HEADERS = HYPRE_seq_mv.h seq_mv.h

AM_CPPFLAGS = \
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities

#
FILES = \
 HYPRE_csr_matrix.c\
 HYPRE_mapped_matrix.c\
 HYPRE_multiblock_matrix.c\
 HYPRE_vector.c\
 csr_matop.c\
 csr_matrix.c\
 csr_matvec.c\
 genpart.c\
 mapped_matrix.c\
 multiblock_matrix.c\
 vector.c\
 HYPRE_seq_mv.h\
 csr_matrix.h\
 headers.h\
 mapped_matrix.h\
 multiblock_matrix.h\
 seq_mv.h\
 vector.h

