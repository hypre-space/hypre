## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_seq_mv.a
 libHYPRE_seq_mv_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_seq_mv.la
 libHYPRE_seq_mv_la_SOURCES = $(FILES)
 libHYPRE_seq_mv_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = HYPRE_seq_mv.h seq_mv.h
noinst_HEADERS = \
 csr_matrix.h\
 headers.h\
 mapped_matrix.h\
 multiblock_matrix.h\
 vector.h
dependent_includes = $(include_HEADERS)

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
 vector.c

$(SOURCES): $(dependent_includes)
	touch $@

