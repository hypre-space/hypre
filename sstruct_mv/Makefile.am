## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_sstruct_mv.a
 libHYPRE_sstruct_mv_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_sstruct_mv.la
 libHYPRE_sstruct_mv_la_SOURCES = $(FILES)
 libHYPRE_sstruct_mv_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = HYPRE_sstruct_mv.h sstruct_mv.h
noinst_HEADERS = \
 box_map.h\
 headers.h\
 sstruct_graph.h\
 sstruct_grid.h\
 sstruct_matrix.h\
 sstruct_stencil.h\
 sstruct_vector.h
dependent_includes = $(include_HEADERS)

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/struct_mv\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/IJ_mv

#
FILES = \
 HYPRE_sstruct_graph.c\
 HYPRE_sstruct_grid.c\
 HYPRE_sstruct_matrix.c\
 HYPRE_sstruct_stencil.c\
 HYPRE_sstruct_vector.c\
 box_map.c\
 sstruct_axpy.c\
 sstruct_copy.c\
 sstruct_graph.c\
 sstruct_grid.c\
 sstruct_innerprod.c\
 sstruct_matrix.c\
 sstruct_matvec.c\
 sstruct_scale.c\
 sstruct_stencil.c\
 sstruct_vector.c

$(SOURCES): $(dependent_includes)
	touch $@

