## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_sstruct_mv.la
libHYPRE_sstruct_mv_la_SOURCES = $(FILES)
libHYPRE_sstruct_mv_la_LDFLAGS = -avoid-version

include_HEADERS = HYPRE_sstruct_mv.h sstruct_mv.h

AM_CPPFLAGS = \
 -I$(top_srcdir)\
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
 sstruct_vector.c\
 HYPRE_sstruct_mv.h\
 box_map.h\
 headers.h\
 sstruct_graph.h\
 sstruct_grid.h\
 sstruct_matrix.h\
 sstruct_mv.h\
 sstruct_stencil.h\
 sstruct_vector.h

