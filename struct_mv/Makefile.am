## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_struct_mv.a
 libHYPRE_struct_mv_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_struct_mv.la
 libHYPRE_struct_mv_la_SOURCES = $(FILES)
 libHYPRE_struct_mv_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = struct_mv.h HYPRE_struct_mv.h
noinst_HEADERS = \
 box.h\
 box_neighbors.h\
 box_pthreads.h\
 computation.h\
 headers.h\
 hypre_box_smp_forloop.h\
 struct_communication.h\
 struct_grid.h\
 struct_matrix.h\
 struct_stencil.h\
 struct_vector.h
dependent_includes = $(include_HEADERS) $(noinst_HEADERS)

AM_CPPFLAGS = \
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities

#
FILES = \
 F90_HYPRE_struct_grid.c\
 F90_HYPRE_struct_matrix.c\
 F90_HYPRE_struct_stencil.c\
 F90_HYPRE_struct_vector.c\
 HYPRE_struct_grid.c\
 HYPRE_struct_matrix.c\
 HYPRE_struct_stencil.c\
 HYPRE_struct_vector.c\
 box.c\
 box_algebra.c\
 box_alloc.c\
 box_neighbors.c\
 communication_info.c\
 computation.c\
 grow.c\
 project.c\
 struct_axpy.c\
 struct_communication.c\
 struct_copy.c\
 struct_grid.c\
 struct_innerprod.c\
 struct_io.c\
 struct_matrix.c\
 struct_matrix_mask.c\
 struct_matvec.c\
 struct_scale.c\
 struct_stencil.c\
 struct_vector.c

OBJECTS = $(FILES:.c=.o)

$(OBJECTS): $(dependent_includes)

