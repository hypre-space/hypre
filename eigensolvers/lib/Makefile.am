## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_lobpcg.a
 libHYPRE_lobpcg_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_lobpcg_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_lobpcg.la
 libHYPRE_lobpcg_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_lobpcg_la_OBJECTS
 libHYPRE_lobpcg_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = \
 $(top_srcdir)/eigensolvers/multivector/HYPRE_interpreter.h\
 $(top_srcdir)/eigensolvers/krylov/HYPRE_lobpcg.h\
 $(top_srcdir)/eigensolvers/krylov/lobpcg.h\
 $(top_srcdir)/eigensolvers/multivector/multivector.h\
 $(top_srcdir)/eigensolvers/parcsr_ls/HYPRE_parcsr_int.h\
 $(top_srcdir)/eigensolvers/sstruct_ls/HYPRE_sstruct_int.h\
 $(top_srcdir)/eigensolvers/struct_ls/HYPRE_struct_int.h\
 $(top_srcdir)/eigensolvers/utilities/fortran_matrix.h\
 $(top_srcdir)/utilities/fortran.h

dependent_includes = $(include_HEADERS) 

AM_CPPFLAGS = \
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/blas\
 -I$(top_srcdir)/lapack\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/distributed_matrix\
 -I$(top_srcdir)/distributed_ls/Euclid/include\
 -I$(top_srcdir)/matrix_matrix\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/struct_ls\
 -I$(top_srcdir)/struct_mv\
 -I$(top_srcdir)/sstruct_ls\
 -I$(top_srcdir)/sstruct_mv\
 -I$(top_srcdir)/eigensolvers/krylov\
 -I$(top_srcdir)/eigensolvers/multivector\
 -I$(top_srcdir)/eigensolvers/parcsr_ls\
 -I$(top_srcdir)/eigensolvers/struct_ls\
 -I$(top_srcdir)/eigensolvers/sstruct_ls\
 -I$(top_srcdir)/eigensolvers/utilities

#
FILES = \
 $(top_srcdir)/eigensolvers/krylov/HYPRE_lobpcg.c\
 $(top_srcdir)/eigensolvers/krylov/lobpcg.c\
 $(top_srcdir)/eigensolvers/multivector/multivector.c\
 $(top_srcdir)/eigensolvers/multivector/temp_multivector.c\
 $(top_srcdir)/eigensolvers/parcsr_ls/HYPRE_parcsr_int.c\
 $(top_srcdir)/eigensolvers/sstruct_ls/HYPRE_sstruct_int.c\
 $(top_srcdir)/eigensolvers/struct_ls/HYPRE_struct_int.c\
 $(top_srcdir)/eigensolvers/utilities/fortran_matrix.c

$(OBJECTS): $(dependent_includes)

