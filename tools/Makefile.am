## Process this file with automake to create Makefile.in

bin_SCRIPTS = V99-09-23.hypre_convert V2000-05-31.hypre_convert
noinst_PROGRAMS = one_to_many \
 one_to_many_vector \
 create_2d_laplacian \
 create_3d_laplacian

one_to_many_SOURCES = one_to_many.c

one_to_many_vector_SOURCES = one_to_many_vector.c

create_2d_laplacian_SOURCES = create_2d_laplacian.c

create_3d_laplacian_SOURCES = create_3d_laplacian.c

AM_CPPFLAGS = -DHYPRE_TIMING\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/struct_mv

LDFLAGS = \
 -L$(srcdir)/../struct_mv\
 -L$(srcdir)/../utilities\
 -lHYPRE_struct_mv\
 -lHYPRE_utilities
