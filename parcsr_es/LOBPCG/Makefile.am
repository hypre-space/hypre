## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_lobpcg.a
 libHYPRE_lobpcg_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_lobpcg.la
 libHYPRE_lobpcg_la_SOURCES = $(FILES)
 libHYPRE_lobpcg_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = HYPRE_lobpcg.h lobpcg.h

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/krylov

EXTRA_DIST = \
 IJ_eigen_solver.c\
 doc.txt\
 readme.txt
#
FILES = \
 lobpcg.c\
 lobpcg_matrix.c\
 lobpcg_utilities.c\
 HYPRE_lobpcg.h\
 lobpcg.h\
 mmio.h
