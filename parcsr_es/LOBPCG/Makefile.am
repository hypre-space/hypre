## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_lobpcg.la
libHYPRE_lobpcg_la_SOURCES = $(FILES)
libHYPRE_lobpcg_la_LDFLAGS = -avoid-version

include_HEADERS = HYPRE_lobpcg.h lobpcg.h

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/utilities

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
