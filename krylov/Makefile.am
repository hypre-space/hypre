## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_krylov.la
libHYPRE_krylov_la_SOURCES = $(FILES)
libHYPRE_krylov_la_LDFLAGS = -avoid-version

include_HEADERS = krylov.h

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities

#
FILES = \
 HYPRE_bicgstab.c\
 HYPRE_cgnr.c\
 HYPRE_gmres.c\
 HYPRE_pcg.c\
 bicgstab.c\
 cgnr.c\
 gmres.c\
 pcg.c\
 all_krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 krylov.h\
 pcg.h

