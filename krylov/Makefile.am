## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_krylov.a
 libHYPRE_krylov_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_krylov.la
 libHYPRE_krylov_la_SOURCES = $(FILES)
 libHYPRE_krylov_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = krylov.h
noinst_HEADERS = \
 all_krylov.h\
 bicgstab.h\
 cgnr.h\
 gmres.h\
 pcg.h
dependent_includes = $(include_HEADERS)

AM_CPPFLAGS = \
 -I$(top_builddir)/utilities\
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
 pcg.c

$(SOURCES): $(dependent_includes)
	touch $@
