## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_PETScSolverParILUT.a
 libHYPRE_PETScSolverParILUT_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_PETScSolverParILUT.la
 libHYPRE_PETScSolverParILUT_la_SOURCES = $(FILES)
 libHYPRE_PETScSolverParILUT_la_LDFLAGS = -avoid-version
endif !STATICONLY

#include_HEADERS = PETScSolverParILUT.h
noinst_HEADERS = PETScSolverParILUT.h
dependent_includes = $(noinst_HEADERS)

CPPFLAGS = -I..\
 -I$(top_srcdir)\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/PETScMat_ls/pilut

#
FILES = \
 HYPRE_PETScSolverParILUT.c HYPRE_PETScSolverParILUTSetup.c

$(SOURCES): $(dependent_includes)
	touch $@

