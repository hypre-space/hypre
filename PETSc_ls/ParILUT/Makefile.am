## Process this file with automake to create Makefile.in

AUTOMAKE_OPTIONS = foreign no-dependencies no-installinfo no-installman no-texinfo.tex

lib_LTLIBRARIES = libHYPRE_PETScSolverParILUT.la
libHYPRE_PETScSolverParILUT_la_SOURCES = $(FILES)
libHYPRE_PETScSolverParILUT_la_LDFLAGS = -avoid-version
#include_HEADERS = PETScSolverParILUT.h

CPPFLAGS = -I..\
 -I$(top_srcdir)\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/PETScMat_ls/pilut

#
FILES = \
 HYPRE_PETScSolverParILUT.c HYPRE_PETScSolverParILUTSetup.c\
 PETScSolverParILUT.h
