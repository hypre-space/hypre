## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_ParaSails.a
 libHYPRE_ParaSails_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_ParaSails.la
 libHYPRE_ParaSails_la_SOURCES = $(FILES)
 libHYPRE_ParaSails_la_LDFLAGS = -avoid-version
endif !STATICONLY

noinst_HEADERS = \
 Common.h\
 ConjGrad.h\
 DiagScale.h\
 Hash.h\
 LoadBal.h\
 Matrix.h\
 Mem.h\
 Numbering.h\
 OrderStat.h\
 ParaSails.h\
 PrunedRows.h\
 RowPatt.h\
 StoredRows.h\
 hypre_ParaSails.h
dependent_includes = $(noinst_HEADERS)

AM_CPPFLAGS = \
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/distributed_matrix

EXTRA_DIST = \
 convert.c\
 driver.c\
 lapdriver.c\
 permute.c
#
FILES = \
 ConjGrad.c\
 DiagScale.c\
 FlexGmres.c\
 Hash.c\
 LoadBal.c\
 Matrix.c\
 Mem.c\
 Numbering.c\
 OrderStat.c\
 ParaSails.c\
 PrunedRows.c\
 RowPatt.c\
 StoredRows.c\
 hypre_ParaSails.c

OBJECTS = $(FILES:.c=.o)

$(OBJECTS): $(dependent_includes)

