## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_blas.a
 libHYPRE_blas_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_blas_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_blas.la
 libHYPRE_blas_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_blas_la_OBJECTS
 libHYPRE_blas_la_LDFLAGS = -avoid-version
endif !STATICONLY

noinst_HEADERS = \
 f2c.h\
 hypre_blas.h
dependent_includes = $(noinst_HEADERS)

#
FILES = \
 dasum.c\
 daxpy.c\
 dcopy.c\
 ddot.c\
 dgemm.c\
 dgemv.c\
 dger.c\
 dnrm2.c\
 drot.c\
 dscal.c\
 dswap.c\
 dsymm.c\
 dsymv.c\
 dsyr2.c\
 dsyr2k.c\
 dsyrk.c\
 dtrmm.c\
 dtrmv.c\
 dtrsm.c\
 dtrsv.c\
 idamax.c\
 blas_utils.c

$(OBJECTS): $(dependent_includes)

