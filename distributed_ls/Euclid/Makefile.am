## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_Euclid.a
 libHYPRE_Euclid_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_Euclid_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_Euclid.la
 libHYPRE_Euclid_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_Euclid_la_OBJECTS
 libHYPRE_Euclid_la_LDFLAGS = -avoid-version
endif !STATICONLY

noinst_HEADERS = \
 Euclid_dh.h\
 ExternalRows_dh.h\
 Factor_dh.h\
 Hash_dh.h\
 Hash_i_dh.h\
 MatGenFD.h\
 Mat_dh.h\
 Mem_dh.h\
 Numbering_dh.h\
 Parser_dh.h\
 SortedList_dh.h\
 SortedSet_dh.h\
 SubdomainGraph_dh.h\
 TimeLog_dh.h\
 Timer_dh.h\
 Vec_dh.h\
 blas_dh.h\
 euclid.h\
 euclid_common.h\
 euclid_config.h\
 getRow_dh.h\
 guards_dh.h\
 ilu_dh.h\
 io_dh.h\
 krylov_dh.h\
 macros_dh.h\
 mat_dh_private.h\
 shellSort_dh.h\
 sig_dh.h
dependent_includes = $(noinst_HEADERS)

EUCLID_CFLAGS =  -DMPI_MODE -DMPI_TIMING -DUSING_MPI -DHYPRE_GET_ROW \
                 -DHYPRE_MODE -DOPTIMIZED_DH
AM_CPPFLAGS = ${EUCLID_CFLAGS} \
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/distributed_matrix

EXTRA_DIST = \
 driver.c
#
FILES = \
 Euclid_apply.c\
 Euclid_dh.c\
 ExternalRows_dh.c\
 Factor_dh.c\
 Hash_dh.c\
 Hash_i_dh.c\
 MatGenFD.c\
 Mat_dh.c\
 Mem_dh.c\
 Numbering_dh.c\
 Parser_dh.c\
 SortedList_dh.c\
 SortedSet_dh.c\
 SubdomainGraph_dh.c\
 TimeLog_dh.c\
 Timer_dh.c\
 Vec_dh.c\
 blas_dh.c\
 getRow_dh.c\
 globalObjects.c\
 ilu_mpi_bj.c\
 ilu_mpi_pilu.c\
 ilu_seq.c\
 io_dh.c\
 krylov_dh.c\
 mat_dh_private.c\
 shellSort_dh.c\
 sig_dh.c

$(OBJECTS): $(dependent_includes)

