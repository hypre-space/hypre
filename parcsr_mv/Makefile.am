## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_parcsr_mv.la
libHYPRE_parcsr_mv_la_SOURCES = $(FILES)
libHYPRE_parcsr_mv_la_LDFLAGS = -avoid-version

include_HEADERS = parcsr_mv.h HYPRE_parcsr_mv.h

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/seq_mv

EXTRA_DIST = \
 parchord_to_parcsr.c\
 par_chord_matrix.h\
 driver.c\
 driver_aat.c\
 driver_boolaat.c\
 driver_boolmatmul.c\
 driver_matmul.c\
 driver_matvec.c\
 headers
#
FILES = \
 F90_HYPRE_parcsr_matrix.c\
 F90_HYPRE_parcsr_vector.c\
 F90_par_vector.c\
 F90_parcsr_matrix.c\
 HYPRE_parcsr_matrix.c\
 HYPRE_parcsr_vector.c\
 communicationT.c\
 numbers.c\
 par_csr_aat.c\
 par_csr_bool_matop.c\
 par_csr_bool_matrix.c\
 par_csr_communication.c\
 par_csr_matop.c\
 par_csr_matrix.c\
 par_csr_matvec.c\
 par_vector.c\
 HYPRE_parcsr_mv.h\
 headers.h\
 numbers.h\
 par_csr_communication.h\
 par_csr_matrix.h\
 par_vector.h\
 parcsr_mv.h

