## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_DistributedMatrixPilutSolver.la
libHYPRE_DistributedMatrixPilutSolver_la_SOURCES = $(FILES)
libHYPRE_DistributedMatrixPilutSolver_la_LDFLAGS = -avoid-version

include_HEADERS = HYPRE_DistributedMatrixPilutSolver_protos.h\
 HYPRE_DistributedMatrixPilutSolver_types.h

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/distributed_matrix

EXTRA_DIST = \
 hypre.c
#
FILES = \
 HYPRE_DistributedMatrixPilutSolver.c\
 comm.c\
 debug.c\
 distributed_qsort.c\
 distributed_qsort_si.c\
 ilut.c\
 parilut.c\
 parutil.c\
 pblas1.c\
 serilut.c\
 trifactor.c\
 util.c\
 DistributedMatrixPilutSolver.h\
 HYPRE_DistributedMatrixPilutSolver_protos.h\
 HYPRE_DistributedMatrixPilutSolver_types.h\
 const.h\
 ilu.h\
 internal_protos.h\
 macros.h\
 struct.h

