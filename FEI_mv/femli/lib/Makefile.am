## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_mli.a
 libHYPRE_mli_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_mli_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_mli.la
 libHYPRE_mli_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_mli_la_OBJECTS
 libHYPRE_mli_la_LDFLAGS = -avoid-version
endif !STATICONLY

noinst_HEADERS = \
 $(top_srcdir)/FEI_mv/femli/util/mli_utils_fortran.f\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method.h\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgcr.h\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgrs.h\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgsa.h\
 $(top_srcdir)/FEI_mv/femli/base/mli.h\
 $(top_srcdir)/FEI_mv/femli/base/mli_defs.h\
 $(top_srcdir)/FEI_mv/femli/base/mli_oneLevel.h\
 $(top_srcdir)/FEI_mv/femli/cintface/cmli.h\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_febase.h\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_fedata.h\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_sfei.h\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_fedata_utils.h\
 $(top_srcdir)/FEI_mv/femli/mapper/mli_mapper.h\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix.h\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix_misc.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_jacobi.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_bjacobi.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_gs.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_sgs.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_bsgs.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_hsgs.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_hschwarz.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_parasails.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_mls.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_chebyshev.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_cg.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_gmres.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_kaczmarz.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_superlu.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_seqsuperlu.h\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_arpacksuperlu.h\
 $(top_srcdir)/FEI_mv/femli/util/mli_utils.h\
 $(top_srcdir)/FEI_mv/femli/vector/mli_vector.h
dependent_includes = $(noinst_HEADERS)

AM_CPPFLAGS = -DMLI_SUPERLU \
 -I$(top_srcdir)\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/FEI_mv/femli\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/parcsr_es/LOBPCG\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/parcsr_ls\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/distributed_matrix\
 -I$(top_srcdir)/distributed_ls\
 -I$(top_srcdir)/FEI_mv/SuperLU

AM_CXXFLAGS = -DMPICH_SKIP_MPICXX

#
FILES = \
 $(top_srcdir)/FEI_mv/femli/util/mli_utils.c\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_calib.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_coarsen1.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_dd_fedata.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_amgsa_dd_sfei.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgcr.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgrs.cxx\
 $(top_srcdir)/FEI_mv/femli/amgs/mli_method_amgsa.cxx\
 $(top_srcdir)/FEI_mv/femli/base/mli.cxx\
 $(top_srcdir)/FEI_mv/femli/base/mli_oneLevel.cxx\
 $(top_srcdir)/FEI_mv/femli/cintface/cmli.cxx\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_fedata.cxx\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_fedata_utils.cxx\
 $(top_srcdir)/FEI_mv/femli/fedata/mli_sfei.cxx\
 $(top_srcdir)/FEI_mv/femli/mapper/mli_mapper.cxx\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix.cxx\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix_mult.cxx\
 $(top_srcdir)/FEI_mv/femli/matrix/mli_matrix_utils.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_jacobi.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_bjacobi.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_gs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_sgs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_bsgs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_hsgs.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_hschwarz.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_parasails.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_mls.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_chebyshev.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_cg.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_gmres.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_kaczmarz.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_superlu.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_seqsuperlu.cxx\
 $(top_srcdir)/FEI_mv/femli/solver/mli_solver_arpacksuperlu.cxx\
 $(top_srcdir)/FEI_mv/femli/vector/mli_vector.cxx

$(OBJECTS): $(dependent_includes)

