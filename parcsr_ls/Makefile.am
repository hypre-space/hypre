## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_parcsr_ls.a
 libHYPRE_parcsr_ls_a_SOURCES = $(FILES)
 OBJECTS = libHYPRE_parcsr_ls_a_OBJECTS
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_parcsr_ls.la
 libHYPRE_parcsr_ls_la_SOURCES = $(FILES)
 OBJECTS = libHYPRE_parcsr_ls_la_OBJECTS
 libHYPRE_parcsr_ls_la_LDFLAGS = -avoid-version
endif !STATICONLY

include_HEADERS = parcsr_ls.h HYPRE_parcsr_ls.h
noinst_HEADERS = \
 Common.h\
 headers.h\
 par_amg.h\
 schwarz.h
dependent_includes = $(include_HEADERS) $(noinst_HEADERS)

AM_CPPFLAGS = \
 -DHYPRE_MODE -DOPTIMIZED_DH\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/distributed_matrix\
 -I$(top_srcdir)/distributed_ls/Euclid/include\
 -I$(top_srcdir)/matrix_matrix\
 -I$(top_srcdir)/IJ_mv

EXTRA_DIST = \
 driver.c
#
FILES = \
 F90_HYPRE_parcsr_ParaSails.c\
 F90_HYPRE_parcsr_amg.c\
 F90_HYPRE_parcsr_bicgstab.c\
 F90_HYPRE_parcsr_cgnr.c\
 F90_HYPRE_parcsr_gmres.c\
 F90_HYPRE_parcsr_pcg.c\
 F90_HYPRE_parcsr_pilut.c\
 F90_hypre_laplace.c\
 HYPRE_parcsr_Euclid.c\
 HYPRE_parcsr_ParaSails.c\
 HYPRE_parcsr_amg.c\
 HYPRE_parcsr_bicgstab.c\
 HYPRE_parcsr_cgnr.c\
 HYPRE_parcsr_gmres.c\
 HYPRE_parcsr_hybrid.c\
 HYPRE_parcsr_pcg.c\
 HYPRE_parcsr_pilut.c\
 HYPRE_parcsr_schwarz.c\
 amg_hybrid.c\
 par_amg.c\
 par_amg_setup.c\
 par_amg_solve.c\
 par_amg_solveT.c\
 par_cg_relax_wt.c\
 par_coarse_parms.c\
 par_coarsen.c\
 par_cycle.c\
 par_difconv.c\
 par_gsmg.c\
 par_indepset.c\
 par_interp.c\
 par_laplace.c\
 par_laplace_27pt.c\
 par_laplace_9pt.c\
 par_rap.c\
 par_rap_communication.c\
 par_relax.c\
 par_relax_interface.c\
 par_scaled_matnorm.c\
 par_schwarz.c\
 par_stats.c\
 par_strength.c\
 pcg_par.c\
 schwarz.c

$(OBJECTS): $(dependent_includes)

