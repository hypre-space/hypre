## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_struct_ls.la
libHYPRE_struct_ls_la_SOURCES = $(FILES)
libHYPRE_struct_ls_la_LDFLAGS = -avoid-version

include_HEADERS = struct_ls.h HYPRE_struct_ls.h

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/struct_mv

EXTRA_DIST = smg_residual_unrolled.c
#
FILES = \
 F90_HYPRE_struct_hybrid.c\
 F90_HYPRE_struct_pcg.c\
 F90_HYPRE_struct_pfmg.c\
 F90_HYPRE_struct_smg.c\
 HYPRE_struct_bicgstab.c\
 HYPRE_struct_gmres.c\
 HYPRE_struct_hybrid.c\
 HYPRE_struct_jacobi.c\
 HYPRE_struct_pcg.c\
 HYPRE_struct_pfmg.c\
 HYPRE_struct_smg.c\
 HYPRE_struct_sparse_msg.c\
 coarsen.c\
 cyclic_reduction.c\
 general.c\
 hybrid.c\
 jacobi.c\
 pcg_struct.c\
 pfmg.c\
 pfmg2_setup_rap.c\
 pfmg3_setup_rap.c\
 pfmg_relax.c\
 pfmg_setup.c\
 pfmg_setup_interp.c\
 pfmg_setup_rap.c\
 pfmg_setup_rap5.c\
 pfmg_setup_rap7.c\
 pfmg_solve.c\
 point_relax.c\
 semi_interp.c\
 semi_restrict.c\
 semi_setup_rap.c\
 smg.c\
 smg2_setup_rap.c\
 smg3_setup_rap.c\
 smg_axpy.c\
 smg_relax.c\
 smg_residual.c\
 smg_setup.c\
 smg_setup_interp.c\
 smg_setup_rap.c\
 smg_setup_restrict.c\
 smg_solve.c\
 sparse_msg.c\
 sparse_msg2_setup_rap.c\
 sparse_msg3_setup_rap.c\
 sparse_msg_filter.c\
 sparse_msg_interp.c\
 sparse_msg_restrict.c\
 sparse_msg_setup.c\
 sparse_msg_setup_rap.c\
 sparse_msg_solve.c\
 HYPRE_struct_ls.h\
 headers.h\
 pfmg.h\
 smg.h\
 sparse_msg.h\
 struct_ls.h

