## Process this file with automake to create Makefile.in

lib_LTLIBRARIES = libHYPRE_sstruct_ls.la
libHYPRE_sstruct_ls_la_SOURCES = $(FILES)
libHYPRE_sstruct_ls_la_LDFLAGS = -avoid-version

include_HEADERS = HYPRE_sstruct_ls.h sstruct_ls.h

AM_CPPFLAGS = \
 -I$(top_srcdir)\
 -I$(top_builddir)/utilities\
 -I$(top_srcdir)/utilities\
 -I$(top_srcdir)/krylov\
 -I$(top_srcdir)/struct_mv\
 -I$(top_srcdir)/seq_mv\
 -I$(top_srcdir)/parcsr_mv\
 -I$(top_srcdir)/IJ_mv\
 -I$(top_srcdir)/sstruct_mv\
 -I$(top_srcdir)/struct_ls\
 -I$(top_srcdir)/parcsr_ls

#
FILES = \
 HYPRE_sstruct_bicgstab.c\
 HYPRE_sstruct_gmres.c\
 HYPRE_sstruct_pcg.c\
 HYPRE_sstruct_split.c\
 HYPRE_sstruct_sys_pfmg.c\
 krylov.c\
 krylov_sstruct.c\
 node_relax.c\
 sys_pfmg.c\
 sys_pfmg_relax.c\
 sys_pfmg_setup.c\
 sys_pfmg_setup_interp.c\
 sys_pfmg_setup_rap.c\
 sys_pfmg_solve.c\
 sys_semi_interp.c\
 sys_semi_restrict.c\
 HYPRE_sstruct_ls.h\
 headers.h\
 sstruct_ls.h\
 sys_pfmg.h

