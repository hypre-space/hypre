## Process this file with automake to create Makefile.in

if STATICONLY
 lib_LIBRARIES = libHYPRE_superlu.a
 libHYPRE_superlu_a_SOURCES = $(FILES)
else !STATICONLY
 lib_LTLIBRARIES = libHYPRE_superlu.la
 libHYPRE_superlu_la_SOURCES = $(FILES)
 libHYPRE_superlu_la_LDFLAGS = -avoid-version
endif !STATICONLY

AM_CPPFLAGS = \
 -I$(top_srcdir)/utilities

#
FILES = \
 colamd.c\
 dcolumn_bmod.c\
 dcolumn_dfs.c\
 dcopy_to_ucol.c\
 dgscon.c\
 dgsequ.c\
 dgsrfs.c\
 dgssv.c\
 dgssvx.c\
 dgstrf.c\
 dgstrs.c\
 dlacon.c\
 dlangs.c\
 dlaqgs.c\
 dmemory.c\
 dpanel_bmod.c\
 dpanel_dfs.c\
 dpivotL.c\
 dpivotgrowth.c\
 dpruneL.c\
 dreadhb.c\
 dsnode_bmod.c\
 dsnode_dfs.c\
 dsp_blas2.c\
 dsp_blas3.c\
 dutil.c\
 get_perm_c.c\
 mmd.c\
 relax_snode.c\
 sp_coletree.c\
 sp_ienv.c\
 sp_preorder.c\
 superlu_memory.c\
 superlu_timer.c\
 superlu_util.c\
 Cnames.h\
 colamd.h\
 dsp_defs.h\
 supermatrix.h\
 superlu_util.h

