
set(HYPRE_superlu_HEADERS
  SRC/Cnames.h
  SRC/colamd.h
  SRC/old_colamd.h
  SRC/slu_cdefs.h
  SRC/slu_Cnames.h
  SRC/slu_dcomplex.h
  SRC/slu_ddefs.h
  SRC/slu_scomplex.h
  SRC/slu_sdefs.h
  SRC/slu_util.h
  SRC/slu_zdefs.h
  SRC/supermatrix.h
)

set(HYPRE_superlu_SRCS 
  SRC/colamd.c 
  SRC/dcolumn_bmod.c 
  SRC/dcolumn_dfs.c 
  SRC/dcopy_to_ucol.c 
  SRC/dgscon.c 
  SRC/dgsequ.c 
  SRC/dgsrfs.c 
  SRC/dgssv.c 
  SRC/dgssvx.c 
  SRC/dgstrf.c 
  SRC/dgstrs.c 
  SRC/dlacon.c 
  SRC/dlangs.c 
  SRC/dlaqgs.c 
  SRC/dmemory.c 
  SRC/dpanel_bmod.c 
  SRC/dpanel_dfs.c 
  SRC/dpivotgrowth.c 
  SRC/dpivotL.c 
  SRC/dpruneL.c 
  SRC/dreadhb.c 
  SRC/dsnode_bmod.c 
  SRC/dsnode_dfs.c 
  SRC/dsp_blas2.c 
  SRC/dsp_blas3.c 
  SRC/dutil.c 
  SRC/get_perm_c.c 
  SRC/heap_relax_snode.c 
  SRC/memory.c 
  SRC/mmd.c 
  SRC/relax_snode.c 
  SRC/sp_coletree.c 
  SRC/sp_ienv.c 
  SRC/sp_preorder.c 
  SRC/superlu_timer.c 
  SRC/slu_util.c
)

add_definitions(-DNO_TIMER)
add_library(HYPRE_superlu OBJECT ${HYPRE_superlu_SRCS})

install (FILES ${HYPRE_superlu_HEADERS} DESTINATION include)
