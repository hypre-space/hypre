/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/

#include <HYPRE_config.h>

#include "HYPRE_sstruct_ls.h"

#ifndef hypre_SSTRUCT_LS_HEADER
#define hypre_SSTRUCT_LS_HEADER

#include "_hypre_utilities.h"
#include "krylov.h"
#include "_hypre_struct_ls.h"
#include "_hypre_sstruct_mv.h"
#include "_hypre_parcsr_ls.h"
#include "multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructOwnInfo data structure
 * This structure is for the coarsen fboxes that are on this processor,
 * and the cboxes of cgrid/(all coarsened fboxes) on this processor (i.e.,
 * the coarse boxes of the composite cgrid (no underlying) on this processor).
 *--------------------------------------------------------------------------*/
#ifndef hypre_OWNINFODATA_HEADER
#define hypre_OWNINFODATA_HEADER


typedef struct 
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *own_boxes;    /* size of fgrid */
   HYPRE_Int           **own_cboxnums; /* local cbox number- each fbox
                                          leads to an array of cboxes */

   hypre_BoxArrayArray  *own_composite_cboxes;  /* size of cgrid */
   HYPRE_Int             own_composite_size;
} hypre_SStructOwnInfoData;


/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructOwnInfoData;
 *--------------------------------------------------------------------------*/

#define hypre_SStructOwnInfoDataSize(own_data)       ((own_data) -> size)
#define hypre_SStructOwnInfoDataOwnBoxes(own_data)   ((own_data) -> own_boxes)
#define hypre_SStructOwnInfoDataOwnBoxNums(own_data) \
((own_data) -> own_cboxnums)
#define hypre_SStructOwnInfoDataCompositeCBoxes(own_data) \
((own_data) -> own_composite_cboxes)
#define hypre_SStructOwnInfoDataCompositeSize(own_data) \
((own_data) -> own_composite_size)

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_RECVINFODATA_HEADER
#define hypre_RECVINFODATA_HEADER


typedef struct 
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *recv_boxes;
   HYPRE_Int           **recv_procs;

} hypre_SStructRecvInfoData;

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct 
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *send_boxes;
   HYPRE_Int           **send_procs;
   HYPRE_Int           **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




typedef struct
{
    hypre_IJMatrix    *Face_iedge;
    hypre_IJMatrix    *Element_iedge;
    hypre_IJMatrix    *Edge_iedge;
                                                                                                                            
    hypre_IJMatrix    *Element_Face;
    hypre_IJMatrix    *Element_Edge;
                                                                                                                            
} hypre_PTopology;

/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Header info for the Maxwell solver
 *
 *****************************************************************************/

#ifndef hypre_MAXWELL_HEADER
#define hypre_MAXWELL_HEADER

/*--------------------------------------------------------------------------
 * hypre_MaxwellData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
                      
   double                  tol;
   HYPRE_Int               max_iter;
   HYPRE_Int               rel_change;
   HYPRE_Int               zero_guess;
   HYPRE_Int               ndim;
                      
   HYPRE_Int               num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int               num_post_relax; /* number of post relaxation sweeps */

   HYPRE_Int               constant_coef;
   
   hypre_Index            *rfactor;

   hypre_SStructGrid     **egrid_l;
                    
   HYPRE_IJMatrix          Aen;
   hypre_ParCSRMatrix    **Aen_l;

   /* these will be extracted from the amg_data structure. Note that there is no grid
      underlying these matrices and vectors if they are generated by the amg_setup.
      So, will be stored as Parcsr_matrices and Par_vectors. */
   hypre_SStructMatrix    *Ann;
   hypre_SStructVector    *bn;
   hypre_SStructVector    *xn;

   void                   *amg_vdata;
   hypre_ParCSRMatrix    **Ann_l;
   hypre_SStructStencil  **Ann_stencils;
   hypre_ParCSRMatrix    **Pn_l;
   hypre_ParCSRMatrix    **RnT_l;
   hypre_ParVector       **bn_l;
   hypre_ParVector       **xn_l;
   hypre_ParVector       **resn_l;
   hypre_ParVector       **en_l;
   hypre_ParVector       **nVtemp_l;
   hypre_ParVector       **nVtemp2_l;
   HYPRE_Int             **nCF_marker_l;
   double                 *nrelax_weight;
   double                 *nomega;
   HYPRE_Int               nrelax_type;
   HYPRE_Int               node_numlevels;

   hypre_ParCSRMatrix     *Tgrad;
   hypre_ParCSRMatrix     *T_transpose;

   /* edge data structure. These will have grids. */
   HYPRE_Int               edge_maxlevels;  
   HYPRE_Int               edge_numlevels;  
   hypre_ParCSRMatrix    **Aee_l;
   hypre_ParVector       **be_l;
   hypre_ParVector       **xe_l;
   hypre_ParVector       **rese_l;
   hypre_ParVector       **ee_l;
   hypre_ParVector       **eVtemp_l;
   hypre_ParVector       **eVtemp2_l;
   HYPRE_Int             **eCF_marker_l;
   double                 *erelax_weight;
   double                 *eomega;
   HYPRE_Int               erelax_type;

   /* edge data structure. These will have no grid. */
   hypre_IJMatrix        **Pe_l;
   hypre_IJMatrix        **ReT_l;
   HYPRE_Int             **BdryRanks_l;
   HYPRE_Int              *BdryRanksCnts_l;

   /* edge-node data structure. These will have grids. */
   HYPRE_Int               en_numlevels;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int               print_level;
   HYPRE_Int               logging;
   double                 *norms;
   double                 *rel_norms;

} hypre_MaxwellData;

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




typedef struct
{
   HYPRE_Int row;
                                                                                                                                    
   HYPRE_Int ncols;
   HYPRE_Int      *cols;
   double   *data;
                                                                                                                                    
} hypre_MaxwellOffProcRow;


/* eliminate_rowscols.c */
HYPRE_Int hypre_ParCSRMatrixEliminateRowsCols ( hypre_ParCSRMatrix *A , HYPRE_Int nrows_to_eliminate , HYPRE_Int *rows_to_eliminate );
HYPRE_Int hypre_CSRMatrixEliminateRowsColsDiag ( hypre_ParCSRMatrix *A , HYPRE_Int nrows_to_eliminate , HYPRE_Int *rows_to_eliminate );
HYPRE_Int hypre_CSRMatrixEliminateRowsOffd ( hypre_ParCSRMatrix *A , HYPRE_Int nrows_to_eliminate , HYPRE_Int *rows_to_eliminate );
HYPRE_Int hypre_CSRMatrixEliminateColsOffd ( hypre_CSRMatrix *Aoffd , HYPRE_Int ncols_to_eliminate , HYPRE_Int *cols_to_eliminate );

/* fac_amr_fcoarsen.c */
HYPRE_Int hypre_AMR_FCoarsen ( hypre_SStructMatrix *A , hypre_SStructMatrix *fac_A , hypre_SStructPMatrix *A_crse , hypre_Index refine_factors , HYPRE_Int level );

/* fac_amr_rap.c */
HYPRE_Int hypre_AMR_RAP ( hypre_SStructMatrix *A , hypre_Index *rfactors , hypre_SStructMatrix **fac_A_ptr );

/* fac_amr_zero_data.c */
HYPRE_Int hypre_ZeroAMRVectorData ( hypre_SStructVector *b , HYPRE_Int *plevels , hypre_Index *rfactors );
HYPRE_Int hypre_ZeroAMRMatrixData ( hypre_SStructMatrix *A , HYPRE_Int part_crse , hypre_Index rfactors );

/* fac.c */
void *hypre_FACCreate ( MPI_Comm comm );
HYPRE_Int hypre_FACDestroy2 ( void *fac_vdata );
HYPRE_Int hypre_FACSetTol ( void *fac_vdata , double tol );
HYPRE_Int hypre_FACSetPLevels ( void *fac_vdata , HYPRE_Int nparts , HYPRE_Int *plevels );
HYPRE_Int hypre_FACSetPRefinements ( void *fac_vdata , HYPRE_Int nparts , HYPRE_Int (*prefinements )[3 ]);
HYPRE_Int hypre_FACSetMaxLevels ( void *fac_vdata , HYPRE_Int nparts );
HYPRE_Int hypre_FACSetMaxIter ( void *fac_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_FACSetRelChange ( void *fac_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_FACSetZeroGuess ( void *fac_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_FACSetRelaxType ( void *fac_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_FACSetJacobiWeight ( void *fac_vdata , double weight );
HYPRE_Int hypre_FACSetNumPreSmooth ( void *fac_vdata , HYPRE_Int num_pre_smooth );
HYPRE_Int hypre_FACSetNumPostSmooth ( void *fac_vdata , HYPRE_Int num_post_smooth );
HYPRE_Int hypre_FACSetCoarseSolverType ( void *fac_vdata , HYPRE_Int csolver_type );
HYPRE_Int hypre_FACSetLogging ( void *fac_vdata , HYPRE_Int logging );
HYPRE_Int hypre_FACGetNumIterations ( void *fac_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_FACPrintLogging ( void *fac_vdata , HYPRE_Int myid );
HYPRE_Int hypre_FACGetFinalRelativeResidualNorm ( void *fac_vdata , double *relative_residual_norm );

/* fac_cf_coarsen.c */
HYPRE_Int hypre_AMR_CFCoarsen ( hypre_SStructMatrix *A , hypre_SStructMatrix *fac_A , hypre_Index refine_factors , HYPRE_Int level );

/* fac_CFInterfaceExtents.c */
hypre_BoxArray *hypre_CFInterfaceExtents ( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_StructStencil *stencils , hypre_Index rfactors );
HYPRE_Int hypre_CFInterfaceExtents2 ( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_StructStencil *stencils , hypre_Index rfactors , hypre_BoxArray *cf_interface );

/* fac_cfstencil_box.c */
hypre_Box *hypre_CF_StenBox ( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_Index stencil_shape , hypre_Index rfactors , HYPRE_Int ndim );

/* fac_interp2.c */
HYPRE_Int hypre_FacSemiInterpCreate2 ( void **fac_interp_vdata_ptr );
HYPRE_Int hypre_FacSemiInterpDestroy2 ( void *fac_interp_vdata );
HYPRE_Int hypre_FacSemiInterpSetup2 ( void *fac_interp_vdata , hypre_SStructVector *e , hypre_SStructPVector *ec , hypre_Index rfactors );
HYPRE_Int hypre_FAC_IdentityInterp2 ( void *fac_interp_vdata , hypre_SStructPVector *xc , hypre_SStructVector *e );
HYPRE_Int hypre_FAC_WeightedInterp2 ( void *fac_interp_vdata , hypre_SStructPVector *xc , hypre_SStructVector *e_parts );

/* fac_relax.c */
HYPRE_Int hypre_FacLocalRelax ( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *x , hypre_SStructPVector *b , HYPRE_Int num_relax , HYPRE_Int *zero_guess );

/* fac_restrict2.c */
HYPRE_Int hypre_FacSemiRestrictCreate2 ( void **fac_restrict_vdata_ptr );
HYPRE_Int hypre_FacSemiRestrictSetup2 ( void *fac_restrict_vdata , hypre_SStructVector *r , HYPRE_Int part_crse , HYPRE_Int part_fine , hypre_SStructPVector *rc , hypre_Index rfactors );
HYPRE_Int hypre_FACRestrict2 ( void *fac_restrict_vdata , hypre_SStructVector *xf , hypre_SStructPVector *xc );
HYPRE_Int hypre_FacSemiRestrictDestroy2 ( void *fac_restrict_vdata );

/* fac_setup2.c */
HYPRE_Int hypre_FacSetup2 ( void *fac_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b , hypre_SStructVector *x );

/* fac_solve3.c */
HYPRE_Int hypre_FACSolve3 ( void *fac_vdata , hypre_SStructMatrix *A_user , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* fac_zero_cdata.c */
HYPRE_Int hypre_FacZeroCData ( void *fac_vdata , hypre_SStructMatrix *A );

/* fac_zero_stencilcoef.c */
HYPRE_Int hypre_FacZeroCFSten ( hypre_SStructPMatrix *Af , hypre_SStructPMatrix *Ac , hypre_SStructGrid *grid , HYPRE_Int fine_part , hypre_Index rfactors );
HYPRE_Int hypre_FacZeroFCSten ( hypre_SStructPMatrix *A , hypre_SStructGrid *grid , HYPRE_Int fine_part );

/* hypre_bsearch.c */
HYPRE_Int hypre_LowerBinarySearch ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int list_length );
HYPRE_Int hypre_UpperBinarySearch ( HYPRE_Int *list , HYPRE_Int value , HYPRE_Int list_length );

/* hypre_MaxwellSolve2.c */
HYPRE_Int hypre_MaxwellSolve2 ( void *maxwell_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *f , hypre_SStructVector *u );

/* hypre_MaxwellSolve.c */
HYPRE_Int hypre_MaxwellSolve ( void *maxwell_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *f , hypre_SStructVector *u );

/* HYPRE_sstruct_bicgstab.c */
HYPRE_Int HYPRE_SStructBiCGSTABCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructBiCGSTABDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructBiCGSTABSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructBiCGSTABSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructBiCGSTABSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructBiCGSTABSetAbsoluteTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructBiCGSTABSetMinIter ( HYPRE_SStructSolver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_SStructBiCGSTABSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructBiCGSTABSetStopCrit ( HYPRE_SStructSolver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_SStructBiCGSTABSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
HYPRE_Int HYPRE_SStructBiCGSTABSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructBiCGSTABSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_SStructBiCGSTABGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
HYPRE_Int HYPRE_SStructBiCGSTABGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_flexgmres.c */
HYPRE_Int HYPRE_SStructFlexGMRESCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructFlexGMRESDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructFlexGMRESSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructFlexGMRESSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructFlexGMRESSetKDim ( HYPRE_SStructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_SStructFlexGMRESSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructFlexGMRESSetAbsoluteTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructFlexGMRESSetMinIter ( HYPRE_SStructSolver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_SStructFlexGMRESSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructFlexGMRESSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
HYPRE_Int HYPRE_SStructFlexGMRESSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructFlexGMRESSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int level );
HYPRE_Int HYPRE_SStructFlexGMRESGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
HYPRE_Int HYPRE_SStructFlexGMRESGetResidual ( HYPRE_SStructSolver solver , void **residual );
HYPRE_Int HYPRE_SStructFlexGMRESSetModifyPC ( HYPRE_SStructSolver solver , HYPRE_PtrToModifyPCFcn modify_pc );

/* HYPRE_sstruct_gmres.c */
HYPRE_Int HYPRE_SStructGMRESCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructGMRESDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructGMRESSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructGMRESSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructGMRESSetKDim ( HYPRE_SStructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_SStructGMRESSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructGMRESSetAbsoluteTol ( HYPRE_SStructSolver solver , double atol );
HYPRE_Int HYPRE_SStructGMRESSetMinIter ( HYPRE_SStructSolver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_SStructGMRESSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructGMRESSetStopCrit ( HYPRE_SStructSolver solver , HYPRE_Int stop_crit );
HYPRE_Int HYPRE_SStructGMRESSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
HYPRE_Int HYPRE_SStructGMRESSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructGMRESSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int level );
HYPRE_Int HYPRE_SStructGMRESGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructGMRESGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
HYPRE_Int HYPRE_SStructGMRESGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_int.c */
HYPRE_Int hypre_SStructPVectorSetRandomValues ( hypre_SStructPVector *pvector , HYPRE_Int seed );
HYPRE_Int hypre_SStructVectorSetRandomValues ( hypre_SStructVector *vector , HYPRE_Int seed );
HYPRE_Int hypre_SStructSetRandomValues ( void *v , HYPRE_Int seed );
HYPRE_Int HYPRE_SStructSetupInterpreter ( mv_InterfaceInterpreter *i );
HYPRE_Int HYPRE_SStructSetupMatvec ( HYPRE_MatvecFunctions *mv );

/* HYPRE_sstruct_InterFAC.c */
HYPRE_Int HYPRE_SStructFACCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructFACDestroy2 ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructFACAMR_RAP ( HYPRE_SStructMatrix A , HYPRE_Int (*rfactors )[3 ], HYPRE_SStructMatrix *fac_A );
HYPRE_Int HYPRE_SStructFACSetup2 ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructFACSolve3 ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructFACSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructFACSetPLevels ( HYPRE_SStructSolver solver , HYPRE_Int nparts , HYPRE_Int *plevels );
HYPRE_Int HYPRE_SStructFACZeroCFSten ( HYPRE_SStructMatrix A , HYPRE_SStructGrid grid , HYPRE_Int part , HYPRE_Int rfactors [3 ]);
HYPRE_Int HYPRE_SStructFACZeroFCSten ( HYPRE_SStructMatrix A , HYPRE_SStructGrid grid , HYPRE_Int part );
HYPRE_Int HYPRE_SStructFACZeroAMRMatrixData ( HYPRE_SStructMatrix A , HYPRE_Int part_crse , HYPRE_Int rfactors [3 ]);
HYPRE_Int HYPRE_SStructFACZeroAMRVectorData ( HYPRE_SStructVector b , HYPRE_Int *plevels , HYPRE_Int (*rfactors )[3 ]);
HYPRE_Int HYPRE_SStructFACSetPRefinements ( HYPRE_SStructSolver solver , HYPRE_Int nparts , HYPRE_Int (*rfactors )[3 ]);
HYPRE_Int HYPRE_SStructFACSetMaxLevels ( HYPRE_SStructSolver solver , HYPRE_Int max_levels );
HYPRE_Int HYPRE_SStructFACSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructFACSetRelChange ( HYPRE_SStructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_SStructFACSetZeroGuess ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructFACSetNonZeroGuess ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructFACSetRelaxType ( HYPRE_SStructSolver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_SStructFACSetJacobiWeight ( HYPRE_SStructSolver solver , double weight );
HYPRE_Int HYPRE_SStructFACSetNumPreRelax ( HYPRE_SStructSolver solver , HYPRE_Int num_pre_relax );
HYPRE_Int HYPRE_SStructFACSetNumPostRelax ( HYPRE_SStructSolver solver , HYPRE_Int num_post_relax );
HYPRE_Int HYPRE_SStructFACSetCoarseSolverType ( HYPRE_SStructSolver solver , HYPRE_Int csolver_type );
HYPRE_Int HYPRE_SStructFACSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructFACGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructFACGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_lgmres.c */
HYPRE_Int HYPRE_SStructLGMRESCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructLGMRESDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructLGMRESSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructLGMRESSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructLGMRESSetKDim ( HYPRE_SStructSolver solver , HYPRE_Int k_dim );
HYPRE_Int HYPRE_SStructLGMRESSetAugDim ( HYPRE_SStructSolver solver , HYPRE_Int aug_dim );
HYPRE_Int HYPRE_SStructLGMRESSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructLGMRESSetAbsoluteTol ( HYPRE_SStructSolver solver , double atol );
HYPRE_Int HYPRE_SStructLGMRESSetMinIter ( HYPRE_SStructSolver solver , HYPRE_Int min_iter );
HYPRE_Int HYPRE_SStructLGMRESSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructLGMRESSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
HYPRE_Int HYPRE_SStructLGMRESSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructLGMRESSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int level );
HYPRE_Int HYPRE_SStructLGMRESGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructLGMRESGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
HYPRE_Int HYPRE_SStructLGMRESGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_maxwell.c */
HYPRE_Int HYPRE_SStructMaxwellCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructMaxwellDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructMaxwellSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructMaxwellSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructMaxwellSolve2 ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_MaxwellGrad ( HYPRE_SStructGrid grid , HYPRE_ParCSRMatrix *T );
HYPRE_Int HYPRE_SStructMaxwellSetGrad ( HYPRE_SStructSolver solver , HYPRE_ParCSRMatrix T );
HYPRE_Int HYPRE_SStructMaxwellSetRfactors ( HYPRE_SStructSolver solver , HYPRE_Int rfactors [3 ]);
HYPRE_Int HYPRE_SStructMaxwellSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructMaxwellSetConstantCoef ( HYPRE_SStructSolver solver , HYPRE_Int constant_coef );
HYPRE_Int HYPRE_SStructMaxwellSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructMaxwellSetRelChange ( HYPRE_SStructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_SStructMaxwellSetNumPreRelax ( HYPRE_SStructSolver solver , HYPRE_Int num_pre_relax );
HYPRE_Int HYPRE_SStructMaxwellSetNumPostRelax ( HYPRE_SStructSolver solver , HYPRE_Int num_post_relax );
HYPRE_Int HYPRE_SStructMaxwellSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructMaxwellSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_SStructMaxwellPrintLogging ( HYPRE_SStructSolver solver , HYPRE_Int myid );
HYPRE_Int HYPRE_SStructMaxwellGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructMaxwellGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
HYPRE_Int HYPRE_SStructMaxwellPhysBdy ( HYPRE_SStructGrid *grid_l , HYPRE_Int num_levels , HYPRE_Int rfactors [3 ], HYPRE_Int ***BdryRanks_ptr , HYPRE_Int **BdryRanksCnt_ptr );
HYPRE_Int HYPRE_SStructMaxwellEliminateRowsCols ( HYPRE_ParCSRMatrix parA , HYPRE_Int nrows , HYPRE_Int *rows );
HYPRE_Int HYPRE_SStructMaxwellZeroVector ( HYPRE_ParVector v , HYPRE_Int *rows , HYPRE_Int nrows );

/* HYPRE_sstruct_pcg.c */
HYPRE_Int HYPRE_SStructPCGCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructPCGDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructPCGSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructPCGSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructPCGSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructPCGSetAbsoluteTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructPCGSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructPCGSetTwoNorm ( HYPRE_SStructSolver solver , HYPRE_Int two_norm );
HYPRE_Int HYPRE_SStructPCGSetRelChange ( HYPRE_SStructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_SStructPCGSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
HYPRE_Int HYPRE_SStructPCGSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructPCGSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int level );
HYPRE_Int HYPRE_SStructPCGGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructPCGGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
HYPRE_Int HYPRE_SStructPCGGetResidual ( HYPRE_SStructSolver solver , void **residual );
HYPRE_Int HYPRE_SStructDiagScaleSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructDiagScale ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );

/* HYPRE_sstruct_split.c */
HYPRE_Int HYPRE_SStructSplitCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver_ptr );
HYPRE_Int HYPRE_SStructSplitDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructSplitSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructSplitSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructSplitSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructSplitSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructSplitSetZeroGuess ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructSplitSetNonZeroGuess ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructSplitSetStructSolver ( HYPRE_SStructSolver solver , HYPRE_Int ssolver );
HYPRE_Int HYPRE_SStructSplitGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructSplitGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_sys_pfmg.c */
HYPRE_Int HYPRE_SStructSysPFMGCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
HYPRE_Int HYPRE_SStructSysPFMGDestroy ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructSysPFMGSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructSysPFMGSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
HYPRE_Int HYPRE_SStructSysPFMGSetTol ( HYPRE_SStructSolver solver , double tol );
HYPRE_Int HYPRE_SStructSysPFMGSetMaxIter ( HYPRE_SStructSolver solver , HYPRE_Int max_iter );
HYPRE_Int HYPRE_SStructSysPFMGSetRelChange ( HYPRE_SStructSolver solver , HYPRE_Int rel_change );
HYPRE_Int HYPRE_SStructSysPFMGSetZeroGuess ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructSysPFMGSetNonZeroGuess ( HYPRE_SStructSolver solver );
HYPRE_Int HYPRE_SStructSysPFMGSetRelaxType ( HYPRE_SStructSolver solver , HYPRE_Int relax_type );
HYPRE_Int HYPRE_SStructSysPFMGSetJacobiWeight ( HYPRE_SStructSolver solver , double weight );
HYPRE_Int HYPRE_SStructSysPFMGSetNumPreRelax ( HYPRE_SStructSolver solver , HYPRE_Int num_pre_relax );
HYPRE_Int HYPRE_SStructSysPFMGSetNumPostRelax ( HYPRE_SStructSolver solver , HYPRE_Int num_post_relax );
HYPRE_Int HYPRE_SStructSysPFMGSetSkipRelax ( HYPRE_SStructSolver solver , HYPRE_Int skip_relax );
HYPRE_Int HYPRE_SStructSysPFMGSetDxyz ( HYPRE_SStructSolver solver , double *dxyz );
HYPRE_Int HYPRE_SStructSysPFMGSetLogging ( HYPRE_SStructSolver solver , HYPRE_Int logging );
HYPRE_Int HYPRE_SStructSysPFMGSetPrintLevel ( HYPRE_SStructSolver solver , HYPRE_Int print_level );
HYPRE_Int HYPRE_SStructSysPFMGGetNumIterations ( HYPRE_SStructSolver solver , HYPRE_Int *num_iterations );
HYPRE_Int HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );

/* krylov.c */
HYPRE_Int hypre_SStructKrylovIdentitySetup ( void *vdata , void *A , void *b , void *x );
HYPRE_Int hypre_SStructKrylovIdentity ( void *vdata , void *A , void *b , void *x );

/* krylov_sstruct.c */
char *hypre_SStructKrylovCAlloc ( HYPRE_Int count , HYPRE_Int elt_size );
HYPRE_Int hypre_SStructKrylovFree ( char *ptr );
void *hypre_SStructKrylovCreateVector ( void *vvector );
void *hypre_SStructKrylovCreateVectorArray ( HYPRE_Int n , void *vvector );
HYPRE_Int hypre_SStructKrylovDestroyVector ( void *vvector );
void *hypre_SStructKrylovMatvecCreate ( void *A , void *x );
HYPRE_Int hypre_SStructKrylovMatvec ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
HYPRE_Int hypre_SStructKrylovMatvecDestroy ( void *matvec_data );
double hypre_SStructKrylovInnerProd ( void *x , void *y );
HYPRE_Int hypre_SStructKrylovCopyVector ( void *x , void *y );
HYPRE_Int hypre_SStructKrylovClearVector ( void *x );
HYPRE_Int hypre_SStructKrylovScaleVector ( double alpha , void *x );
HYPRE_Int hypre_SStructKrylovAxpy ( double alpha , void *x , void *y );
HYPRE_Int hypre_SStructKrylovCommInfo ( void *A , HYPRE_Int *my_id , HYPRE_Int *num_procs );

/* maxwell_grad.c */
hypre_ParCSRMatrix *hypre_Maxwell_Grad ( hypre_SStructGrid *grid );

/* maxwell_physbdy.c */
HYPRE_Int hypre_Maxwell_PhysBdy ( hypre_SStructGrid **grid_l , HYPRE_Int num_levels , hypre_Index rfactors , HYPRE_Int ***BdryRanksl_ptr , HYPRE_Int **BdryRanksCntsl_ptr );
HYPRE_Int hypre_Maxwell_VarBdy ( hypre_SStructPGrid *pgrid , hypre_BoxArrayArray **bdry );

/* maxwell_PNedelec_bdy.c */
HYPRE_Int hypre_Maxwell_PNedelec_Bdy ( hypre_StructGrid *cell_grid , hypre_SStructPGrid *pgrid , hypre_BoxArrayArray ****bdry_ptr );

/* maxwell_PNedelec.c */
hypre_IJMatrix *hypre_Maxwell_PNedelec ( hypre_SStructGrid *fgrid_edge , hypre_SStructGrid *cgrid_edge , hypre_Index rfactor );

/* maxwell_semi_interp.c */
HYPRE_Int hypre_CreatePTopology ( void **PTopology_vdata_ptr );
HYPRE_Int hypre_DestroyPTopology ( void *PTopology_vdata );
hypre_IJMatrix *hypre_Maxwell_PTopology ( hypre_SStructGrid *fgrid_edge , hypre_SStructGrid *cgrid_edge , hypre_SStructGrid *fgrid_face , hypre_SStructGrid *cgrid_face , hypre_SStructGrid *fgrid_element , hypre_SStructGrid *cgrid_element , hypre_ParCSRMatrix *Aee , hypre_Index rfactor , void *PTopology_vdata );
HYPRE_Int hypre_CollapseStencilToStencil ( hypre_ParCSRMatrix *Aee , hypre_SStructGrid *grid , HYPRE_Int part , HYPRE_Int var , hypre_Index pt_location , HYPRE_Int collapse_dir , HYPRE_Int new_stencil_dir , double **collapsed_vals_ptr );
HYPRE_Int hypre_TriDiagSolve ( double *diag , double *upper , double *lower , double *rhs , HYPRE_Int size );

/* maxwell_TV.c */
void *hypre_MaxwellTVCreate ( MPI_Comm comm );
HYPRE_Int hypre_MaxwellTVDestroy ( void *maxwell_vdata );
HYPRE_Int hypre_MaxwellSetRfactors ( void *maxwell_vdata , HYPRE_Int rfactor [3 ]);
HYPRE_Int hypre_MaxwellSetGrad ( void *maxwell_vdata , hypre_ParCSRMatrix *T );
HYPRE_Int hypre_MaxwellSetConstantCoef ( void *maxwell_vdata , HYPRE_Int constant_coef );
HYPRE_Int hypre_MaxwellSetTol ( void *maxwell_vdata , double tol );
HYPRE_Int hypre_MaxwellSetMaxIter ( void *maxwell_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_MaxwellSetRelChange ( void *maxwell_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_MaxwellSetNumPreRelax ( void *maxwell_vdata , HYPRE_Int num_pre_relax );
HYPRE_Int hypre_MaxwellSetNumPostRelax ( void *maxwell_vdata , HYPRE_Int num_post_relax );
HYPRE_Int hypre_MaxwellGetNumIterations ( void *maxwell_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_MaxwellSetPrintLevel ( void *maxwell_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_MaxwellSetLogging ( void *maxwell_vdata , HYPRE_Int logging );
HYPRE_Int hypre_MaxwellPrintLogging ( void *maxwell_vdata , HYPRE_Int myid );
HYPRE_Int hypre_MaxwellGetFinalRelativeResidualNorm ( void *maxwell_vdata , double *relative_residual_norm );

/* maxwell_TV_setup.c */
HYPRE_Int hypre_MaxwellTV_Setup ( void *maxwell_vdata , hypre_SStructMatrix *Aee_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );
HYPRE_Int hypre_CoarsenPGrid ( hypre_SStructGrid *fgrid , hypre_Index index , hypre_Index stride , HYPRE_Int part , hypre_SStructGrid *cgrid , HYPRE_Int *nboxes );
hypre_Box *hypre_BoxContraction ( hypre_Box *box , hypre_StructGrid *sgrid , hypre_Index rfactor );

/* maxwell_zeroBC.c */
HYPRE_Int hypre_ParVectorZeroBCValues ( hypre_ParVector *v , HYPRE_Int *rows , HYPRE_Int nrows );
HYPRE_Int hypre_SeqVectorZeroBCValues ( hypre_Vector *v , HYPRE_Int *rows , HYPRE_Int nrows );

/* nd1_amge_interpolation.c */
HYPRE_Int hypre_ND1AMGeInterpolation ( hypre_ParCSRMatrix *Aee , hypre_ParCSRMatrix *ELEM_idof , hypre_ParCSRMatrix *FACE_idof , hypre_ParCSRMatrix *EDGE_idof , hypre_ParCSRMatrix *ELEM_FACE , hypre_ParCSRMatrix *ELEM_EDGE , HYPRE_Int num_OffProcRows , hypre_MaxwellOffProcRow **OffProcRows , hypre_IJMatrix *IJ_dof_DOF );
HYPRE_Int hypre_HarmonicExtension ( hypre_CSRMatrix *A , hypre_CSRMatrix *P , HYPRE_Int num_DOF , HYPRE_Int *DOF , HYPRE_Int num_idof , HYPRE_Int *idof , HYPRE_Int num_bdof , HYPRE_Int *bdof );

/* node_relax.c */
void *hypre_NodeRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_NodeRelaxDestroy ( void *relax_vdata );
HYPRE_Int hypre_NodeRelaxSetup ( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
HYPRE_Int hypre_NodeRelax ( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
HYPRE_Int hypre_NodeRelaxSetTol ( void *relax_vdata , double tol );
HYPRE_Int hypre_NodeRelaxSetMaxIter ( void *relax_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_NodeRelaxSetZeroGuess ( void *relax_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_NodeRelaxSetWeight ( void *relax_vdata , double weight );
HYPRE_Int hypre_NodeRelaxSetNumNodesets ( void *relax_vdata , HYPRE_Int num_nodesets );
HYPRE_Int hypre_NodeRelaxSetNodeset ( void *relax_vdata , HYPRE_Int nodeset , HYPRE_Int nodeset_size , hypre_Index nodeset_stride , hypre_Index *nodeset_indices );
HYPRE_Int hypre_NodeRelaxSetNodesetRank ( void *relax_vdata , HYPRE_Int nodeset , HYPRE_Int nodeset_rank );
HYPRE_Int hypre_NodeRelaxSetTempVec ( void *relax_vdata , hypre_SStructPVector *t );

/* sstruct_amr_intercommunication.c */
HYPRE_Int hypre_SStructAMRInterCommunication ( hypre_SStructSendInfoData *sendinfo , hypre_SStructRecvInfoData *recvinfo , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , HYPRE_Int num_values , MPI_Comm comm , hypre_CommPkg **comm_pkg_ptr );

/* sstruct_owninfo.c */
HYPRE_Int hypre_SStructIndexScaleF_C ( hypre_Index findex , hypre_Index index , hypre_Index stride , hypre_Index cindex );
HYPRE_Int hypre_SStructIndexScaleC_F ( hypre_Index cindex , hypre_Index index , hypre_Index stride , hypre_Index findex );
hypre_SStructOwnInfoData *hypre_SStructOwnInfo ( hypre_StructGrid *fgrid , hypre_StructGrid *cgrid , hypre_BoxManager *cboxman , hypre_BoxManager *fboxman , hypre_Index rfactor );
HYPRE_Int hypre_SStructOwnInfoDataDestroy ( hypre_SStructOwnInfoData *owninfo_data );

/* sstruct_recvinfo.c */
hypre_SStructRecvInfoData *hypre_SStructRecvInfo ( hypre_StructGrid *cgrid , hypre_BoxManager *fboxman , hypre_Index rfactor );
HYPRE_Int hypre_SStructRecvInfoDataDestroy ( hypre_SStructRecvInfoData *recvinfo_data );

/* sstruct_sendinfo.c */
hypre_SStructSendInfoData *hypre_SStructSendInfo ( hypre_StructGrid *fgrid , hypre_BoxManager *cboxman , hypre_Index rfactor );
HYPRE_Int hypre_SStructSendInfoDataDestroy ( hypre_SStructSendInfoData *sendinfo_data );

/* sstruct_sharedDOFComm.c */
hypre_MaxwellOffProcRow *hypre_MaxwellOffProcRowCreate ( HYPRE_Int ncols );
HYPRE_Int hypre_MaxwellOffProcRowDestroy ( void *OffProcRow_vdata );
HYPRE_Int hypre_SStructSharedDOF_ParcsrMatRowsComm ( hypre_SStructGrid *grid , hypre_ParCSRMatrix *A , HYPRE_Int *num_offprocrows_ptr , hypre_MaxwellOffProcRow ***OffProcRows_ptr );

/* sys_pfmg.c */
void *hypre_SysPFMGCreate ( MPI_Comm comm );
HYPRE_Int hypre_SysPFMGDestroy ( void *sys_pfmg_vdata );
HYPRE_Int hypre_SysPFMGSetTol ( void *sys_pfmg_vdata , double tol );
HYPRE_Int hypre_SysPFMGSetMaxIter ( void *sys_pfmg_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_SysPFMGSetRelChange ( void *sys_pfmg_vdata , HYPRE_Int rel_change );
HYPRE_Int hypre_SysPFMGSetZeroGuess ( void *sys_pfmg_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_SysPFMGSetRelaxType ( void *sys_pfmg_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_SysPFMGSetJacobiWeight ( void *sys_pfmg_vdata , double weight );
HYPRE_Int hypre_SysPFMGSetNumPreRelax ( void *sys_pfmg_vdata , HYPRE_Int num_pre_relax );
HYPRE_Int hypre_SysPFMGSetNumPostRelax ( void *sys_pfmg_vdata , HYPRE_Int num_post_relax );
HYPRE_Int hypre_SysPFMGSetSkipRelax ( void *sys_pfmg_vdata , HYPRE_Int skip_relax );
HYPRE_Int hypre_SysPFMGSetDxyz ( void *sys_pfmg_vdata , double *dxyz );
HYPRE_Int hypre_SysPFMGSetLogging ( void *sys_pfmg_vdata , HYPRE_Int logging );
HYPRE_Int hypre_SysPFMGSetPrintLevel ( void *sys_pfmg_vdata , HYPRE_Int print_level );
HYPRE_Int hypre_SysPFMGGetNumIterations ( void *sys_pfmg_vdata , HYPRE_Int *num_iterations );
HYPRE_Int hypre_SysPFMGPrintLogging ( void *sys_pfmg_vdata , HYPRE_Int myid );
HYPRE_Int hypre_SysPFMGGetFinalRelativeResidualNorm ( void *sys_pfmg_vdata , double *relative_residual_norm );

/* sys_pfmg_relax.c */
void *hypre_SysPFMGRelaxCreate ( MPI_Comm comm );
HYPRE_Int hypre_SysPFMGRelaxDestroy ( void *sys_pfmg_relax_vdata );
HYPRE_Int hypre_SysPFMGRelax ( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
HYPRE_Int hypre_SysPFMGRelaxSetup ( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
HYPRE_Int hypre_SysPFMGRelaxSetType ( void *sys_pfmg_relax_vdata , HYPRE_Int relax_type );
HYPRE_Int hypre_SysPFMGRelaxSetJacobiWeight ( void *sys_pfmg_relax_vdata , double weight );
HYPRE_Int hypre_SysPFMGRelaxSetPreRelax ( void *sys_pfmg_relax_vdata );
HYPRE_Int hypre_SysPFMGRelaxSetPostRelax ( void *sys_pfmg_relax_vdata );
HYPRE_Int hypre_SysPFMGRelaxSetTol ( void *sys_pfmg_relax_vdata , double tol );
HYPRE_Int hypre_SysPFMGRelaxSetMaxIter ( void *sys_pfmg_relax_vdata , HYPRE_Int max_iter );
HYPRE_Int hypre_SysPFMGRelaxSetZeroGuess ( void *sys_pfmg_relax_vdata , HYPRE_Int zero_guess );
HYPRE_Int hypre_SysPFMGRelaxSetTempVec ( void *sys_pfmg_relax_vdata , hypre_SStructPVector *t );

/* sys_pfmg_setup.c */
HYPRE_Int hypre_SysPFMGSetup ( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );
HYPRE_Int hypre_SysStructCoarsen ( hypre_SStructPGrid *fgrid , hypre_Index index , hypre_Index stride , HYPRE_Int prune , hypre_SStructPGrid **cgrid_ptr );

/* sys_pfmg_setup_interp.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateInterpOp ( hypre_SStructPMatrix *A , hypre_SStructPGrid *cgrid , HYPRE_Int cdir );
HYPRE_Int hypre_SysPFMGSetupInterpOp ( hypre_SStructPMatrix *A , HYPRE_Int cdir , hypre_Index findex , hypre_Index stride , hypre_SStructPMatrix *P );

/* sys_pfmg_setup_rap.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateRAPOp ( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , hypre_SStructPGrid *coarse_grid , HYPRE_Int cdir );
HYPRE_Int hypre_SysPFMGSetupRAPOp ( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , HYPRE_Int cdir , hypre_Index cindex , hypre_Index cstride , hypre_SStructPMatrix *Ac );

/* sys_pfmg_solve.c */
HYPRE_Int hypre_SysPFMGSolve ( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* sys_semi_interp.c */
HYPRE_Int hypre_SysSemiInterpCreate ( void **sys_interp_vdata_ptr );
HYPRE_Int hypre_SysSemiInterpSetup ( void *sys_interp_vdata , hypre_SStructPMatrix *P , HYPRE_Int P_stored_as_transpose , hypre_SStructPVector *xc , hypre_SStructPVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
HYPRE_Int hypre_SysSemiInterp ( void *sys_interp_vdata , hypre_SStructPMatrix *P , hypre_SStructPVector *xc , hypre_SStructPVector *e );
HYPRE_Int hypre_SysSemiInterpDestroy ( void *sys_interp_vdata );

/* sys_semi_restrict.c */
HYPRE_Int hypre_SysSemiRestrictCreate ( void **sys_restrict_vdata_ptr );
HYPRE_Int hypre_SysSemiRestrictSetup ( void *sys_restrict_vdata , hypre_SStructPMatrix *R , HYPRE_Int R_stored_as_transpose , hypre_SStructPVector *r , hypre_SStructPVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
HYPRE_Int hypre_SysSemiRestrict ( void *sys_restrict_vdata , hypre_SStructPMatrix *R , hypre_SStructPVector *r , hypre_SStructPVector *rc );
HYPRE_Int hypre_SysSemiRestrictDestroy ( void *sys_restrict_vdata );

#ifdef __cplusplus
}
#endif

#endif

