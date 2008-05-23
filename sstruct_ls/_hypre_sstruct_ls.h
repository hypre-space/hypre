/*BHEADER**********************************************************************
 * Copyright (c) 2007,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
   int                   size;

   hypre_BoxArrayArray  *own_boxes;    /* size of fgrid */
   int                 **own_cboxnums; /* local cbox number- each fbox
                                          leads to an array of cboxes */

   hypre_BoxArrayArray  *own_composite_cboxes;  /* size of cgrid */
   int                   own_composite_size;
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
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructRecvInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_RECVINFODATA_HEADER
#define hypre_RECVINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *recv_boxes;
   int                 **recv_procs;

} hypre_SStructRecvInfoData;

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct 
{
   int                   size;

   hypre_BoxArrayArray  *send_boxes;
   int                 **send_procs;
   int                 **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
   int                     max_iter;
   int                     rel_change;
   int                     zero_guess;
   int                     ndim;
                      
   int                     num_pre_relax;  /* number of pre relaxation sweeps */
   int                     num_post_relax; /* number of post relaxation sweeps */

   int                     constant_coef;
   
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
   int                   **nCF_marker_l;
   double                 *nrelax_weight;
   double                 *nomega;
   int                     nrelax_type;
   int                     node_numlevels;

   hypre_ParCSRMatrix     *Tgrad;
   hypre_ParCSRMatrix     *T_transpose;

   /* edge data structure. These will have grids. */
   int                     edge_maxlevels;  
   int                     edge_numlevels;  
   hypre_ParCSRMatrix    **Aee_l;
   hypre_ParVector       **be_l;
   hypre_ParVector       **xe_l;
   hypre_ParVector       **rese_l;
   hypre_ParVector       **ee_l;
   hypre_ParVector       **eVtemp_l;
   hypre_ParVector       **eVtemp2_l;
   int                   **eCF_marker_l;
   double                 *erelax_weight;
   double                 *eomega;
   int                     erelax_type;

   /* edge data structure. These will have no grid. */
   hypre_IJMatrix        **Pe_l;
   hypre_IJMatrix        **ReT_l;
   int                   **BdryRanks_l;
   int                    *BdryRanksCnts_l;

   /* edge-node data structure. These will have grids. */
   int                     en_numlevels;

   /* log info (always logged) */
   int                     num_iterations;
   int                     time_index;

   /* additional log info (logged when `logging' > 0) */
   int                     print_level;
   int                     logging;
   double                 *norms;
   double                 *rel_norms;

} hypre_MaxwellData;

#endif
/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




typedef struct
{
   int       row;
                                                                                                                                    
   int       ncols;
   int      *cols;
   double   *data;
                                                                                                                                    
} hypre_MaxwellOffProcRow;


/* eliminate_rowscols.c */
int hypre_ParCSRMatrixEliminateRowsCols ( hypre_ParCSRMatrix *A , int nrows_to_eliminate , int *rows_to_eliminate );
int hypre_CSRMatrixEliminateRowsColsDiag ( hypre_ParCSRMatrix *A , int nrows_to_eliminate , int *rows_to_eliminate );
int hypre_CSRMatrixEliminateRowsOffd ( hypre_ParCSRMatrix *A , int nrows_to_eliminate , int *rows_to_eliminate );
int hypre_CSRMatrixEliminateColsOffd ( hypre_CSRMatrix *Aoffd , int ncols_to_eliminate , int *cols_to_eliminate );

/* fac_amr_fcoarsen.c */
int hypre_AMR_FCoarsen ( hypre_SStructMatrix *A , hypre_SStructMatrix *fac_A , hypre_SStructPMatrix *A_crse , hypre_Index refine_factors , int level );

/* fac_amr_rap.c */
int hypre_AMR_RAP ( hypre_SStructMatrix *A , hypre_Index *rfactors , hypre_SStructMatrix **fac_A_ptr );

/* fac_amr_zero_data.c */
int hypre_ZeroAMRVectorData ( hypre_SStructVector *b , int *plevels , hypre_Index *rfactors );
int hypre_ZeroAMRMatrixData ( hypre_SStructMatrix *A , int part_crse , hypre_Index rfactors );

/* fac.c */
void *hypre_FACCreate ( MPI_Comm comm );
int hypre_FACDestroy2 ( void *fac_vdata );
int hypre_FACSetTol ( void *fac_vdata , double tol );
int hypre_FACSetPLevels ( void *fac_vdata , int nparts , int *plevels );
int hypre_FACSetPRefinements ( void *fac_vdata , int nparts , int (*prefinements )[3 ]);
int hypre_FACSetMaxLevels ( void *fac_vdata , int nparts );
int hypre_FACSetMaxIter ( void *fac_vdata , int max_iter );
int hypre_FACSetRelChange ( void *fac_vdata , int rel_change );
int hypre_FACSetZeroGuess ( void *fac_vdata , int zero_guess );
int hypre_FACSetRelaxType ( void *fac_vdata , int relax_type );
int hypre_FACSetJacobiWeight ( void *fac_vdata , double weight );
int hypre_FACSetNumPreSmooth ( void *fac_vdata , int num_pre_smooth );
int hypre_FACSetNumPostSmooth ( void *fac_vdata , int num_post_smooth );
int hypre_FACSetCoarseSolverType ( void *fac_vdata , int csolver_type );
int hypre_FACSetLogging ( void *fac_vdata , int logging );
int hypre_FACGetNumIterations ( void *fac_vdata , int *num_iterations );
int hypre_FACPrintLogging ( void *fac_vdata , int myid );
int hypre_FACGetFinalRelativeResidualNorm ( void *fac_vdata , double *relative_residual_norm );

/* fac_cf_coarsen.c */
int hypre_AMR_CFCoarsen ( hypre_SStructMatrix *A , hypre_SStructMatrix *fac_A , hypre_Index refine_factors , int level );

/* fac_CFInterfaceExtents.c */
hypre_BoxArray *hypre_CFInterfaceExtents ( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_StructStencil *stencils , hypre_Index rfactors );
int hypre_CFInterfaceExtents2 ( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_StructStencil *stencils , hypre_Index rfactors , hypre_BoxArray *cf_interface );

/* fac_cfstencil_box.c */
hypre_Box *hypre_CF_StenBox ( hypre_Box *fgrid_box , hypre_Box *cgrid_box , hypre_Index stencil_shape , hypre_Index rfactors , int ndim );

/* fac_interp2.c */
int hypre_FacSemiInterpCreate2 ( void **fac_interp_vdata_ptr );
int hypre_FacSemiInterpDestroy2 ( void *fac_interp_vdata );
int hypre_FacSemiInterpSetup2 ( void *fac_interp_vdata , hypre_SStructVector *e , hypre_SStructPVector *ec , hypre_Index rfactors );
int hypre_FAC_IdentityInterp2 ( void *fac_interp_vdata , hypre_SStructPVector *xc , hypre_SStructVector *e );
int hypre_FAC_WeightedInterp2 ( void *fac_interp_vdata , hypre_SStructPVector *xc , hypre_SStructVector *e_parts );

/* fac_relax.c */
int hypre_FacLocalRelax ( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *x , hypre_SStructPVector *b , int num_relax , int *zero_guess );

/* fac_restrict2.c */
int hypre_FacSemiRestrictCreate2 ( void **fac_restrict_vdata_ptr );
int hypre_FacSemiRestrictSetup2 ( void *fac_restrict_vdata , hypre_SStructVector *r , int part_crse , int part_fine , hypre_SStructPVector *rc , hypre_Index rfactors );
int hypre_FACRestrict2 ( void *fac_restrict_vdata , hypre_SStructVector *xf , hypre_SStructPVector *xc );
int hypre_FacSemiRestrictDestroy2 ( void *fac_restrict_vdata );

/* fac_setup2.c */
int hypre_FacSetup2 ( void *fac_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b , hypre_SStructVector *x );

/* fac_solve3.c */
int hypre_FACSolve3 ( void *fac_vdata , hypre_SStructMatrix *A_user , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* fac_zero_cdata.c */
int hypre_FacZeroCData ( void *fac_vdata , hypre_SStructMatrix *A );

/* fac_zero_stencilcoef.c */
int hypre_FacZeroCFSten ( hypre_SStructPMatrix *Af , hypre_SStructPMatrix *Ac , hypre_SStructGrid *grid , int fine_part , hypre_Index rfactors );
int hypre_FacZeroFCSten ( hypre_SStructPMatrix *A , hypre_SStructGrid *grid , int fine_part );

/* hypre_bsearch.c */
int hypre_LowerBinarySearch ( int *list , int value , int list_length );
int hypre_UpperBinarySearch ( int *list , int value , int list_length );

/* hypre_MaxwellSolve2.c */
int hypre_MaxwellSolve2 ( void *maxwell_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *f , hypre_SStructVector *u );

/* hypre_MaxwellSolve.c */
int hypre_MaxwellSolve ( void *maxwell_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *f , hypre_SStructVector *u );

/* HYPRE_sstruct_bicgstab.c */
int HYPRE_SStructBiCGSTABCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructBiCGSTABDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructBiCGSTABSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructBiCGSTABSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructBiCGSTABSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructBiCGSTABSetAbsoluteTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructBiCGSTABSetMinIter ( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructBiCGSTABSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructBiCGSTABSetStopCrit ( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructBiCGSTABSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructBiCGSTABSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructBiCGSTABSetPrintLevel ( HYPRE_SStructSolver solver , int print_level );
int HYPRE_SStructBiCGSTABGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructBiCGSTABGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_flexgmres.c */
int HYPRE_SStructFlexGMRESCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructFlexGMRESDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructFlexGMRESSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructFlexGMRESSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructFlexGMRESSetKDim ( HYPRE_SStructSolver solver , int k_dim );
int HYPRE_SStructFlexGMRESSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructFlexGMRESSetAbsoluteTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructFlexGMRESSetMinIter ( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructFlexGMRESSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructFlexGMRESSetStopCrit ( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructFlexGMRESSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructFlexGMRESSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructFlexGMRESSetPrintLevel ( HYPRE_SStructSolver solver , int level );
int HYPRE_SStructFlexGMRESGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructFlexGMRESGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_gmres.c */
int HYPRE_SStructGMRESCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructGMRESDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructGMRESSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructGMRESSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructGMRESSetKDim ( HYPRE_SStructSolver solver , int k_dim );
int HYPRE_SStructGMRESSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructGMRESSetAbsoluteTol ( HYPRE_SStructSolver solver , double atol );
int HYPRE_SStructGMRESSetMinIter ( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructGMRESSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructGMRESSetStopCrit ( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructGMRESSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructGMRESSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructGMRESSetPrintLevel ( HYPRE_SStructSolver solver , int level );
int HYPRE_SStructGMRESGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructGMRESGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructGMRESGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_int.c */
int hypre_SStructPVectorSetRandomValues ( hypre_SStructPVector *pvector , int seed );
int hypre_SStructVectorSetRandomValues ( hypre_SStructVector *vector , int seed );
int hypre_SStructSetRandomValues ( void *v , int seed );
int HYPRE_SStructSetupInterpreter ( mv_InterfaceInterpreter *i );
int HYPRE_SStructSetupMatvec ( HYPRE_MatvecFunctions *mv );

/* HYPRE_sstruct_InterFAC.c */
int HYPRE_SStructFACCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructFACDestroy2 ( HYPRE_SStructSolver solver );
int HYPRE_SStructFACAMR_RAP ( HYPRE_SStructMatrix A , int (*rfactors )[3 ], HYPRE_SStructMatrix *fac_A );
int HYPRE_SStructFACSetup2 ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructFACSolve3 ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructFACSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructFACSetPLevels ( HYPRE_SStructSolver solver , int nparts , int *plevels );
int HYPRE_SStructFACZeroCFSten ( HYPRE_SStructMatrix A , HYPRE_SStructGrid grid , int part , int rfactors [3 ]);
int HYPRE_SStructFACZeroFCSten ( HYPRE_SStructMatrix A , HYPRE_SStructGrid grid , int part );
int HYPRE_SStructFACZeroAMRMatrixData ( HYPRE_SStructMatrix A , int part_crse , int rfactors [3 ]);
int HYPRE_SStructFACZeroAMRVectorData ( HYPRE_SStructVector b , int *plevels , int (*rfactors )[3 ]);
int HYPRE_SStructFACSetPRefinements ( HYPRE_SStructSolver solver , int nparts , int (*rfactors )[3 ]);
int HYPRE_SStructFACSetMaxLevels ( HYPRE_SStructSolver solver , int max_levels );
int HYPRE_SStructFACSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructFACSetRelChange ( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructFACSetZeroGuess ( HYPRE_SStructSolver solver );
int HYPRE_SStructFACSetNonZeroGuess ( HYPRE_SStructSolver solver );
int HYPRE_SStructFACSetRelaxType ( HYPRE_SStructSolver solver , int relax_type );
int HYPRE_SStructFACSetJacobiWeight ( HYPRE_SStructSolver solver , double weight );
int HYPRE_SStructFACSetNumPreRelax ( HYPRE_SStructSolver solver , int num_pre_relax );
int HYPRE_SStructFACSetNumPostRelax ( HYPRE_SStructSolver solver , int num_post_relax );
int HYPRE_SStructFACSetCoarseSolverType ( HYPRE_SStructSolver solver , int csolver_type );
int HYPRE_SStructFACSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructFACGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructFACGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_lgmres.c */
int HYPRE_SStructLGMRESCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructLGMRESDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructLGMRESSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructLGMRESSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructLGMRESSetKDim ( HYPRE_SStructSolver solver , int k_dim );
int HYPRE_SStructLGMRESSetAugDim ( HYPRE_SStructSolver solver , int aug_dim );
int HYPRE_SStructLGMRESSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructLGMRESSetAbsoluteTol ( HYPRE_SStructSolver solver , double atol );
int HYPRE_SStructLGMRESSetMinIter ( HYPRE_SStructSolver solver , int min_iter );
int HYPRE_SStructLGMRESSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructLGMRESSetStopCrit ( HYPRE_SStructSolver solver , int stop_crit );
int HYPRE_SStructLGMRESSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructLGMRESSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructLGMRESSetPrintLevel ( HYPRE_SStructSolver solver , int level );
int HYPRE_SStructLGMRESGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructLGMRESGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructLGMRESGetResidual ( HYPRE_SStructSolver solver , void **residual );

/* HYPRE_sstruct_maxwell.c */
int HYPRE_SStructMaxwellCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructMaxwellDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructMaxwellSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructMaxwellSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructMaxwellSolve2 ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_MaxwellGrad ( HYPRE_SStructGrid grid , HYPRE_ParCSRMatrix *T );
int HYPRE_SStructMaxwellSetGrad ( HYPRE_SStructSolver solver , HYPRE_ParCSRMatrix T );
int HYPRE_SStructMaxwellSetRfactors ( HYPRE_SStructSolver solver , int rfactors [3 ]);
int HYPRE_SStructMaxwellSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructMaxwellSetConstantCoef ( HYPRE_SStructSolver solver , int constant_coef );
int HYPRE_SStructMaxwellSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructMaxwellSetRelChange ( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructMaxwellSetNumPreRelax ( HYPRE_SStructSolver solver , int num_pre_relax );
int HYPRE_SStructMaxwellSetNumPostRelax ( HYPRE_SStructSolver solver , int num_post_relax );
int HYPRE_SStructMaxwellSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructMaxwellSetPrintLevel ( HYPRE_SStructSolver solver , int print_level );
int HYPRE_SStructMaxwellPrintLogging ( HYPRE_SStructSolver solver , int myid );
int HYPRE_SStructMaxwellGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructMaxwellGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructMaxwellPhysBdy ( HYPRE_SStructGrid *grid_l , int num_levels , int rfactors [3 ], int ***BdryRanks_ptr , int **BdryRanksCnt_ptr );
int HYPRE_SStructMaxwellEliminateRowsCols ( HYPRE_ParCSRMatrix parA , int nrows , int *rows );
int HYPRE_SStructMaxwellZeroVector ( HYPRE_ParVector v , int *rows , int nrows );

/* HYPRE_sstruct_pcg.c */
int HYPRE_SStructPCGCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructPCGDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructPCGSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructPCGSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructPCGSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructPCGSetAbsoluteTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructPCGSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructPCGSetTwoNorm ( HYPRE_SStructSolver solver , int two_norm );
int HYPRE_SStructPCGSetRelChange ( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructPCGSetPrecond ( HYPRE_SStructSolver solver , HYPRE_PtrToSStructSolverFcn precond , HYPRE_PtrToSStructSolverFcn precond_setup , void *precond_data );
int HYPRE_SStructPCGSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructPCGSetPrintLevel ( HYPRE_SStructSolver solver , int level );
int HYPRE_SStructPCGGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructPCGGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );
int HYPRE_SStructPCGGetResidual ( HYPRE_SStructSolver solver , void **residual );
int HYPRE_SStructDiagScaleSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );
int HYPRE_SStructDiagScale ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector y , HYPRE_SStructVector x );

/* HYPRE_sstruct_split.c */
int HYPRE_SStructSplitCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver_ptr );
int HYPRE_SStructSplitDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSplitSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSplitSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructSplitSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructSplitSetZeroGuess ( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetNonZeroGuess ( HYPRE_SStructSolver solver );
int HYPRE_SStructSplitSetStructSolver ( HYPRE_SStructSolver solver , int ssolver );
int HYPRE_SStructSplitGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructSplitGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );

/* HYPRE_sstruct_sys_pfmg.c */
int HYPRE_SStructSysPFMGCreate ( MPI_Comm comm , HYPRE_SStructSolver *solver );
int HYPRE_SStructSysPFMGDestroy ( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetup ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSysPFMGSolve ( HYPRE_SStructSolver solver , HYPRE_SStructMatrix A , HYPRE_SStructVector b , HYPRE_SStructVector x );
int HYPRE_SStructSysPFMGSetTol ( HYPRE_SStructSolver solver , double tol );
int HYPRE_SStructSysPFMGSetMaxIter ( HYPRE_SStructSolver solver , int max_iter );
int HYPRE_SStructSysPFMGSetRelChange ( HYPRE_SStructSolver solver , int rel_change );
int HYPRE_SStructSysPFMGSetZeroGuess ( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetNonZeroGuess ( HYPRE_SStructSolver solver );
int HYPRE_SStructSysPFMGSetRelaxType ( HYPRE_SStructSolver solver , int relax_type );
int HYPRE_SStructSysPFMGSetJacobiWeight ( HYPRE_SStructSolver solver , double weight );
int HYPRE_SStructSysPFMGSetNumPreRelax ( HYPRE_SStructSolver solver , int num_pre_relax );
int HYPRE_SStructSysPFMGSetNumPostRelax ( HYPRE_SStructSolver solver , int num_post_relax );
int HYPRE_SStructSysPFMGSetSkipRelax ( HYPRE_SStructSolver solver , int skip_relax );
int HYPRE_SStructSysPFMGSetDxyz ( HYPRE_SStructSolver solver , double *dxyz );
int HYPRE_SStructSysPFMGSetLogging ( HYPRE_SStructSolver solver , int logging );
int HYPRE_SStructSysPFMGSetPrintLevel ( HYPRE_SStructSolver solver , int print_level );
int HYPRE_SStructSysPFMGGetNumIterations ( HYPRE_SStructSolver solver , int *num_iterations );
int HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm ( HYPRE_SStructSolver solver , double *norm );

/* krylov.c */
int hypre_SStructKrylovIdentitySetup ( void *vdata , void *A , void *b , void *x );
int hypre_SStructKrylovIdentity ( void *vdata , void *A , void *b , void *x );

/* krylov_sstruct.c */
char *hypre_SStructKrylovCAlloc ( int count , int elt_size );
int hypre_SStructKrylovFree ( char *ptr );
void *hypre_SStructKrylovCreateVector ( void *vvector );
void *hypre_SStructKrylovCreateVectorArray ( int n , void *vvector );
int hypre_SStructKrylovDestroyVector ( void *vvector );
void *hypre_SStructKrylovMatvecCreate ( void *A , void *x );
int hypre_SStructKrylovMatvec ( void *matvec_data , double alpha , void *A , void *x , double beta , void *y );
int hypre_SStructKrylovMatvecDestroy ( void *matvec_data );
double hypre_SStructKrylovInnerProd ( void *x , void *y );
int hypre_SStructKrylovCopyVector ( void *x , void *y );
int hypre_SStructKrylovClearVector ( void *x );
int hypre_SStructKrylovScaleVector ( double alpha , void *x );
int hypre_SStructKrylovAxpy ( double alpha , void *x , void *y );
int hypre_SStructKrylovCommInfo ( void *A , int *my_id , int *num_procs );

/* maxwell_grad.c */
hypre_ParCSRMatrix *hypre_Maxwell_Grad ( hypre_SStructGrid *grid );

/* maxwell_physbdy.c */
int hypre_Maxwell_PhysBdy ( hypre_SStructGrid **grid_l , int num_levels , hypre_Index rfactors , int ***BdryRanksl_ptr , int **BdryRanksCntsl_ptr );
int hypre_Maxwell_VarBdy ( hypre_SStructPGrid *pgrid , hypre_BoxArrayArray **bdry );

/* maxwell_PNedelec_bdy.c */
int hypre_Maxwell_PNedelec_Bdy ( hypre_StructGrid *cell_grid , hypre_SStructPGrid *pgrid , hypre_BoxArrayArray ****bdry_ptr );

/* maxwell_PNedelec.c */
hypre_IJMatrix *hypre_Maxwell_PNedelec ( hypre_SStructGrid *fgrid_edge , hypre_SStructGrid *cgrid_edge , hypre_Index rfactor );

/* maxwell_semi_interp.c */
int hypre_CreatePTopology ( void **PTopology_vdata_ptr );
int hypre_DestroyPTopology ( void *PTopology_vdata );
hypre_IJMatrix *hypre_Maxwell_PTopology ( hypre_SStructGrid *fgrid_edge , hypre_SStructGrid *cgrid_edge , hypre_SStructGrid *fgrid_face , hypre_SStructGrid *cgrid_face , hypre_SStructGrid *fgrid_element , hypre_SStructGrid *cgrid_element , hypre_ParCSRMatrix *Aee , hypre_Index rfactor , void *PTopology_vdata );
int hypre_CollapseStencilToStencil ( hypre_ParCSRMatrix *Aee , hypre_SStructGrid *grid , int part , int var , hypre_Index pt_location , int collapse_dir , int new_stencil_dir , double **collapsed_vals_ptr );
int hypre_TriDiagSolve ( double *diag , double *upper , double *lower , double *rhs , int size );

/* maxwell_TV.c */
void *hypre_MaxwellTVCreate ( MPI_Comm comm );
int hypre_MaxwellTVDestroy ( void *maxwell_vdata );
int hypre_MaxwellSetRfactors ( void *maxwell_vdata , int rfactor [3 ]);
int hypre_MaxwellSetGrad ( void *maxwell_vdata , hypre_ParCSRMatrix *T );
int hypre_MaxwellSetConstantCoef ( void *maxwell_vdata , int constant_coef );
int hypre_MaxwellSetTol ( void *maxwell_vdata , double tol );
int hypre_MaxwellSetMaxIter ( void *maxwell_vdata , int max_iter );
int hypre_MaxwellSetRelChange ( void *maxwell_vdata , int rel_change );
int hypre_MaxwellSetNumPreRelax ( void *maxwell_vdata , int num_pre_relax );
int hypre_MaxwellSetNumPostRelax ( void *maxwell_vdata , int num_post_relax );
int hypre_MaxwellGetNumIterations ( void *maxwell_vdata , int *num_iterations );
int hypre_MaxwellSetPrintLevel ( void *maxwell_vdata , int print_level );
int hypre_MaxwellSetLogging ( void *maxwell_vdata , int logging );
int hypre_MaxwellPrintLogging ( void *maxwell_vdata , int myid );
int hypre_MaxwellGetFinalRelativeResidualNorm ( void *maxwell_vdata , double *relative_residual_norm );

/* maxwell_TV_setup.c */
int hypre_MaxwellTV_Setup ( void *maxwell_vdata , hypre_SStructMatrix *Aee_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );
int hypre_CoarsenPGrid ( hypre_SStructGrid *fgrid , hypre_Index index , hypre_Index stride , int part , hypre_SStructGrid *cgrid , int *nboxes );
hypre_Box *hypre_BoxContraction ( hypre_Box *box , hypre_StructGrid *sgrid , hypre_Index rfactor );

/* maxwell_zeroBC.c */
int hypre_ParVectorZeroBCValues ( hypre_ParVector *v , int *rows , int nrows );
int hypre_SeqVectorZeroBCValues ( hypre_Vector *v , int *rows , int nrows );

/* nd1_amge_interpolation.c */
int hypre_ND1AMGeInterpolation ( hypre_ParCSRMatrix *Aee , hypre_ParCSRMatrix *ELEM_idof , hypre_ParCSRMatrix *FACE_idof , hypre_ParCSRMatrix *EDGE_idof , hypre_ParCSRMatrix *ELEM_FACE , hypre_ParCSRMatrix *ELEM_EDGE , int num_OffProcRows , hypre_MaxwellOffProcRow **OffProcRows , hypre_IJMatrix *IJ_dof_DOF );
int hypre_HarmonicExtension ( hypre_CSRMatrix *A , hypre_CSRMatrix *P , int num_DOF , int *DOF , int num_idof , int *idof , int num_bdof , int *bdof );

/* node_relax.c */
void *hypre_NodeRelaxCreate ( MPI_Comm comm );
int hypre_NodeRelaxDestroy ( void *relax_vdata );
int hypre_NodeRelaxSetup ( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_NodeRelax ( void *relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_NodeRelaxSetTol ( void *relax_vdata , double tol );
int hypre_NodeRelaxSetMaxIter ( void *relax_vdata , int max_iter );
int hypre_NodeRelaxSetZeroGuess ( void *relax_vdata , int zero_guess );
int hypre_NodeRelaxSetWeight ( void *relax_vdata , double weight );
int hypre_NodeRelaxSetNumNodesets ( void *relax_vdata , int num_nodesets );
int hypre_NodeRelaxSetNodeset ( void *relax_vdata , int nodeset , int nodeset_size , hypre_Index nodeset_stride , hypre_Index *nodeset_indices );
int hypre_NodeRelaxSetNodesetRank ( void *relax_vdata , int nodeset , int nodeset_rank );
int hypre_NodeRelaxSetTempVec ( void *relax_vdata , hypre_SStructPVector *t );

/* sstruct_amr_intercommunication.c */
int hypre_SStructAMRInterCommunication ( hypre_SStructSendInfoData *sendinfo , hypre_SStructRecvInfoData *recvinfo , hypre_BoxArray *send_data_space , hypre_BoxArray *recv_data_space , int num_values , MPI_Comm comm , hypre_CommPkg **comm_pkg_ptr );

/* sstruct_owninfo.c */
int hypre_SStructIndexScaleF_C ( hypre_Index findex , hypre_Index index , hypre_Index stride , hypre_Index cindex );
int hypre_SStructIndexScaleC_F ( hypre_Index cindex , hypre_Index index , hypre_Index stride , hypre_Index findex );
hypre_SStructOwnInfoData *hypre_SStructOwnInfo ( hypre_StructGrid *fgrid , hypre_StructGrid *cgrid , hypre_BoxMap *cmap , hypre_BoxMap *fmap , hypre_Index rfactor );
int hypre_SStructOwnInfoDataDestroy ( hypre_SStructOwnInfoData *owninfo_data );

/* sstruct_recvinfo.c */
hypre_SStructRecvInfoData *hypre_SStructRecvInfo ( hypre_StructGrid *cgrid , hypre_BoxMap *fmap , hypre_Index rfactor );
int hypre_SStructRecvInfoDataDestroy ( hypre_SStructRecvInfoData *recvinfo_data );

/* sstruct_sendinfo.c */
hypre_SStructSendInfoData *hypre_SStructSendInfo ( hypre_StructGrid *fgrid , hypre_BoxMap *cmap , hypre_Index rfactor );
int hypre_SStructSendInfoDataDestroy ( hypre_SStructSendInfoData *sendinfo_data );

/* sstruct_sharedDOFComm.c */
hypre_MaxwellOffProcRow *hypre_MaxwellOffProcRowCreate ( int ncols );
int hypre_MaxwellOffProcRowDestroy ( void *OffProcRow_vdata );
int hypre_SStructSharedDOF_ParcsrMatRowsComm ( hypre_SStructGrid *grid , hypre_ParCSRMatrix *A , int *num_offprocrows_ptr , hypre_MaxwellOffProcRow ***OffProcRows_ptr );

/* sys_pfmg.c */
void *hypre_SysPFMGCreate ( MPI_Comm comm );
int hypre_SysPFMGDestroy ( void *sys_pfmg_vdata );
int hypre_SysPFMGSetTol ( void *sys_pfmg_vdata , double tol );
int hypre_SysPFMGSetMaxIter ( void *sys_pfmg_vdata , int max_iter );
int hypre_SysPFMGSetRelChange ( void *sys_pfmg_vdata , int rel_change );
int hypre_SysPFMGSetZeroGuess ( void *sys_pfmg_vdata , int zero_guess );
int hypre_SysPFMGSetRelaxType ( void *sys_pfmg_vdata , int relax_type );
int hypre_SysPFMGSetJacobiWeight ( void *sys_pfmg_vdata , double weight );
int hypre_SysPFMGSetNumPreRelax ( void *sys_pfmg_vdata , int num_pre_relax );
int hypre_SysPFMGSetNumPostRelax ( void *sys_pfmg_vdata , int num_post_relax );
int hypre_SysPFMGSetSkipRelax ( void *sys_pfmg_vdata , int skip_relax );
int hypre_SysPFMGSetDxyz ( void *sys_pfmg_vdata , double *dxyz );
int hypre_SysPFMGSetLogging ( void *sys_pfmg_vdata , int logging );
int hypre_SysPFMGSetPrintLevel ( void *sys_pfmg_vdata , int print_level );
int hypre_SysPFMGGetNumIterations ( void *sys_pfmg_vdata , int *num_iterations );
int hypre_SysPFMGPrintLogging ( void *sys_pfmg_vdata , int myid );
int hypre_SysPFMGGetFinalRelativeResidualNorm ( void *sys_pfmg_vdata , double *relative_residual_norm );

/* sys_pfmg_relax.c */
void *hypre_SysPFMGRelaxCreate ( MPI_Comm comm );
int hypre_SysPFMGRelaxDestroy ( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelax ( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_SysPFMGRelaxSetup ( void *sys_pfmg_relax_vdata , hypre_SStructPMatrix *A , hypre_SStructPVector *b , hypre_SStructPVector *x );
int hypre_SysPFMGRelaxSetType ( void *sys_pfmg_relax_vdata , int relax_type );
int hypre_SysPFMGRelaxSetJacobiWeight ( void *sys_pfmg_relax_vdata , double weight );
int hypre_SysPFMGRelaxSetPreRelax ( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelaxSetPostRelax ( void *sys_pfmg_relax_vdata );
int hypre_SysPFMGRelaxSetTol ( void *sys_pfmg_relax_vdata , double tol );
int hypre_SysPFMGRelaxSetMaxIter ( void *sys_pfmg_relax_vdata , int max_iter );
int hypre_SysPFMGRelaxSetZeroGuess ( void *sys_pfmg_relax_vdata , int zero_guess );
int hypre_SysPFMGRelaxSetTempVec ( void *sys_pfmg_relax_vdata , hypre_SStructPVector *t );

/* sys_pfmg_setup.c */
int hypre_SysPFMGSetup ( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );
int hypre_SysStructCoarsen ( hypre_SStructPGrid *fgrid , hypre_Index index , hypre_Index stride , int prune , hypre_SStructPGrid **cgrid_ptr );

/* sys_pfmg_setup_interp.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateInterpOp ( hypre_SStructPMatrix *A , hypre_SStructPGrid *cgrid , int cdir );
int hypre_SysPFMGSetupInterpOp ( hypre_SStructPMatrix *A , int cdir , hypre_Index findex , hypre_Index stride , hypre_SStructPMatrix *P );

/* sys_pfmg_setup_rap.c */
hypre_SStructPMatrix *hypre_SysPFMGCreateRAPOp ( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , hypre_SStructPGrid *coarse_grid , int cdir );
int hypre_SysPFMGSetupRAPOp ( hypre_SStructPMatrix *R , hypre_SStructPMatrix *A , hypre_SStructPMatrix *P , int cdir , hypre_Index cindex , hypre_Index cstride , hypre_SStructPMatrix *Ac );

/* sys_pfmg_solve.c */
int hypre_SysPFMGSolve ( void *sys_pfmg_vdata , hypre_SStructMatrix *A_in , hypre_SStructVector *b_in , hypre_SStructVector *x_in );

/* sys_semi_interp.c */
int hypre_SysSemiInterpCreate ( void **sys_interp_vdata_ptr );
int hypre_SysSemiInterpSetup ( void *sys_interp_vdata , hypre_SStructPMatrix *P , int P_stored_as_transpose , hypre_SStructPVector *xc , hypre_SStructPVector *e , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SysSemiInterp ( void *sys_interp_vdata , hypre_SStructPMatrix *P , hypre_SStructPVector *xc , hypre_SStructPVector *e );
int hypre_SysSemiInterpDestroy ( void *sys_interp_vdata );

/* sys_semi_restrict.c */
int hypre_SysSemiRestrictCreate ( void **sys_restrict_vdata_ptr );
int hypre_SysSemiRestrictSetup ( void *sys_restrict_vdata , hypre_SStructPMatrix *R , int R_stored_as_transpose , hypre_SStructPVector *r , hypre_SStructPVector *rc , hypre_Index cindex , hypre_Index findex , hypre_Index stride );
int hypre_SysSemiRestrict ( void *sys_restrict_vdata , hypre_SStructPMatrix *R , hypre_SStructPVector *r , hypre_SStructPVector *rc );
int hypre_SysSemiRestrictDestroy ( void *sys_restrict_vdata );

#ifdef __cplusplus
}
#endif

#endif

