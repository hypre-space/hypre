/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#define DEBUG 0
#define PRINT_CF 0

#define DEBUG_SAVE_ALL_OPS 0
/*****************************************************************************
 *
 * Routines for driving the setup phase of AMG
 *
 *****************************************************************************/
/**** Get the Strength Matrix ****/
HYPRE_Int hypre_Strength_Options(hypre_ParCSRMatrix   *A,
		                 HYPRE_Real            strong_threshold,
				 HYPRE_Real            max_row_sum,
				 HYPRE_Int             num_functions,
				 HYPRE_Int             nodal,
				 HYPRE_Int             nodal_diag,
				 HYPRE_Int             useSabs,
				 HYPRE_Int            *dof_func_data,
				 hypre_ParCSRMatrix  **S_ptr)
{
   hypre_ParCSRMatrix *S = NULL;
   hypre_ParCSRMatrix *AN = NULL;

   if (nodal) /* if we are solving systems and
                 not using the unknown approach then we need to
                 convert A to a nodal matrix - values that represent the
                 blocks  - before getting the strength matrix*/
   {
      hypre_BoomerAMGCreateNodalA(A, num_functions,
                                  dof_func_data, hypre_abs(nodal), nodal_diag, &AN);
      /* dof array not needed for creating S because we pass in that
         the number of functions is 1 */
      /* creat s two different ways - depending on if any entries in AN are negative: */

      /* first: positive and negative entries */
      if (nodal == 3 || nodal == 6 || nodal_diag > 0)
      {
         hypre_BoomerAMGCreateS(AN, strong_threshold, max_row_sum,
                                1, NULL, &S);
      }
      else /* all entries are positive */
      {
         hypre_BoomerAMGCreateSabs(AN, strong_threshold, max_row_sum,
                                  1, NULL, &S);
      }
      hypre_ParCSRMatrixDestroy(AN);
   }
   else /* standard AMG or unknown approach */
   {
      if (!useSabs)
      {
         hypre_BoomerAMGCreateS(A, strong_threshold, max_row_sum,
                                num_functions, dof_func_data, &S);
      }
      else
      {
         hypre_BoomerAMGCreateSabs(A, strong_threshold, 1.0,
                                   1, NULL, &S);
      }
   }
   *S_ptr = S;

   return hypre_error_flag;
}
/**** Do the appropriate coarsening ****/
HYPRE_Int hypre_Coarsen_Options(hypre_ParCSRMatrix   *S,
		                hypre_ParCSRMatrix   *A,
				HYPRE_Int             level,
				HYPRE_Int             debug_flag,
				HYPRE_Int             coarsen_type,
				HYPRE_Int             measure_type,
				HYPRE_Int             coarsen_cut_factor,
				HYPRE_Int             agg_num_levels,
				HYPRE_Int             num_paths,
				HYPRE_Int             local_num_vars,
				hypre_IntArray       *dof_func,
                                HYPRE_BigInt         *coarse_pnts_global,
				hypre_IntArray      **CF2_marker_ptr,
				hypre_IntArray      **CF_marker_ptr)
{
   hypre_IntArray *CF_marker = NULL;
   hypre_IntArray *CF2_marker = NULL;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   if (coarsen_type == 6)
      hypre_BoomerAMGCoarsenFalgout(S, A, measure_type,
                                    coarsen_cut_factor, debug_flag, &CF_marker);
   else if (coarsen_type == 7)
      hypre_BoomerAMGCoarsen(S, A, 2, debug_flag, &CF_marker);
   else if (coarsen_type == 8)
      hypre_BoomerAMGCoarsenPMIS(S, A, 0, debug_flag, &CF_marker);
   else if (coarsen_type == 9)
      hypre_BoomerAMGCoarsenPMIS(S, A, 2, debug_flag, &CF_marker);
   else if (coarsen_type == 10)
      hypre_BoomerAMGCoarsenHMIS(S, A, measure_type,
                                 coarsen_cut_factor, debug_flag, &CF_marker);
   else if (coarsen_type)
   {
      hypre_BoomerAMGCoarsenRuge(S, A, measure_type, coarsen_type,
                                 coarsen_cut_factor, debug_flag, &CF_marker);
   }
   else
   {
      hypre_BoomerAMGCoarsen(S, A, 0, debug_flag, &CF_marker);
   }
   *CF_marker_ptr = CF_marker;

   if (level < agg_num_levels)
   {
      hypre_ParCSRMatrix *S2 = NULL;
      hypre_IntArray     *coarse_dof_func = NULL;

      hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                 1, dof_func, CF_marker,
                                 &coarse_dof_func, coarse_pnts_global);
      hypre_BoomerAMGCreate2ndS(S, hypre_IntArrayData(CF_marker), num_paths,
                                coarse_pnts_global, &S2);
      if (coarsen_type == 10)
      {
         hypre_BoomerAMGCoarsenHMIS(S2, S2, measure_type + 3, coarsen_cut_factor,
                                    debug_flag, &CF2_marker);
      }
      else if (coarsen_type == 8)
      {
         hypre_BoomerAMGCoarsenPMIS(S2, S2, 3, debug_flag, &CF2_marker);
      }
      else if (coarsen_type == 9)
      {
         hypre_BoomerAMGCoarsenPMIS(S2, S2, 4, debug_flag, &CF2_marker);
      }
      else if (coarsen_type == 6)
      {
         hypre_BoomerAMGCoarsenFalgout(S2, S2, measure_type, coarsen_cut_factor,
                                       debug_flag, &CF2_marker);
      }
      else if (coarsen_type == 7)
      {
         hypre_BoomerAMGCoarsen(S2, S2, 2, debug_flag, &CF2_marker);
      }
      else if (coarsen_type)
      {
         hypre_BoomerAMGCoarsenRuge(S2, S2, measure_type, coarsen_type,
                                    coarsen_cut_factor, debug_flag, &CF2_marker);
      }
      else
      {
         hypre_BoomerAMGCoarsen(S2, S2, 0, debug_flag, &CF2_marker);
      }

      hypre_ParCSRMatrixDestroy(S2);
      *CF2_marker_ptr = CF2_marker;
   }
   return hypre_error_flag;
}

/*** Interpolation options ***/

HYPRE_Int hypre_Interp_Options(hypre_ParCSRMatrix  *A,
		               hypre_ParCSRMatrix  *S,
		               hypre_IntArray      *CF_marker,
			       HYPRE_BigInt        *coarse_pnts_global,
			       HYPRE_Int           *dof_func_data,
			       HYPRE_Int            interp_type,
			       HYPRE_Int            num_functions,
			       HYPRE_Int            debug_flag,
			       HYPRE_Int            P_max_elmts,
			       HYPRE_Real           trunc_factor,
			       HYPRE_Int            sep_weight,
			       hypre_ParCSRMatrix **P_ptr)

{
   hypre_ParCSRMatrix *P = NULL;
   HYPRE_Int *CF_marker_data = hypre_IntArrayData(CF_marker);
   if (interp_type == 4)
   {
      hypre_BoomerAMGBuildMultipass(A, CF_marker_data,
                                    S, coarse_pnts_global, num_functions, dof_func_data,
                                    debug_flag, trunc_factor, P_max_elmts, sep_weight, &P);
   }
   else if (interp_type == 2)
   {
      hypre_BoomerAMGBuildInterpHE(A, CF_marker_data,
                                   S, coarse_pnts_global, num_functions, dof_func_data,
                                   debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 3 || interp_type == 15)
   {
      hypre_BoomerAMGBuildDirInterp(A, CF_marker_data,
                                    S, coarse_pnts_global, num_functions, dof_func_data,
                                    debug_flag, trunc_factor, P_max_elmts,
                                    interp_type, &P);
   }
   else if (interp_type == 6) /*Extended+i classical interpolation */
   {
      hypre_BoomerAMGBuildExtPIInterp(A, CF_marker_data,
                                      S, coarse_pnts_global, num_functions, dof_func_data,
                                      debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 14) /*Extended classical interpolation */
   {
      hypre_BoomerAMGBuildExtInterp(A, CF_marker_data,
                                    S, coarse_pnts_global, num_functions, dof_func_data,
                                    debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 16) /*Extended classical MM interpolation */
   {
      hypre_BoomerAMGBuildModExtInterp(A, CF_marker_data, S, coarse_pnts_global,
                                       num_functions, dof_func_data,
                                       debug_flag,
                                       trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 17) /*Extended+i MM interpolation */
   {
      hypre_BoomerAMGBuildModExtPIInterp(A, CF_marker_data, S, coarse_pnts_global,
                                         num_functions, dof_func_data,
                                         debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 18) /*Extended+e MM interpolation */
   {
      hypre_BoomerAMGBuildModExtPEInterp(A, CF_marker_data, S, coarse_pnts_global,
                                         num_functions, dof_func_data,
                                         debug_flag, trunc_factor, P_max_elmts, &P);
   }

   else if (interp_type == 7) /*Extended+i (if no common C) interpolation */
   {
      hypre_BoomerAMGBuildExtPICCInterp(A, CF_marker_data,
                                        S, coarse_pnts_global, num_functions, dof_func_data,
                                        debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 12) /*FF interpolation */
   {
      hypre_BoomerAMGBuildFFInterp(A, CF_marker_data,
                                   S, coarse_pnts_global, num_functions, dof_func_data,
                                   debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 13) /*FF1 interpolation */
   {
      hypre_BoomerAMGBuildFF1Interp(A, CF_marker_data,
                                    S, coarse_pnts_global, num_functions, dof_func_data,
                                    debug_flag, trunc_factor, P_max_elmts, &P);
   }
   else if (interp_type == 8) /*Standard interpolation */
   {
      hypre_BoomerAMGBuildStdInterp(A, CF_marker_data,
                                    S, coarse_pnts_global, num_functions, dof_func_data,
                                    debug_flag, trunc_factor, P_max_elmts, sep_weight, &P);
   }
   else
   {
      hypre_BoomerAMGBuildInterp(A, CF_marker_data,
                                 S, coarse_pnts_global, num_functions, dof_func_data,
                                 debug_flag, trunc_factor, P_max_elmts, &P);
   }

   return hypre_error_flag;
}

/******* Interpolation options for Aggressive Coarsening ***/

HYPRE_Int hypre_MPassInterp_Options(hypre_ParCSRMatrix  *A,
   		                    hypre_ParCSRMatrix  *S,
   		                    hypre_IntArray      *CF_marker,
   			            hypre_IntArray      *dof_func,
   			            HYPRE_BigInt        *coarse_pnts_global,
   			            HYPRE_Int            agg_interp_type,
   			            HYPRE_Int            num_functions,
   			            HYPRE_Int            debug_flag,
   			            HYPRE_Int            agg_P_max_elmts,
   			            HYPRE_Real           agg_trunc_factor,
   			            HYPRE_Int            sep_weight,
   			            hypre_ParCSRMatrix **P_ptr)
{
   hypre_ParCSRMatrix *P = NULL;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   hypre_IntArray *coarse_dof_func = NULL;
   HYPRE_Int *CF_marker_data = hypre_IntArrayData(CF_marker);
   HYPRE_Int *dof_func_data = hypre_IntArrayData(dof_func);
   HYPRE_Int local_num_vars = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   
   hypre_BoomerAMGCoarseParms(comm, local_num_vars, num_functions, dof_func,
                              CF_marker, &coarse_dof_func, coarse_pnts_global);
   if (agg_interp_type == 4)
   {
      hypre_BoomerAMGBuildMultipass(A, CF_marker_data, S, coarse_pnts_global,
                                    num_functions, dof_func_data, debug_flag,
                                    agg_trunc_factor, agg_P_max_elmts, sep_weight,
                                    &P);
   }
   else if (agg_interp_type == 8)
   {
      hypre_BoomerAMGBuildModMultipass(A, CF_marker_data, S, coarse_pnts_global,
                                       agg_trunc_factor, agg_P_max_elmts, 8,
                                       num_functions, dof_func_data, &P);
   }
   else if (agg_interp_type == 9)
   {
      hypre_BoomerAMGBuildModMultipass(A, CF_marker_data, S, coarse_pnts_global,
                                       agg_trunc_factor, agg_P_max_elmts, 9,
                                       num_functions, dof_func_data, &P);
   }
   *P_ptr = P;
   return hypre_error_flag;
}

HYPRE_Int hypre_StageOneInterp_Options(hypre_ParCSRMatrix  *A,
   		                       hypre_ParCSRMatrix  *S,
   		                       hypre_IntArray      *CF_marker,
   			               HYPRE_BigInt        *coarse_pnts_global1,
   			               HYPRE_Int           *dof_func_data,
   			               HYPRE_Int            agg_interp_type,
   			               HYPRE_Int            num_functions,
   			               HYPRE_Int            debug_flag,
   			               HYPRE_Int            agg_P12_max_elmts,
   			               HYPRE_Real           agg_P12_trunc_factor,
   			               hypre_ParCSRMatrix **P1_ptr)
{
   hypre_ParCSRMatrix *P1 = NULL;
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int *CF_marker_data = hypre_IntArrayData(CF_marker);

   if (agg_interp_type == 1)
   {
      hypre_BoomerAMGBuildExtPIInterp(A, CF_marker_data, S, coarse_pnts_global1,
                                      num_functions, dof_func_data, debug_flag,
                                      agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
   }
   else if (agg_interp_type == 2)
   {
      hypre_BoomerAMGBuildStdInterp(A, CF_marker_data, S, coarse_pnts_global1,
                                    num_functions, dof_func_data, debug_flag,
                                    agg_P12_trunc_factor, agg_P12_max_elmts, 0, &P1);
   }
   else if (agg_interp_type == 3)
   {
      hypre_BoomerAMGBuildExtInterp(A, CF_marker_data, S, coarse_pnts_global1,
                                    num_functions, dof_func_data, debug_flag,
                                    agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
   }
   else if (agg_interp_type == 5)
   {
      hypre_BoomerAMGBuildModExtInterp(A, CF_marker_data, S, coarse_pnts_global1,
                                       num_functions, dof_func_data,
                                       debug_flag,
                                       agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
   }
   else if (agg_interp_type == 6)
   {
      hypre_BoomerAMGBuildModExtPIInterp(A, CF_marker_data, S, coarse_pnts_global1,
                                         num_functions, dof_func_data,
                                         debug_flag,
                                         agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
   }
   else if (agg_interp_type == 7)
   {
      hypre_BoomerAMGBuildModExtPEInterp(A, CF_marker_data, S, coarse_pnts_global1,
                                         num_functions, dof_func_data,
                                         debug_flag,
                                         agg_P12_trunc_factor, agg_P12_max_elmts, &P1);
   }
   *P1_ptr = P1;
   return hypre_error_flag;
}   
      /*hypre_BoomerAMGCorrectCFMarker2 (CF_marker, CFN_marker);
      hypre_IntArrayDestroy(CFN_marker);
      hypre_BoomerAMGCoarseParms(comm, local_num_vars,
                                 num_functions, dof_func_array, CF_marker,
                                 &coarse_dof_func, coarse_pnts_global);*/

HYPRE_Int hypre_StageTwoInterp_Options(hypre_ParCSRMatrix  *A,
   		                       hypre_ParCSRMatrix  *P1,
   		                       hypre_ParCSRMatrix  *S,
   		                       hypre_IntArray      *CF_marker,
   			               HYPRE_BigInt        *coarse_pnts_global,
   			               HYPRE_BigInt        *coarse_pnts_global1,
   			               HYPRE_Int           *dof_func_data,
   			               HYPRE_Int            agg_interp_type,
   			               HYPRE_Int            num_functions,
   			               HYPRE_Int            debug_flag,
   			               HYPRE_Int            sep_weight,
   			               HYPRE_Int            agg_P_max_elmts,
   			               HYPRE_Int            agg_P12_max_elmts,
   			               HYPRE_Real           agg_trunc_factor,
   			               HYPRE_Real           agg_P12_trunc_factor,
   			               hypre_ParCSRMatrix **P_ptr)
{
   hypre_ParCSRMatrix *P = NULL;
   hypre_ParCSRMatrix *P2 = NULL;
   HYPRE_Int *CF_marker_data = hypre_IntArrayData(CF_marker);

   if (agg_interp_type == 1 || agg_interp_type == 6 )
   {
      hypre_BoomerAMGBuildPartialExtPIInterp(A, CF_marker_data, S, coarse_pnts_global,
                                             coarse_pnts_global1, num_functions,
                                             dof_func_data, debug_flag, agg_P12_trunc_factor,
                                             agg_P12_max_elmts, &P2);
   }
   else if (agg_interp_type == 2)
   {
      hypre_BoomerAMGBuildPartialStdInterp(A, CF_marker_data, S, coarse_pnts_global,
                                           coarse_pnts_global1, num_functions,
                                           dof_func_data, debug_flag, agg_P12_trunc_factor,
                                           agg_P12_max_elmts, 0, &P2);
   }
   else if (agg_interp_type == 3)
   {
      hypre_BoomerAMGBuildPartialExtInterp(A, CF_marker_data, S, coarse_pnts_global,
                                           coarse_pnts_global1, num_functions,
                                           dof_func_data, debug_flag, agg_P12_trunc_factor,
                                           agg_P12_max_elmts, &P2);
   }
   else if (agg_interp_type == 5)
   {
      hypre_BoomerAMGBuildModPartialExtInterp(A, CF_marker_data, S, coarse_pnts_global,
                                              coarse_pnts_global1,
                                              num_functions, dof_func_data,
                                              debug_flag,
                                              agg_P12_trunc_factor, agg_P12_max_elmts, &P2);
   }
   else if (agg_interp_type == 7)
   {
      hypre_BoomerAMGBuildModPartialExtPEInterp(A, CF_marker_data, S, coarse_pnts_global,
                                                coarse_pnts_global1,
                                                num_functions, dof_func_data,
                                                debug_flag,
                                                agg_P12_trunc_factor, agg_P12_max_elmts, &P2);
   }

   P = hypre_ParMatmul(P1, P2);
   hypre_ParCSRMatrixDestroy(P1);
   hypre_ParCSRMatrixDestroy(P2);

   hypre_BoomerAMGInterpTruncation(P, agg_trunc_factor, agg_P_max_elmts);

   if (agg_trunc_factor != 0.0 || agg_P_max_elmts > 0 ||
       agg_P12_trunc_factor != 0.0 || agg_P12_max_elmts > 0)
   {
      hypre_ParCSRMatrixCompressOffdMap(P);
   }

   hypre_MatvecCommPkgCreate(P);
   *P_ptr = P;

   return hypre_error_flag;
}

/*  adding L1 norms per level */
HYPRE_Int hypre_Level_L1Norms(hypre_ParCSRMatrix *A,
		              hypre_IntArray     *CF_marker,
			      HYPRE_Int          *grid_relax_type,
			      HYPRE_Int           level,
			      HYPRE_Int           num_levels,
			      HYPRE_Int           relax_order,
			      hypre_Vector       **l1_norm_ptr)
{
   HYPRE_Real *l1_norm_data = NULL;
   hypre_Vector *l1_norm = NULL;

   l1_norm = hypre_SeqVectorCreate(num_levels);

   if ( grid_relax_type[1] == 8  || grid_relax_type[1] == 89 ||
        grid_relax_type[1] == 13 || grid_relax_type[1] == 14 ||
        grid_relax_type[2] == 8  || grid_relax_type[2] == 89 ||
        grid_relax_type[2] == 13 || grid_relax_type[2] == 14)
   {
      if (relax_order)
      {
         hypre_ParCSRComputeL1Norms(A, 4, hypre_IntArrayData(CF_marker), &l1_norm_data);
      }
      else
      {
         hypre_ParCSRComputeL1Norms(A, 4, NULL, &l1_norm_data);
      }
   }
   else if ( grid_relax_type[3] == 8  || grid_relax_type[3] == 89 ||
             grid_relax_type[3] == 13 || grid_relax_type[3] == 14)
   {
      hypre_ParCSRComputeL1Norms(A, 4, NULL, &l1_norm_data);
   }


   if (grid_relax_type[1] == 30 || grid_relax_type[2] == 30)
   {
      if (relax_order)
      {
         hypre_ParCSRComputeL1Norms(A, 3, hypre_IntArrayData(CF_marker), &l1_norm_data);
      }
      else
      {
         hypre_ParCSRComputeL1Norms(A, 3, NULL, &l1_norm_data);
      }
   }
   else if (grid_relax_type[3] == 30)
   {
      hypre_ParCSRComputeL1Norms(A, 3, NULL, &l1_norm_data);
   }

   if (grid_relax_type[1] == 88 || grid_relax_type[2] == 88 )
   {
      if (relax_order)
      {
         hypre_ParCSRComputeL1Norms(A, 6, hypre_IntArrayData(CF_marker), &l1_norm_data);
      }
      else
      {
         hypre_ParCSRComputeL1Norms(A, 6, NULL, &l1_norm_data);
      }
   }
   else if (grid_relax_type[3] == 88)
   {
      hypre_ParCSRComputeL1Norms(A, 6, NULL, &l1_norm_data);
   }

   if (grid_relax_type[1] == 18 || grid_relax_type[2] == 18)
   {
      if (relax_order)
      {
         hypre_ParCSRComputeL1Norms(A, 1, hypre_IntArrayData(CF_marker), &l1_norm_data);
      }
      else
      {
         hypre_ParCSRComputeL1Norms(A, 1, NULL, &l1_norm_data);
      }
   }
   else if (grid_relax_type[3] == 18)
   {
      hypre_ParCSRComputeL1Norms(A, 1, NULL, &l1_norm_data);
   }
   else if ( grid_relax_type[1]  == 7 || grid_relax_type[2] == 7   ||
        (grid_relax_type[3] == 7 && level == (num_levels - 1))    ||
        grid_relax_type[1]  == 11 || grid_relax_type[2] == 11 ||
        (grid_relax_type[3] == 11 && level == (num_levels - 1))   ||
        grid_relax_type[1]  == 12 || grid_relax_type[2] == 12 ||
        (grid_relax_type[3] == 12 && level == (num_levels - 1)) )
   {
      hypre_ParCSRComputeL1Norms(A, 5, NULL, &l1_norm_data);
   }

   hypre_VectorData(l1_norm) = l1_norm_data;
   *l1_norm_ptr = l1_norm;
   return hypre_error_flag;
}
