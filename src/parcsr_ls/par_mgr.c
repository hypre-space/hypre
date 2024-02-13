/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_mgr.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

//#ifdef HYPRE_USING_DSUPERLU
//#include "dsuperlu.h"
//#endif

/* Need to define these hypre_lapack protos here instead of including _hypre_lapack.h to avoid conflicts with
 * dsuperlu.h on some lapack functions. Alternative is to move superLU related functions to a separate file.
*/
/* dgetrf.c */
//HYPRE_Int hypre_dgetrf ( HYPRE_Int *m, HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *lda, HYPRE_Int *ipiv,
//                         HYPRE_Int *info );
/* dgetri.c */
//HYPRE_Int hypre_dgetri ( HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *lda, HYPRE_Int *ipiv,
//                         HYPRE_Real *work, HYPRE_Int *lwork, HYPRE_Int *info);

/* Create */
void *
hypre_MGRCreate(void)
{
   hypre_ParMGRData  *mgr_data;

   mgr_data = hypre_CTAlloc(hypre_ParMGRData,  1, HYPRE_MEMORY_HOST);

   /* block data */
   (mgr_data -> block_size) = 1;
   (mgr_data -> block_num_coarse_indexes) = NULL;
   (mgr_data -> point_marker_array) = NULL;
   (mgr_data -> block_cf_marker) = NULL;

   /* general data */
   (mgr_data -> max_num_coarse_levels) = 10;
   (mgr_data -> A_array) = NULL;
   (mgr_data -> B_array) = NULL;
   (mgr_data -> B_FF_array) = NULL;
#if defined(HYPRE_USING_GPU)
   (mgr_data -> P_FF_array) = NULL;
#endif
   (mgr_data -> P_array) = NULL;
   (mgr_data -> R_array) = NULL;
   (mgr_data -> RT_array) = NULL;
   (mgr_data -> RAP) = NULL;
   (mgr_data -> CF_marker_array) = NULL;
   (mgr_data -> coarse_indices_lvls) = NULL;

   (mgr_data -> A_ff_array) = NULL;
   (mgr_data -> F_fine_array) = NULL;
   (mgr_data -> U_fine_array) = NULL;
   (mgr_data -> aff_solver) = NULL;
   (mgr_data -> fine_grid_solver_setup) = NULL;
   (mgr_data -> fine_grid_solver_solve) = NULL;

   (mgr_data -> F_array) = NULL;
   (mgr_data -> U_array) = NULL;
   (mgr_data -> residual) = NULL;
   (mgr_data -> rel_res_norms) = NULL;
   (mgr_data -> Vtemp) = NULL;
   (mgr_data -> Ztemp) = NULL;
   (mgr_data -> Utemp) = NULL;
   (mgr_data -> Ftemp) = NULL;

   (mgr_data -> num_iterations) = 0;
   (mgr_data -> num_interp_sweeps) = 1;
   (mgr_data -> num_restrict_sweeps) = 1;
   (mgr_data -> trunc_factor) = 0.0;
   (mgr_data -> max_row_sum) = 0.9;
   (mgr_data -> strong_threshold) = 0.25;
   (mgr_data -> P_max_elmts) = NULL;

   (mgr_data -> coarse_grid_solver) = NULL;
   (mgr_data -> coarse_grid_solver_setup) = NULL;
   (mgr_data -> coarse_grid_solver_solve) = NULL;

   //(mgr_data -> global_smoother) = NULL;

   (mgr_data -> use_default_cgrid_solver) = 1;
   (mgr_data -> fsolver_mode) = -1; // user or hypre -prescribed F-solver
   (mgr_data -> omega) = 1.;
   (mgr_data -> max_iter) = 20;
   (mgr_data -> tol) = 1.0e-6;
   (mgr_data -> relax_type) = 0;
   (mgr_data -> Frelax_type) = NULL;
   (mgr_data -> relax_order) = 1; // not fully utilized. Only used to compute L1-norms.
   (mgr_data -> num_relax_sweeps) = NULL;
   (mgr_data -> relax_weight) = 1.0;

   (mgr_data -> interp_type) = NULL;
   (mgr_data -> restrict_type) = NULL;
   (mgr_data -> level_smooth_iters) = NULL;
   (mgr_data -> level_smooth_type) = NULL;
   (mgr_data -> level_smoother) = NULL;
   (mgr_data -> global_smooth_cycle) = 1; // Pre = 1 or Post  = 2 global smoothing

   (mgr_data -> logging) = 0;
   (mgr_data -> print_level) = 0;
   (mgr_data -> frelax_print_level) = 0;
   (mgr_data -> cg_print_level) = 0;
   (mgr_data -> data_path) = NULL;

   (mgr_data -> l1_norms) = NULL;

   (mgr_data -> reserved_coarse_size) = 0;
   (mgr_data -> reserved_coarse_indexes) = NULL;
   (mgr_data -> reserved_Cpoint_local_indexes) = NULL;

   (mgr_data -> level_diaginv) = NULL;
   (mgr_data -> frelax_diaginv) = NULL;
   //(mgr_data -> global_smooth_iters) = 1;
   //(mgr_data -> global_smooth_type) = 0;

   (mgr_data -> set_non_Cpoints_to_F) = 0;
   (mgr_data -> idx_array) = NULL;

   (mgr_data -> Frelax_method) = NULL;
   (mgr_data -> VcycleRelaxVtemp) = NULL;
   (mgr_data -> VcycleRelaxZtemp) = NULL;
   (mgr_data -> FrelaxVcycleData) = NULL;
   (mgr_data -> Frelax_num_functions) = NULL;
   (mgr_data -> max_local_lvls) = 10;

   (mgr_data -> mgr_coarse_grid_method) = NULL;

   (mgr_data -> print_coarse_system) = 0;

   (mgr_data -> set_c_points_method) = 0;
   (mgr_data -> lvl_to_keep_cpoints) = 0;
   (mgr_data -> cg_convergence_factor) = 0.0;

   (mgr_data -> block_jacobi_bsize) = 0;
   (mgr_data -> blk_size) = NULL;

   (mgr_data -> truncate_coarse_grid_threshold) = 0.0;

   (mgr_data -> GSElimData) = NULL;

   return (void *) mgr_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
HYPRE_Int
hypre_MGRDestroy( void *data )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) data;
   hypre_Solver      *aff_base;

   HYPRE_Int i;
   HYPRE_Int num_coarse_levels = (mgr_data -> num_coarse_levels);

   /* block info data */
   if ((mgr_data -> block_cf_marker))
   {
      for (i = 0; i < (mgr_data -> max_num_coarse_levels); i++)
      {
         hypre_TFree((mgr_data -> block_cf_marker)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree((mgr_data -> block_cf_marker), HYPRE_MEMORY_HOST);
   }

   hypre_TFree(mgr_data -> block_num_coarse_indexes, HYPRE_MEMORY_HOST);

   /* final residual vector */
   if ((mgr_data -> residual))
   {
      hypre_ParVectorDestroy( (mgr_data -> residual) );
      (mgr_data -> residual) = NULL;
   }

   hypre_TFree( (mgr_data -> rel_res_norms), HYPRE_MEMORY_HOST);

   /* temp vectors for solve phase */
   if ((mgr_data -> Vtemp))
   {
      hypre_ParVectorDestroy( (mgr_data -> Vtemp) );
      (mgr_data -> Vtemp) = NULL;
   }
   if ((mgr_data -> Ztemp))
   {
      hypre_ParVectorDestroy( (mgr_data -> Ztemp) );
      (mgr_data -> Ztemp) = NULL;
   }
   if ((mgr_data -> Utemp))
   {
      hypre_ParVectorDestroy( (mgr_data -> Utemp) );
      (mgr_data -> Utemp) = NULL;
   }
   if ((mgr_data -> Ftemp))
   {
      hypre_ParVectorDestroy( (mgr_data -> Ftemp) );
      (mgr_data -> Ftemp) = NULL;
   }
   /* coarse grid solver */
   if ((mgr_data -> use_default_cgrid_solver))
   {
      if ((mgr_data -> coarse_grid_solver))
      {
         hypre_BoomerAMGDestroy( (mgr_data -> coarse_grid_solver) );
      }
      (mgr_data -> coarse_grid_solver) = NULL;
   }
   /* l1_norms */
   if ((mgr_data -> l1_norms))
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         hypre_SeqVectorDestroy((mgr_data -> l1_norms)[i]);
      }
      hypre_TFree((mgr_data -> l1_norms), HYPRE_MEMORY_HOST);
   }

   /* coarse_indices_lvls */
   if ((mgr_data -> coarse_indices_lvls))
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         hypre_TFree((mgr_data -> coarse_indices_lvls)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree((mgr_data -> coarse_indices_lvls), HYPRE_MEMORY_HOST);
   }

   /* linear system and cf marker array */
   if (mgr_data -> A_array || mgr_data -> P_array ||
       mgr_data -> RT_array || mgr_data -> R_array ||
       mgr_data -> CF_marker_array)
   {
      for (i = 1; i < num_coarse_levels + 1; i++)
      {
         hypre_ParVectorDestroy((mgr_data -> F_array)[i]);
         hypre_ParVectorDestroy((mgr_data -> U_array)[i]);

         if ((mgr_data -> P_array)[i - 1])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> P_array)[i - 1]);
         }

         if ((mgr_data -> R_array)[i - 1])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> R_array)[i - 1]);
         }

         if ((mgr_data -> RT_array)[i - 1])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> RT_array)[i - 1]);
         }

         hypre_IntArrayDestroy(mgr_data -> CF_marker_array[i - 1]);
      }
      for (i = 1; i < (num_coarse_levels); i++)
      {
         if ((mgr_data -> A_array)[i])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> A_array)[i]);
         }
      }
   }

   /* Block relaxation/interpolation matrices */
   if (hypre_ParMGRDataBArray(mgr_data))
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         hypre_ParCSRMatrixDestroy(hypre_ParMGRDataB(mgr_data, i));
      }
   }

   if (hypre_ParMGRDataBFFArray(mgr_data))
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         hypre_ParCSRMatrixDestroy(hypre_ParMGRDataBFF(mgr_data, i));
      }
   }

#if defined(HYPRE_USING_GPU)
   if (mgr_data -> P_FF_array)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         if ((mgr_data -> P_array)[i])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> P_FF_array)[i]);
         }
      }
      //hypre_TFree(P_FF_array, hypre_HandleMemoryLocation(hypre_handle()));
      hypre_TFree((mgr_data -> P_FF_array), HYPRE_MEMORY_HOST);
      (mgr_data -> P_FF_array) = NULL;
   }
#endif

   /* AMG for Frelax */
   if (mgr_data -> A_ff_array || mgr_data -> F_fine_array || mgr_data -> U_fine_array)
   {
      for (i = 1; i < num_coarse_levels + 1; i++)
      {
         if (mgr_data -> F_fine_array[i])
         {
            hypre_ParVectorDestroy((mgr_data -> F_fine_array)[i]);
         }
         if (mgr_data -> U_fine_array[i])
         {
            hypre_ParVectorDestroy((mgr_data -> U_fine_array)[i]);
         }
      }
      for (i = 1; i < (num_coarse_levels); i++)
      {
         if ((mgr_data -> A_ff_array)[i])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> A_ff_array)[i]);
         }
      }
      if (mgr_data -> fsolver_mode != 0)
      {
         if ((mgr_data -> A_ff_array)[0])
         {
            hypre_ParCSRMatrixDestroy((mgr_data -> A_ff_array)[0]);
         }
      }
      hypre_TFree(mgr_data -> F_fine_array, HYPRE_MEMORY_HOST);
      (mgr_data -> F_fine_array) = NULL;
      hypre_TFree(mgr_data -> U_fine_array, HYPRE_MEMORY_HOST);
      (mgr_data -> U_fine_array) = NULL;
      hypre_TFree(mgr_data -> A_ff_array, HYPRE_MEMORY_HOST);
      (mgr_data -> A_ff_array) = NULL;
   }

   if (mgr_data -> aff_solver)
   {
      for (i = 1; i < (num_coarse_levels); i++)
      {
         if ((mgr_data -> aff_solver)[i])
         {
            aff_base = (hypre_Solver*) (mgr_data -> aff_solver)[i];
            hypre_SolverDestroy(aff_base)((HYPRE_Solver) (aff_base));
         }
      }
      if (mgr_data -> fsolver_mode == 2)
      {
         hypre_BoomerAMGDestroy((mgr_data -> aff_solver)[0]);
      }
      hypre_TFree(mgr_data -> aff_solver, HYPRE_MEMORY_HOST);
      (mgr_data -> aff_solver) = NULL;
   }

   if (mgr_data -> level_diaginv)
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         hypre_TFree((mgr_data -> level_diaginv)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(mgr_data -> level_diaginv, HYPRE_MEMORY_HOST);
   }

   if (mgr_data -> frelax_diaginv)
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         hypre_TFree((mgr_data -> frelax_diaginv)[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(mgr_data -> frelax_diaginv, HYPRE_MEMORY_HOST);
   }
   hypre_TFree((mgr_data -> F_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> U_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> A_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> B_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> B_FF_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> P_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> R_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> RT_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> CF_marker_array), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> restrict_type), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> interp_type), HYPRE_MEMORY_HOST);
   hypre_TFree((mgr_data -> P_max_elmts), HYPRE_MEMORY_HOST);
   /* Frelax_type */
   hypre_TFree(mgr_data -> Frelax_type, HYPRE_MEMORY_HOST);
   /* Frelax_method */
   hypre_TFree(mgr_data -> Frelax_method, HYPRE_MEMORY_HOST);
   /* Frelax_num_functions */
   hypre_TFree(mgr_data -> Frelax_num_functions, HYPRE_MEMORY_HOST);

   /* data for V-cycle F-relaxation */
   if ((mgr_data -> VcycleRelaxVtemp))
   {
      hypre_ParVectorDestroy( (mgr_data -> VcycleRelaxVtemp) );
      (mgr_data -> VcycleRelaxVtemp) = NULL;
   }
   if ((mgr_data -> VcycleRelaxZtemp))
   {
      hypre_ParVectorDestroy( (mgr_data -> VcycleRelaxZtemp) );
      (mgr_data -> VcycleRelaxZtemp) = NULL;
   }
   if (mgr_data -> FrelaxVcycleData)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         hypre_MGRDestroyFrelaxVcycleData((mgr_data -> FrelaxVcycleData)[i]);
      }
      hypre_TFree(mgr_data -> FrelaxVcycleData, HYPRE_MEMORY_HOST);
   }
   /* data for reserved coarse nodes */
   hypre_TFree(mgr_data -> reserved_coarse_indexes, HYPRE_MEMORY_HOST);
   /* index array for setting Cpoints by global block */
   if ((mgr_data -> set_c_points_method) == 1)
   {
      hypre_TFree(mgr_data -> idx_array, HYPRE_MEMORY_HOST);
   }
   /* array for setting option to use non-Galerkin coarse grid */
   hypre_TFree(mgr_data -> mgr_coarse_grid_method, HYPRE_MEMORY_HOST);
   /* coarse level matrix - RAP */
   if ((mgr_data -> RAP))
   {
      hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
   }

   if ((mgr_data -> level_smoother) != NULL)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         if ((mgr_data -> level_smooth_iters)[i] > 0)
         {
            if ((mgr_data -> level_smooth_type)[i] == 8)
            {
               HYPRE_EuclidDestroy((mgr_data -> level_smoother)[i]);
            }
            else if ((mgr_data -> level_smooth_type)[i] == 16)
            {
               HYPRE_ILUDestroy((mgr_data -> level_smoother)[i]);
            }
         }
      }
      hypre_TFree(mgr_data -> level_smoother, HYPRE_MEMORY_HOST);
   }

   /* free level data */
   hypre_TFree(mgr_data -> blk_size, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_data -> level_smooth_type, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_data -> level_smooth_iters, HYPRE_MEMORY_HOST);
   hypre_TFree(mgr_data -> num_relax_sweeps, HYPRE_MEMORY_HOST);

   if (mgr_data -> GSElimData)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         if ((mgr_data -> GSElimData)[i])
         {
            hypre_MGRDestroyGSElimData((mgr_data -> GSElimData)[i]);
            (mgr_data -> GSElimData)[i] = NULL;
         }
      }
      hypre_TFree(mgr_data -> GSElimData, HYPRE_MEMORY_HOST);
   }

   /* Free the data path filename */
   hypre_TFree(mgr_data -> data_path, HYPRE_MEMORY_HOST);

   /* mgr data */
   hypre_TFree(mgr_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRCreateGSElimData
 *
 * Create data for Gaussian Elimination for F-relaxation.
 *--------------------------------------------------------------------------*/

void *
hypre_MGRCreateGSElimData( void )
{
   hypre_ParAMGData  *gsdata = hypre_CTAlloc(hypre_ParAMGData,  1, HYPRE_MEMORY_HOST);

   hypre_ParAMGDataGSSetup(gsdata)          = 0;
   hypre_ParAMGDataGEMemoryLocation(gsdata) = HYPRE_MEMORY_UNDEFINED;
   hypre_ParAMGDataNewComm(gsdata)          = hypre_MPI_COMM_NULL;
   hypre_ParAMGDataCommInfo(gsdata)         = NULL;
   hypre_ParAMGDataAMat(gsdata)             = NULL;
   hypre_ParAMGDataAWork(gsdata)            = NULL;
   hypre_ParAMGDataAPiv(gsdata)             = NULL;
   hypre_ParAMGDataBVec(gsdata)             = NULL;
   hypre_ParAMGDataUVec(gsdata)             = NULL;

   return (void *) gsdata;
}

/*--------------------------------------------------------------------------
 * hypre_MGRDestroyGSElimData
 *
 * Destroy data for Gaussian Elimination for F-relaxation.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRDestroyGSElimData( void *data )
{
   hypre_ParAMGData  *gsdata   = (hypre_ParAMGData*) data;
   MPI_Comm           new_comm = hypre_ParAMGDataNewComm(gsdata);

#if defined(HYPRE_USING_MAGMA)
   hypre_TFree(hypre_ParAMGDataAPiv(gsdata),  HYPRE_MEMORY_HOST);
#else
   hypre_TFree(hypre_ParAMGDataAPiv(gsdata),  hypre_ParAMGDataGEMemoryLocation(gsdata));
#endif
   hypre_TFree(hypre_ParAMGDataAMat(gsdata),  hypre_ParAMGDataGEMemoryLocation(gsdata));
   hypre_TFree(hypre_ParAMGDataAWork(gsdata), hypre_ParAMGDataGEMemoryLocation(gsdata));
   hypre_TFree(hypre_ParAMGDataBVec(gsdata),  hypre_ParAMGDataGEMemoryLocation(gsdata));
   hypre_TFree(hypre_ParAMGDataUVec(gsdata),  hypre_ParAMGDataGEMemoryLocation(gsdata));
   hypre_TFree(hypre_ParAMGDataCommInfo(gsdata), HYPRE_MEMORY_HOST);

   if (new_comm != hypre_MPI_COMM_NULL)
   {
      hypre_MPI_Comm_free(&new_comm);
   }

   hypre_TFree(gsdata, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* Create data for V-cycle F-relaxtion */
void *
hypre_MGRCreateFrelaxVcycleData( void )
{
   hypre_ParAMGData  *vdata = hypre_CTAlloc(hypre_ParAMGData,  1, HYPRE_MEMORY_HOST);

   hypre_ParAMGDataAArray(vdata) = NULL;
   hypre_ParAMGDataPArray(vdata) = NULL;
   hypre_ParAMGDataFArray(vdata) = NULL;
   hypre_ParAMGDataCFMarkerArray(vdata) = NULL;
   hypre_ParAMGDataVtemp(vdata)  = NULL;
   //   hypre_ParAMGDataAMat(vdata)  = NULL;
   //   hypre_ParAMGDataBVec(vdata)  = NULL;
   hypre_ParAMGDataZtemp(vdata)  = NULL;
   //   hypre_ParAMGDataCommInfo(vdata) = NULL;
   hypre_ParAMGDataUArray(vdata) = NULL;
   hypre_ParAMGDataNewComm(vdata) = hypre_MPI_COMM_NULL;
   hypre_ParAMGDataNumLevels(vdata) = 0;
   hypre_ParAMGDataMaxLevels(vdata) = 10;
   hypre_ParAMGDataNumFunctions(vdata) = 1;
   hypre_ParAMGDataSCommPkgSwitch(vdata) = 1.0;
   hypre_ParAMGDataRelaxOrder(vdata) = 1;
   hypre_ParAMGDataMaxCoarseSize(vdata) = 9;
   hypre_ParAMGDataMinCoarseSize(vdata) = 0;
   hypre_ParAMGDataUserCoarseRelaxType(vdata) = 9;

   /* Gaussian Elim data */
   hypre_ParAMGDataGSSetup(vdata) = 0;
   hypre_ParAMGDataAMat(vdata) = NULL;
   hypre_ParAMGDataAWork(vdata) = NULL;
   hypre_ParAMGDataBVec(vdata) = NULL;
   hypre_ParAMGDataCommInfo(vdata) = NULL;

   return (void *) vdata;
}

/* Destroy data for V-cycle F-relaxation */
HYPRE_Int
hypre_MGRDestroyFrelaxVcycleData( void *data )
{
   hypre_ParAMGData * vdata = (hypre_ParAMGData*) data;
   HYPRE_Int i;
   HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(vdata);
   MPI_Comm new_comm = hypre_ParAMGDataNewComm(vdata);

   hypre_TFree(hypre_ParAMGDataDofFuncArray(vdata)[0], HYPRE_MEMORY_HOST);
   for (i = 1; i < num_levels + 1; i++)
   {
      if (hypre_ParAMGDataAArray(vdata)[i])
      {
         hypre_ParCSRMatrixDestroy(hypre_ParAMGDataAArray(vdata)[i]);
      }

      if (hypre_ParAMGDataPArray(vdata)[i - 1])
      {
         hypre_ParCSRMatrixDestroy(hypre_ParAMGDataPArray(vdata)[i - 1]);
      }

      hypre_IntArrayDestroy(hypre_ParAMGDataCFMarkerArray(vdata)[i - 1]);
      hypre_ParVectorDestroy(hypre_ParAMGDataFArray(vdata)[i]);
      hypre_ParVectorDestroy(hypre_ParAMGDataUArray(vdata)[i]);
      hypre_TFree(hypre_ParAMGDataDofFuncArray(vdata)[i], HYPRE_MEMORY_HOST);
   }

   if (num_levels < 1)
   {
      hypre_IntArrayDestroy(hypre_ParAMGDataCFMarkerArray(vdata)[0]);
   }

   /* Points to VcycleRelaxVtemp of mgr_data, which is already destroyed */
   //hypre_ParVectorDestroy(hypre_ParAMGDataVtemp(vdata));
   hypre_TFree(hypre_ParAMGDataFArray(vdata), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParAMGDataUArray(vdata), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParAMGDataAArray(vdata), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParAMGDataPArray(vdata), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParAMGDataCFMarkerArray(vdata), HYPRE_MEMORY_HOST);
   //hypre_TFree(hypre_ParAMGDataGridRelaxType(vdata), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParAMGDataDofFuncArray(vdata), HYPRE_MEMORY_HOST);

   /* Points to VcycleRelaxZtemp of mgr_data, which is already destroyed */
   /*
     if (hypre_ParAMGDataZtemp(vdata))
         hypre_ParVectorDestroy(hypre_ParAMGDataZtemp(vdata));
   */

#if defined(HYPRE_USING_MAGMA)
   hypre_TFree(hypre_ParAMGDataAPiv(vdata),  HYPRE_MEMORY_HOST);
#else
   hypre_TFree(hypre_ParAMGDataAPiv(vdata),  hypre_ParAMGDataGEMemoryLocation(vdata));
#endif
   hypre_TFree(hypre_ParAMGDataAMat(vdata),  hypre_ParAMGDataGEMemoryLocation(vdata));
   hypre_TFree(hypre_ParAMGDataAWork(vdata), hypre_ParAMGDataGEMemoryLocation(vdata));
   hypre_TFree(hypre_ParAMGDataBVec(vdata),  hypre_ParAMGDataGEMemoryLocation(vdata));
   hypre_TFree(hypre_ParAMGDataUVec(vdata),  hypre_ParAMGDataGEMemoryLocation(vdata));
   hypre_TFree(hypre_ParAMGDataCommInfo(vdata), HYPRE_MEMORY_HOST);

   if (new_comm != hypre_MPI_COMM_NULL)
   {
      hypre_MPI_Comm_free (&new_comm);
   }
   hypre_TFree(vdata, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* Set C-point variables for each reduction level */
/* Currently not implemented */
HYPRE_Int
hypre_MGRSetReductionLevelCpoints( void      *mgr_vdata,
                                   HYPRE_Int  nlevels,
                                   HYPRE_Int *num_coarse_points,
                                   HYPRE_Int  **level_coarse_indexes)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> num_coarse_levels) = nlevels;
   (mgr_data -> num_coarse_per_level) = num_coarse_points;
   (mgr_data -> level_coarse_indexes) = level_coarse_indexes;
   return hypre_error_flag;
}

/* Initialize some data */
/* Set whether non-coarse points on each level should be explicitly tagged as F-points */
HYPRE_Int
hypre_MGRSetNonCpointsToFpoints( void      *mgr_vdata, HYPRE_Int nonCptToFptFlag)
{
   hypre_ParMGRData *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> set_non_Cpoints_to_F) = nonCptToFptFlag;

   return hypre_error_flag;
}

/* Set whether the reserved C points are reduced before the coarse grid solve */
HYPRE_Int
hypre_MGRSetReservedCpointsLevelToKeep(void *mgr_vdata, HYPRE_Int level)
{
   hypre_ParMGRData *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> lvl_to_keep_cpoints) = level;

   return hypre_error_flag;
}

/* Set Cpoints by contiguous blocks, i.e. p1, p2, ..., pn, s1, s2, ..., sn, ... */
HYPRE_Int
hypre_MGRSetCpointsByContiguousBlock( void  *mgr_vdata,
                                      HYPRE_Int  block_size,
                                      HYPRE_Int  max_num_levels,
                                      HYPRE_BigInt  *begin_idx_array,
                                      HYPRE_Int  *block_num_coarse_points,
                                      HYPRE_Int  **block_coarse_indexes)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   if ((mgr_data -> idx_array) != NULL)
   {
      hypre_TFree(mgr_data -> idx_array, HYPRE_MEMORY_HOST);
      (mgr_data -> idx_array) = NULL;
   }
   HYPRE_BigInt *index_array = hypre_CTAlloc(HYPRE_BigInt, block_size, HYPRE_MEMORY_HOST);
   if (begin_idx_array != NULL)
   {
      for (i = 0; i < block_size; i++)
      {
         index_array[i] = *(begin_idx_array + i);
      }
   }
   hypre_MGRSetCpointsByBlock(mgr_data, block_size, max_num_levels, block_num_coarse_points,
                              block_coarse_indexes);
   (mgr_data -> idx_array) = index_array;
   (mgr_data -> set_c_points_method) = 1;
   return hypre_error_flag;
}

/* Initialize/ set local block data information */
HYPRE_Int
hypre_MGRSetCpointsByBlock( void      *mgr_vdata,
                            HYPRE_Int  block_size,
                            HYPRE_Int  max_num_levels,
                            HYPRE_Int  *block_num_coarse_points,
                            HYPRE_Int  **block_coarse_indexes)
{
   HYPRE_Int  i, j;
   HYPRE_Int  **block_cf_marker = NULL;
   HYPRE_Int *block_num_coarse_indexes = NULL;

   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   /* free block cf_marker data if not previously destroyed */
   if ((mgr_data -> block_cf_marker) != NULL)
   {
      for (i = 0; i < (mgr_data -> max_num_coarse_levels); i++)
      {
         if ((mgr_data -> block_cf_marker)[i])
         {
            hypre_TFree((mgr_data -> block_cf_marker)[i], HYPRE_MEMORY_HOST);
            (mgr_data -> block_cf_marker)[i] = NULL;
         }
      }
      hypre_TFree(mgr_data -> block_cf_marker, HYPRE_MEMORY_HOST);
      (mgr_data -> block_cf_marker) = NULL;
   }
   if ((mgr_data -> block_num_coarse_indexes))
   {
      hypre_TFree((mgr_data -> block_num_coarse_indexes), HYPRE_MEMORY_HOST);
      (mgr_data -> block_num_coarse_indexes) = NULL;
   }

   /* store block cf_marker */
   block_cf_marker = hypre_CTAlloc(HYPRE_Int *, max_num_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_levels; i++)
   {
      block_cf_marker[i] = hypre_CTAlloc(HYPRE_Int, block_size, HYPRE_MEMORY_HOST);
      memset(block_cf_marker[i], FMRK, block_size * sizeof(HYPRE_Int));
   }
   for (i = 0; i < max_num_levels; i++)
   {
      for (j = 0; j < block_num_coarse_points[i]; j++)
      {
         (block_cf_marker[i])[block_coarse_indexes[i][j]] = CMRK;
      }
   }

   /* store block_num_coarse_points */
   if (max_num_levels > 0)
   {
      block_num_coarse_indexes = hypre_CTAlloc(HYPRE_Int,  max_num_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_levels; i++)
      {
         block_num_coarse_indexes[i] = block_num_coarse_points[i];
      }
   }
   /* set block data */
   (mgr_data -> max_num_coarse_levels) = max_num_levels;
   (mgr_data -> block_size) = block_size;
   (mgr_data -> block_num_coarse_indexes) = block_num_coarse_indexes;
   (mgr_data -> block_cf_marker) = block_cf_marker;
   (mgr_data -> set_c_points_method) = 0;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRSetCpointsByPointMarkerArray( void      *mgr_vdata,
                                       HYPRE_Int  block_size,
                                       HYPRE_Int  max_num_levels,
                                       HYPRE_Int  *lvl_num_coarse_points,
                                       HYPRE_Int  **lvl_coarse_indexes,
                                       HYPRE_Int  *point_marker_array)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int  i, j;
   HYPRE_Int  **block_cf_marker = NULL;
   HYPRE_Int *block_num_coarse_indexes = NULL;

   /* free block cf_marker data if not previously destroyed */
   if ((mgr_data -> block_cf_marker) != NULL)
   {
      for (i = 0; i < (mgr_data -> max_num_coarse_levels); i++)
      {
         if ((mgr_data -> block_cf_marker)[i])
         {
            hypre_TFree((mgr_data -> block_cf_marker)[i], HYPRE_MEMORY_HOST);
            (mgr_data -> block_cf_marker)[i] = NULL;
         }
      }
      hypre_TFree(mgr_data -> block_cf_marker, HYPRE_MEMORY_HOST);
      (mgr_data -> block_cf_marker) = NULL;
   }
   if ((mgr_data -> block_num_coarse_indexes))
   {
      hypre_TFree((mgr_data -> block_num_coarse_indexes), HYPRE_MEMORY_HOST);
      (mgr_data -> block_num_coarse_indexes) = NULL;
   }

   /* store block cf_marker */
   block_cf_marker = hypre_CTAlloc(HYPRE_Int *, max_num_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_levels; i++)
   {
      block_cf_marker[i] = hypre_CTAlloc(HYPRE_Int, block_size, HYPRE_MEMORY_HOST);
      memset(block_cf_marker[i], FMRK, block_size * sizeof(HYPRE_Int));
   }
   for (i = 0; i < max_num_levels; i++)
   {
      for (j = 0; j < lvl_num_coarse_points[i]; j++)
      {
         block_cf_marker[i][j] = lvl_coarse_indexes[i][j];
      }
   }

   /* store block_num_coarse_points */
   if (max_num_levels > 0)
   {
      block_num_coarse_indexes = hypre_CTAlloc(HYPRE_Int,  max_num_levels, HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_levels; i++)
      {
         block_num_coarse_indexes[i] = lvl_num_coarse_points[i];
      }
   }
   /* set block data */
   (mgr_data -> max_num_coarse_levels) = max_num_levels;
   (mgr_data -> block_size) = block_size;
   (mgr_data -> block_num_coarse_indexes) = block_num_coarse_indexes;
   (mgr_data -> block_cf_marker) = block_cf_marker;
   (mgr_data -> point_marker_array) = point_marker_array;
   (mgr_data -> set_c_points_method) = 2;

   return hypre_error_flag;
}

/*Set number of points that remain part of the coarse grid throughout the hierarchy */
HYPRE_Int
hypre_MGRSetReservedCoarseNodes(void      *mgr_vdata,
                                HYPRE_Int reserved_coarse_size,
                                HYPRE_BigInt *reserved_cpt_index)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_BigInt *reserved_coarse_indexes = NULL;
   HYPRE_Int i;

   if (!mgr_data)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! MGR object empty!\n");
      return hypre_error_flag;
   }

   if (reserved_coarse_size < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   /* free data not previously destroyed */
   if ((mgr_data -> reserved_coarse_indexes))
   {
      hypre_TFree((mgr_data -> reserved_coarse_indexes), HYPRE_MEMORY_HOST);
      (mgr_data -> reserved_coarse_indexes) = NULL;
   }

   /* set reserved coarse nodes */
   if (reserved_coarse_size > 0)
   {
      reserved_coarse_indexes = hypre_CTAlloc(HYPRE_BigInt,  reserved_coarse_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < reserved_coarse_size; i++)
      {
         reserved_coarse_indexes[i] = reserved_cpt_index[i];
      }
   }
   (mgr_data -> reserved_coarse_size) = reserved_coarse_size;
   (mgr_data -> reserved_coarse_indexes) = reserved_coarse_indexes;

   return hypre_error_flag;
}

/* Set CF marker array */
HYPRE_Int
hypre_MGRCoarsen(hypre_ParCSRMatrix *S,
                 hypre_ParCSRMatrix *A,
                 HYPRE_Int fixed_coarse_size,
                 HYPRE_Int *fixed_coarse_indexes,
                 HYPRE_Int debug_flag,
                 hypre_IntArray **CF_marker_ptr,
                 HYPRE_Int cflag)
{
   HYPRE_Int   *CF_marker = NULL;
   HYPRE_Int *cindexes = fixed_coarse_indexes;
   HYPRE_Int    i, row, nc;
   HYPRE_Int nloc =  hypre_ParCSRMatrixNumRows(A);
   HYPRE_MemoryLocation memory_location;

   /* If this is the last level, coarsen onto fixed coarse set */
   if (cflag)
   {
      if (*CF_marker_ptr != NULL)
      {
         hypre_IntArrayDestroy(*CF_marker_ptr);
      }
      *CF_marker_ptr = hypre_IntArrayCreate(nloc);
      hypre_IntArrayInitialize(*CF_marker_ptr);
      hypre_IntArraySetConstantValues(*CF_marker_ptr, FMRK);
      memory_location = hypre_IntArrayMemoryLocation(*CF_marker_ptr);

      if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
      {
         hypre_IntArrayMigrate(*CF_marker_ptr, HYPRE_MEMORY_HOST);
      }
      CF_marker = hypre_IntArrayData(*CF_marker_ptr);

      /* first mark fixed coarse set */
      nc = fixed_coarse_size;
      for (i = 0; i < nc; i++)
      {
         CF_marker[cindexes[i]] = CMRK;
      }

      if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
      {
         hypre_IntArrayMigrate(*CF_marker_ptr, HYPRE_MEMORY_DEVICE);
      }
   }
   else
   {
      /* First coarsen to get initial CF splitting.
       * This is then followed by updating the CF marker to pass
       * coarse information to the next levels. NOTE: It may be
       * convenient to implement this way (allows the use of multiple
       * coarsening strategies without changing too much code),
       * but not necessarily the best option, compared to initializing
       * CF_marker first and then coarsening on subgraph which excludes
       * the initialized coarse nodes.
      */
      hypre_BoomerAMGCoarsen(S, A, 0, debug_flag, CF_marker_ptr);
      CF_marker = hypre_IntArrayData(*CF_marker_ptr);

      /* Update CF_marker to correct Cpoints marked as Fpoints. */
      nc = fixed_coarse_size;
      for (i = 0; i < nc; i++)
      {
         CF_marker[cindexes[i]] = CMRK;
      }
      /* set F-points to FMRK. This is necessary since the different coarsening schemes differentiate
       * between type of F-points (example Ruge coarsening). We do not need that distinction here.
      */
      for (row = 0; row < nloc; row++)
      {
         if (CF_marker[row] == CMRK) { continue; }
         CF_marker[row] = FMRK;
      }
#if 0
      /* IMPORTANT: Update coarse_indexes array to define the positions of the fixed coarse points
       * in the next level.
       */
      nc = 0;
      index_i = 0;
      for (row = 0; row < nloc; row++)
      {
         /* loop through new c-points */
         if (CF_marker[row] == CMRK) { nc++; }
         else if (CF_marker[row] == S_CMRK)
         {
            /* previously marked c-point is part of fixed coarse set. Track its current local index */
            cindexes[index_i++] = nc;
            /* reset c-point from S_CMRK to CMRK */
            cf_marker[row] = CMRK;
            nc++;
         }
         /* set F-points to FMRK. This is necessary since the different coarsening schemes differentiate
          * between type of F-points (example Ruge coarsening). We do not need that distinction here.
          */
         else
         {
            CF_marker[row] = FMRK;
         }
      }
      /* check if this should be last level */
      if ( nc == fixed_coarse_size)
      {
         last_level = 1;
      }
      //printf(" nc = %d and fixed coarse size = %d \n", nc, fixed_coarse_size);
#endif
   }

   return hypre_error_flag;
}

/* Scale ParCSR matrix A = scalar * A
 * A: the target CSR matrix
 * vector: array of real numbers
 */
HYPRE_Int
hypre_ParCSRMatrixLeftScale(HYPRE_Real *vector,
                            hypre_ParCSRMatrix *A)
{
   HYPRE_Int i, j, n_local;
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);

   hypre_CSRMatrix *A_offd         = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);

   n_local = hypre_CSRMatrixNumRows(A_diag);

   for (i = 0; i < n_local; i++)
   {
      HYPRE_Real factor = vector[i];
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         A_diag_data[j] *= factor;
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         A_offd_data[j] *= factor;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRComputeNonGalerkinCoarseGrid
 *
 * Computes the level (grid) operator A_H = RAP.
 *
 * Available methods:
 *   1: inv(A_FF) approximated by its (block) diagonal inverse
 *   2: CPR-like approx. with inv(A_FF) approx. by its diagonal inverse
 *   3: CPR-like approx. with inv(A_FF) approx. by its block diagonal inverse
 *   4: inv(A_FF) approximated by sparse approximate inverse
 *   5: Uses classical restriction R = [-Wr I] from input parameters list.
 *
 * Methods 1-4 assume that restriction is the injection operator.
 * Method 5 assumes that interpolation is the injection operator.
 *
 * TODO (VPM): Can we have a single function that works for host and device?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRComputeNonGalerkinCoarseGrid(hypre_ParCSRMatrix    *A_FF,
                                      hypre_ParCSRMatrix    *A_FC,
                                      hypre_ParCSRMatrix    *A_CF,
                                      hypre_ParCSRMatrix    *A_CC,
                                      hypre_ParCSRMatrix    *Wp,
                                      hypre_ParCSRMatrix    *Wr,
                                      HYPRE_Int              bsize,
                                      HYPRE_Int              ordering,
                                      HYPRE_Int              method,
                                      HYPRE_Int              max_elmts,
                                      hypre_ParCSRMatrix   **A_H_ptr)
{
   HYPRE_MemoryLocation   memory_location = hypre_ParCSRMatrixMemoryLocation(A_FF);

   hypre_ParCSRMatrix    *A_H = NULL;
   hypre_ParCSRMatrix    *A_Hc = NULL;
   hypre_ParCSRMatrix    *Wp_tmp = NULL;
   hypre_ParCSRMatrix    *Wr_tmp = NULL;
   hypre_ParCSRMatrix    *A_CF_truncated = NULL;
   hypre_ParCSRMatrix    *A_FF_inv = NULL;
   hypre_ParCSRMatrix    *minus_Wp = NULL;

   HYPRE_Int              i, i1, jj;
   HYPRE_Int              blk_inv_size;
   HYPRE_Real             neg_one = -1.0;
   HYPRE_Real             one = 1.0;

   if (method == 1)
   {
      if (Wp != NULL)
      {
         A_Hc = hypre_ParCSRMatMat(A_CF, Wp);
      }
      else
      {
         // Build block diagonal inverse for A_FF
         hypre_ParCSRMatrixBlockDiagMatrix(A_FF, 1, -1, NULL, 1, &A_FF_inv);

         // compute Wp = A_FF_inv * A_FC
         // NOTE: Use hypre_ParMatmul here instead of hypre_ParCSRMatMat to avoid padding
         // zero entries at diagonals for the latter routine. Use MatMat once this padding
         // issue is resolved since it is more efficient.
         //         hypre_ParCSRMatrix *Wp_tmp = hypre_ParCSRMatMat(A_FF_inv, A_FC);
         Wp_tmp = hypre_ParMatmul(A_FF_inv, A_FC);

         /* Compute correction A_Hc = A_CF * (A_FF_inv * A_FC); */
         A_Hc = hypre_ParCSRMatMat(A_CF, Wp_tmp);
         hypre_ParCSRMatrixDestroy(Wp_tmp);
         hypre_ParCSRMatrixDestroy(A_FF_inv);
      }
   }
   else if (method == 2 || method == 3)
   {
      /* Extract the diagonal of A_CF */
      hypre_MGRTruncateAcfCPR(A_CF, &A_CF_truncated);
      if (Wp != NULL)
      {
         A_Hc = hypre_ParCSRMatMat(A_CF_truncated, Wp);
      }
      else
      {
         blk_inv_size = method == 2 ? 1 : bsize;
         hypre_ParCSRMatrixBlockDiagMatrix(A_FF, blk_inv_size, -1, NULL, 1, &A_FF_inv);

         /* TODO (VPM): We shouldn't need to compute Wr_tmp since we are passing in Wr already */
         Wr_tmp = hypre_ParCSRMatMat(A_CF_truncated, A_FF_inv);
         A_Hc = hypre_ParCSRMatMat(Wr_tmp, A_FC);
         hypre_ParCSRMatrixDestroy(Wr_tmp);
         hypre_ParCSRMatrixDestroy(A_FF_inv);
      }
      hypre_ParCSRMatrixDestroy(A_CF_truncated);
   }
   else if (method == 4)
   {
      /* Approximate inverse for ideal interploation */
      hypre_MGRApproximateInverse(A_FF, &A_FF_inv);

      minus_Wp = hypre_ParCSRMatMat(A_FF_inv, A_FC);
      A_Hc = hypre_ParCSRMatMat(A_CF, minus_Wp);

      hypre_ParCSRMatrixDestroy(minus_Wp);
   }
   else if (method == 5)
   {
      if (!Wr)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Expected Wr matrix!");
         return hypre_error_flag;
      }

      /* A_Hc = Wr * A_FC */
      A_Hc = hypre_ParCSRMatMat(Wr, A_FC);
   }

   /* Drop small entries in the correction term A_Hc */
   if (max_elmts > 0)
   {
      // perform dropping for A_Hc
      // specific to multiphase poromechanics
      // we only keep the diagonal of each block
      HYPRE_Int        n_local_cpoints = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A_Hc));

      hypre_CSRMatrix *A_Hc_diag    = hypre_ParCSRMatrixDiag(A_Hc);
      HYPRE_Complex   *A_Hc_diag_a  = hypre_CSRMatrixData(A_Hc_diag);
      HYPRE_Int       *A_Hc_diag_i  = hypre_CSRMatrixI(A_Hc_diag);
      HYPRE_Int       *A_Hc_diag_j  = hypre_CSRMatrixJ(A_Hc_diag);
      HYPRE_Int        ncol_diag    = hypre_CSRMatrixNumCols(A_Hc_diag);

      hypre_CSRMatrix *A_Hc_offd    = hypre_ParCSRMatrixOffd(A_Hc);
      HYPRE_Complex   *A_Hc_offd_a  = hypre_CSRMatrixData(A_Hc_offd);
      HYPRE_Int       *A_Hc_offd_i  = hypre_CSRMatrixI(A_Hc_offd);
      HYPRE_Int       *A_Hc_offd_j  = hypre_CSRMatrixJ(A_Hc_offd);

      if (ordering == 0) // interleaved ordering
      {
         HYPRE_Int      *A_Hc_diag_i_new, *A_Hc_diag_j_new;
         HYPRE_Complex  *A_Hc_diag_a_new;
         HYPRE_Int       num_nonzeros_diag_new = 0;

         HYPRE_Int      *A_Hc_offd_i_new, *A_Hc_offd_j_new;
         HYPRE_Complex  *A_Hc_offd_a_new;
         HYPRE_Int       num_nonzeros_offd_new = 0;

         /* Allocate new memory */
         A_Hc_diag_i_new = hypre_CTAlloc(HYPRE_Int, n_local_cpoints + 1, memory_location);
         A_Hc_diag_j_new = hypre_CTAlloc(HYPRE_Int, (bsize + max_elmts) * n_local_cpoints,
                                         memory_location);
         A_Hc_diag_a_new = hypre_CTAlloc(HYPRE_Complex, (bsize + max_elmts) * n_local_cpoints,
                                         memory_location);
         A_Hc_offd_i_new = hypre_CTAlloc(HYPRE_Int, n_local_cpoints + 1, memory_location);
         A_Hc_offd_j_new = hypre_CTAlloc(HYPRE_Int, max_elmts * n_local_cpoints,
                                         memory_location);
         A_Hc_offd_a_new = hypre_CTAlloc(HYPRE_Complex, max_elmts * n_local_cpoints,
                                         memory_location);

         for (i = 0; i < n_local_cpoints; i++)
         {
            HYPRE_Int   max_num_nonzeros = A_Hc_diag_i[i + 1] - A_Hc_diag_i[i] +
                                           A_Hc_offd_i[i + 1] - A_Hc_offd_i[i];
            HYPRE_Int  *aux_j     = hypre_CTAlloc(HYPRE_Int, max_num_nonzeros, memory_location);
            HYPRE_Real *aux_data  = hypre_CTAlloc(HYPRE_Real, max_num_nonzeros, memory_location);
            HYPRE_Int   row_start = i - (i % bsize);
            HYPRE_Int   row_stop  = row_start + bsize - 1;
            HYPRE_Int   cnt       = 0;

            for (jj = A_Hc_offd_i[i]; jj < A_Hc_offd_i[i + 1]; jj++)
            {
               aux_j[cnt] = A_Hc_offd_j[jj] + ncol_diag;
               aux_data[cnt] = A_Hc_offd_a[jj];
               cnt++;
            }

            for (jj = A_Hc_diag_i[i]; jj < A_Hc_diag_i[i + 1]; jj++)
            {
               aux_j[cnt] = A_Hc_diag_j[jj];
               aux_data[cnt] = A_Hc_diag_a[jj];
               cnt++;
            }
            hypre_qsort2_abs(aux_j, aux_data, 0, cnt - 1);

            for (jj = A_Hc_diag_i[i]; jj < A_Hc_diag_i[i + 1]; jj++)
            {
               i1 = A_Hc_diag_j[jj];
               if (i1 >= row_start && i1 <= row_stop)
               {
                  // copy data to new arrays
                  A_Hc_diag_j_new[num_nonzeros_diag_new] = i1;
                  A_Hc_diag_a_new[num_nonzeros_diag_new] = A_Hc_diag_a[jj];
                  ++num_nonzeros_diag_new;
               }
               else
               {
                  // Do nothing
               }
            }

            if (max_elmts > 0)
            {
               for (jj = 0; jj < hypre_min(max_elmts, cnt); jj++)
               {
                  HYPRE_Int  col_idx   = aux_j[jj];
                  HYPRE_Real col_value = aux_data[jj];
                  if (col_idx < ncol_diag && (col_idx < row_start || col_idx > row_stop))
                  {
                     A_Hc_diag_j_new[num_nonzeros_diag_new] = col_idx;
                     A_Hc_diag_a_new[num_nonzeros_diag_new] = col_value;
                     ++num_nonzeros_diag_new;
                  }
                  else if (col_idx >= ncol_diag)
                  {
                     A_Hc_offd_j_new[num_nonzeros_offd_new] = col_idx - ncol_diag;
                     A_Hc_offd_a_new[num_nonzeros_offd_new] = col_value;
                     ++num_nonzeros_offd_new;
                  }
               }
            }
            A_Hc_diag_i_new[i + 1] = num_nonzeros_diag_new;
            A_Hc_offd_i_new[i + 1] = num_nonzeros_offd_new;

            hypre_TFree(aux_j, memory_location);
            hypre_TFree(aux_data, memory_location);
         }

         hypre_TFree(A_Hc_diag_i, memory_location);
         hypre_TFree(A_Hc_diag_j, memory_location);
         hypre_TFree(A_Hc_diag_a, memory_location);
         hypre_CSRMatrixI(A_Hc_diag) = A_Hc_diag_i_new;
         hypre_CSRMatrixJ(A_Hc_diag) = A_Hc_diag_j_new;
         hypre_CSRMatrixData(A_Hc_diag) = A_Hc_diag_a_new;
         hypre_CSRMatrixNumNonzeros(A_Hc_diag) = num_nonzeros_diag_new;

         hypre_TFree(A_Hc_offd_i, memory_location);
         hypre_TFree(A_Hc_offd_j, memory_location);
         hypre_TFree(A_Hc_offd_a, memory_location);
         hypre_CSRMatrixI(A_Hc_offd) = A_Hc_offd_i_new;
         hypre_CSRMatrixJ(A_Hc_offd) = A_Hc_offd_j_new;
         hypre_CSRMatrixData(A_Hc_offd) = A_Hc_offd_a_new;
         hypre_CSRMatrixNumNonzeros(A_Hc_offd) = num_nonzeros_offd_new;
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Non-interleaved dropping not implemented!");
         return hypre_error_flag;
      }
   }

   /* Coarse grid / Schur complement */
   hypre_ParCSRMatrixAdd(one, A_CC, neg_one, A_Hc, &A_H);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(A_Hc);

   /* Set output pointer */
   *A_H_ptr = A_H;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRComputeAlgebraicFixedStress(hypre_ParCSRMatrix  *A,
                                     HYPRE_BigInt        *mgr_idx_array,
                                     HYPRE_Solver         A_ff_solver)
{
   HYPRE_Int *U_marker, *S_marker, *P_marker;
   HYPRE_Int n_fine, i;
   HYPRE_BigInt ibegin;
   hypre_ParCSRMatrix *A_up;
   hypre_ParCSRMatrix *A_uu;
   hypre_ParCSRMatrix *A_su;
   hypre_ParCSRMatrix *A_pu;
   hypre_ParVector *e1_vector;
   hypre_ParVector *e2_vector;
   hypre_ParVector *e3_vector;
   hypre_ParVector *e4_vector;
   hypre_ParVector *e5_vector;

   n_fine = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   ibegin = hypre_ParCSRMatrixFirstRowIndex(A);
   hypre_assert(ibegin == mgr_idx_array[0]);
   U_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   S_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
   P_marker = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++)
   {
      U_marker[i] = -1;
      S_marker[i] = -1;
      P_marker[i] = -1;
   }

   // create C and F markers
   for (i = 0; i < n_fine; i++)
   {
      if (i < mgr_idx_array[1] - ibegin)
      {
         U_marker[i] = 1;
      }
      else if (i >= (mgr_idx_array[1] - ibegin) && i < (mgr_idx_array[2] - ibegin))
      {
         S_marker[i] = 1;
      }
      else
      {
         P_marker[i] = 1;
      }
   }

   // Get A_up
   hypre_MGRGetSubBlock(A, U_marker, P_marker, 0, &A_up);
   // GetA_uu
   hypre_MGRGetSubBlock(A, U_marker, U_marker, 0, &A_uu);
   // Get A_su
   hypre_MGRGetSubBlock(A, S_marker, U_marker, 0, &A_su);
   // Get A_pu
   hypre_MGRGetSubBlock(A, P_marker, U_marker, 0, &A_pu);

   e1_vector = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_up),
                                     hypre_ParCSRMatrixGlobalNumCols(A_up),
                                     hypre_ParCSRMatrixColStarts(A_up));
   hypre_ParVectorInitialize(e1_vector);
   hypre_ParVectorSetConstantValues(e1_vector, 1.0);

   e2_vector = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_uu),
                                     hypre_ParCSRMatrixGlobalNumRows(A_uu),
                                     hypre_ParCSRMatrixRowStarts(A_uu));
   hypre_ParVectorInitialize(e2_vector);
   hypre_ParVectorSetConstantValues(e2_vector, 0.0);

   e3_vector = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_uu),
                                     hypre_ParCSRMatrixGlobalNumRows(A_uu),
                                     hypre_ParCSRMatrixRowStarts(A_uu));
   hypre_ParVectorInitialize(e3_vector);
   hypre_ParVectorSetConstantValues(e3_vector, 0.0);

   e4_vector = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_su),
                                     hypre_ParCSRMatrixGlobalNumRows(A_su),
                                     hypre_ParCSRMatrixRowStarts(A_su));
   hypre_ParVectorInitialize(e4_vector);
   hypre_ParVectorSetConstantValues(e4_vector, 0.0);

   e5_vector = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A_pu),
                                     hypre_ParCSRMatrixGlobalNumRows(A_pu),
                                     hypre_ParCSRMatrixRowStarts(A_pu));
   hypre_ParVectorInitialize(e5_vector);
   hypre_ParVectorSetConstantValues(e5_vector, 0.0);

   // compute e2 = A_up * e1
   hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_up, e1_vector, 0.0, e2_vector, e2_vector);

   // solve e3 = A_uu^-1 * e2
   hypre_BoomerAMGSolve(A_ff_solver, A_uu, e2_vector, e3_vector);

   // compute e4 = A_su * e3
   hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_su, e3_vector, 0.0, e4_vector, e4_vector);

   // compute e4 = A_su * e3
   hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_su, e3_vector, 0.0, e4_vector, e4_vector);

   // print e4
   hypre_ParVectorPrintIJ(e4_vector, 1, "Dsp");

   // compute e5 = A_pu * e3
   hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_pu, e3_vector, 0.0, e5_vector, e5_vector);

   hypre_ParVectorPrintIJ(e5_vector, 1, "Dpp");

   hypre_ParVectorDestroy(e1_vector);
   hypre_ParVectorDestroy(e2_vector);
   hypre_ParVectorDestroy(e3_vector);
   hypre_ParCSRMatrixDestroy(A_uu);
   hypre_ParCSRMatrixDestroy(A_up);
   hypre_ParCSRMatrixDestroy(A_pu);
   hypre_ParCSRMatrixDestroy(A_su);
   hypre_TFree(U_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(S_marker, HYPRE_MEMORY_HOST);
   hypre_TFree(P_marker, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}


HYPRE_Int
hypre_MGRApproximateInverse(hypre_ParCSRMatrix      *A,
                            hypre_ParCSRMatrix     **A_inv)
{
   HYPRE_Int print_level, mr_max_row_nnz, mr_max_iter, nsh_max_row_nnz, nsh_max_iter, mr_col_version;
   HYPRE_Real mr_tol, nsh_tol;
   HYPRE_Real *droptol = hypre_CTAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrix *approx_A_inv = NULL;

   print_level = 0;
   nsh_max_iter = 2;
   nsh_max_row_nnz = 2; // default 1000
   mr_max_iter = 1;
   mr_tol = 1.0e-3;
   mr_max_row_nnz = 2; // default 800
   mr_col_version = 0;
   nsh_tol = 1.0e-3;
   droptol[0] = 1.0e-2;
   droptol[1] = 1.0e-2;

   hypre_ILUParCSRInverseNSH(A, &approx_A_inv, droptol, mr_tol, nsh_tol, HYPRE_REAL_MIN,
                             mr_max_row_nnz,
                             nsh_max_row_nnz, mr_max_iter, nsh_max_iter, mr_col_version, print_level);
   *A_inv = approx_A_inv;

   if (droptol) { hypre_TFree(droptol, HYPRE_MEMORY_HOST); }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_blas_smat_inv_n2
 *
 * TODO (VPM): move this function to seq_ls
 *--------------------------------------------------------------------------*/

void hypre_blas_smat_inv_n2 (HYPRE_Real *a)
{
   const HYPRE_Real a11 = a[0], a12 = a[1];
   const HYPRE_Real a21 = a[2], a22 = a[3];
   const HYPRE_Real det_inv = 1.0 / (a11 * a22 - a12 * a21);

   a[0] =  a22 * det_inv;
   a[1] = -a12 * det_inv;
   a[2] = -a21 * det_inv;
   a[3] =  a11 * det_inv;
}

/*--------------------------------------------------------------------------
 * hypre_blas_smat_inv_n3
 *
 * TODO (VPM): move this function to seq_ls
 *--------------------------------------------------------------------------*/

void hypre_blas_smat_inv_n3 (HYPRE_Real *a)
{
   const HYPRE_Real a11 = a[0],  a12 = a[1],  a13 = a[2];
   const HYPRE_Real a21 = a[3],  a22 = a[4],  a23 = a[5];
   const HYPRE_Real a31 = a[6],  a32 = a[7],  a33 = a[8];

   const HYPRE_Real det = a11 * a22 * a33 - a11 * a23 * a32 -
                          a12 * a21 * a33 + a12 * a23 * a31 +
                          a13 * a21 * a32 - a13 * a22 * a31;
   const HYPRE_Real det_inv = 1.0 / det;

   a[0] = (a22 * a33 - a23 * a32) * det_inv;
   a[1] = (a13 * a32 - a12 * a33) * det_inv;
   a[2] = (a12 * a23 - a13 * a22) * det_inv;
   a[3] = (a23 * a31 - a21 * a33) * det_inv;
   a[4] = (a11 * a33 - a13 * a31) * det_inv;
   a[5] = (a13 * a21 - a11 * a23) * det_inv;
   a[6] = (a21 * a32 - a22 * a31) * det_inv;
   a[7] = (a12 * a31 - a11 * a32) * det_inv;
   a[8] = (a11 * a22 - a12 * a21) * det_inv;
}

/*--------------------------------------------------------------------------
 * hypre_blas_smat_inv_n4
 *
 * TODO (VPM): move this function to seq_ls
 *--------------------------------------------------------------------------*/

void hypre_blas_smat_inv_n4 (HYPRE_Real *a)
{
   const HYPRE_Real a11 = a[0],  a12 = a[1],  a13 = a[2],  a14 = a[3];
   const HYPRE_Real a21 = a[4],  a22 = a[5],  a23 = a[6],  a24 = a[7];
   const HYPRE_Real a31 = a[8],  a32 = a[9],  a33 = a[10], a34 = a[11];
   const HYPRE_Real a41 = a[12], a42 = a[13], a43 = a[14], a44 = a[15];

   const HYPRE_Real M11 = a22 * a33 * a44 + a23 * a34 * a42 +
                          a24 * a32 * a43 - a22 * a34 * a43 -
                          a23 * a32 * a44 - a24 * a33 * a42;

   const HYPRE_Real M12 = a12 * a34 * a43 + a13 * a32 * a44 +
                          a14 * a33 * a42 - a12 * a33 * a44 -
                          a13 * a34 * a42 - a14 * a32 * a43;

   const HYPRE_Real M13 = a12 * a23 * a44 + a13 * a24 * a42 +
                          a14 * a22 * a43 - a12 * a24 * a43 -
                          a13 * a22 * a44 - a14 * a23 * a42;

   const HYPRE_Real M14 = a12 * a24 * a33 + a13 * a22 * a34 +
                          a14 * a23 * a32 - a12 * a23 * a34 -
                          a13 * a24 * a32 - a14 * a22 * a33;

   const HYPRE_Real M21 = a21 * a34 * a43 + a23 * a31 * a44 +
                          a24 * a33 * a41 - a21 * a33 * a44 -
                          a23 * a34 * a41 - a24 * a31 * a43;

   const HYPRE_Real M22 = a11 * a33 * a44 + a13 * a34 * a41 +
                          a14 * a31 * a43 - a11 * a34 * a43 -
                          a13 * a31 * a44 - a14 * a33 * a41;

   const HYPRE_Real M23 = a11 * a24 * a43 + a13 * a21 * a44 +
                          a14 * a23 * a41 - a11 * a23 * a44 -
                          a13 * a24 * a41 - a14 * a21 * a43;

   const HYPRE_Real M24 = a11 * a23 * a34 + a13 * a24 * a31 +
                          a14 * a21 * a33 - a11 * a24 * a33 -
                          a13 * a21 * a34 - a14 * a23 * a31;

   const HYPRE_Real M31 = a21 * a32 * a44 + a22 * a34 * a41 +
                          a24 * a31 * a42 - a21 * a34 * a42 -
                          a22 * a31 * a44 - a24 * a32 * a41;

   const HYPRE_Real M32 = a11 * a34 * a42 + a12 * a31 * a44 +
                          a14 * a32 * a41 - a11 * a32 * a44 -
                          a12 * a34 * a41 - a14 * a31 * a42;

   const HYPRE_Real M33 = a11 * a22 * a44 + a12 * a24 * a41 +
                          a14 * a21 * a42 - a11 * a24 * a42 -
                          a12 * a21 * a44 - a14 * a22 * a41;

   const HYPRE_Real M34 = a11 * a24 * a32 + a12 * a21 * a34 +
                          a14 * a22 * a31 - a11 * a22 * a34 -
                          a12 * a24 * a31 - a14 * a21 * a32;

   const HYPRE_Real M41 = a21 * a33 * a42 + a22 * a31 * a43 +
                          a23 * a32 * a41 - a21 * a32 * a43 -
                          a22 * a33 * a41 - a23 * a31 * a42;

   const HYPRE_Real M42 = a11 * a32 * a43 + a12 * a33 * a41 +
                          a13 * a31 * a42 - a11 * a33 * a42 -
                          a12 * a31 * a43 - a13 * a32 * a41;

   const HYPRE_Real M43 = a11 * a23 * a42 + a12 * a21 * a43 +
                          a13 * a22 * a41 - a11 * a22 * a43 -
                          a12 * a23 * a41 - a13 * a21 * a42;

   const HYPRE_Real M44 = a11 * a22 * a33 + a12 * a23 * a31 +
                          a13 * a21 * a32 - a11 * a23 * a32 -
                          a12 * a21 * a33 - a13 * a22 * a31;

   const HYPRE_Real det = a11 * M11 + a12 * M21 + a13 * M31 + a14 * M41;
   HYPRE_Real       det_inv = 1.0 / det;

   //if ( hypre_abs(det) < 1e-22 ) {
   //hypre_printf("### WARNING: Matrix is nearly singular! det = %e\n", det);
   /*
   printf("##----------------------------------------------\n");
   printf("## %12.5e %12.5e %12.5e \n", a0, a1, a2);
   printf("## %12.5e %12.5e %12.5e \n", a3, a4, a5);
   printf("## %12.5e %12.5e %12.5e \n", a5, a6, a7);
   printf("##----------------------------------------------\n");
   getchar();
   */
   //}

   a[0]  = M11 * det_inv; a[1]  = M12 * det_inv; a[2]  = M13 * det_inv; a[3]  = M14 * det_inv;
   a[4]  = M21 * det_inv; a[5]  = M22 * det_inv; a[6]  = M23 * det_inv; a[7]  = M24 * det_inv;
   a[8]  = M31 * det_inv; a[9]  = M32 * det_inv; a[10] = M33 * det_inv; a[11] = M34 * det_inv;
   a[12] = M41 * det_inv; a[13] = M42 * det_inv; a[14] = M43 * det_inv; a[15] = M44 * det_inv;
}

/*--------------------------------------------------------------------------
 * hypre_MGRSmallBlkInverse
 *
 * TODO (VPM): move this function to seq_ls
 *--------------------------------------------------------------------------*/

void hypre_MGRSmallBlkInverse(HYPRE_Real *mat,
                              HYPRE_Int   blk_size)
{
   if (blk_size == 2)
   {
      hypre_blas_smat_inv_n2(mat);
   }
   else if (blk_size == 3)
   {
      hypre_blas_smat_inv_n3(mat);
   }
   else if (blk_size == 4)
   {
      hypre_blas_smat_inv_n4(mat);
   }
}

/*--------------------------------------------------------------------------
 * hypre_MGRSmallBlkInverse
 *
 * TODO (VPM): move this function to seq_ls
 *--------------------------------------------------------------------------*/

void hypre_blas_mat_inv(HYPRE_Real *a,
                        HYPRE_Int n)
{
   HYPRE_Int i, j, k, l, u, kn, in;
   HYPRE_Real alinv;
   if (n == 4)
   {
      hypre_blas_smat_inv_n4(a);
   }
   else
   {
      for (k = 0; k < n; ++k)
      {
         kn = k * n;
         l  = kn + k;

         //if (hypre_abs(a[l]) < HYPRE_REAL_MIN) {
         //   printf("### WARNING: Diagonal entry is close to zero!");
         //   printf("### WARNING: diag_%d=%e\n", k, a[l]);
         //   a[l] = HYPRE_REAL_MIN;
         //}
         alinv = 1.0 / a[l];
         a[l] = alinv;

         for (j = 0; j < k; ++j)
         {
            u = kn + j; a[u] *= alinv;
         }

         for (j = k + 1; j < n; ++j)
         {
            u = kn + j; a[u] *= alinv;
         }

         for (i = 0; i < k; ++i)
         {
            in = i * n;
            for (j = 0; j < n; ++j)
               if (j != k)
               {
                  u = in + j; a[u] -= a[in + k] * a[kn + j];
               } // end if (j!=k)
         }

         for (i = k + 1; i < n; ++i)
         {
            in = i * n;
            for (j = 0; j < n; ++j)
               if (j != k)
               {
                  u = in + j; a[u] -= a[in + k] * a[kn + j];
               } // end if (j!=k)
         }

         for (i = 0; i < k; ++i)
         {
            u = i * n + k; a[u] *= -alinv;
         }

         for (i = k + 1; i < n; ++i)
         {
            u = i * n + k; a[u] *= -alinv;
         }
      } // end for (k=0; k<n; ++k)
   }// end if
}

HYPRE_Int
hypre_block_jacobi_solve( hypre_ParCSRMatrix *A,
                          hypre_ParVector    *f,
                          hypre_ParVector    *u,
                          HYPRE_Int           blk_size,
                          HYPRE_Int           method,
                          HYPRE_Real         *diaginv,
                          hypre_ParVector    *Vtemp )
{
   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Real      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle = NULL;

   HYPRE_Int        n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_Vector    *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real      *u_data  = hypre_VectorData(u_local);

   hypre_Vector    *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Real      *f_data  = hypre_VectorData(f_local);

   hypre_Vector    *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   HYPRE_Real      *Vtemp_data = hypre_VectorData(Vtemp_local);
   HYPRE_Real      *Vext_data = NULL;
   HYPRE_Real      *v_buf_data = NULL;

   HYPRE_Int        i, j, k;
   HYPRE_Int        ii, jj;
   HYPRE_Int        bidx, bidx1;
   HYPRE_Int        num_sends;
   HYPRE_Int        index, start;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Real      *res;

   const HYPRE_Int  nb2 = blk_size * blk_size;
   const HYPRE_Int  n_block = n / blk_size;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //   HYPRE_Int num_threads = hypre_NumThreads();

   res = hypre_CTAlloc(HYPRE_Real, blk_size, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (num_procs > 1)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(HYPRE_Real,
                                 hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends),
                                 HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd)
      {
         A_offd_j = hypre_CSRMatrixJ(A_offd);
         A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            v_buf_data[index++] = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);
   }

   /*-----------------------------------------------------------------
   * Copy current approximation into temporary vector.
   *-----------------------------------------------------------------*/

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n; i++)
   {
      Vtemp_data[i] = u_data[i];
      //printf("u_old[%d] = %e\n",i,Vtemp_data[i]);
   }
   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   /*-----------------------------------------------------------------
   * Relax points block by block
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      for (j = 0; j < blk_size; j++)
      {
         bidx = i * blk_size + j;
         res[j] = f_data[bidx];
         for (jj = A_diag_i[bidx]; jj < A_diag_i[bidx + 1]; jj++)
         {
            ii = A_diag_j[jj];
            if (method == 0)
            {
               // Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            else if (method == 1)
            {
               // Gauss-Seidel for diagonal part
               res[j] -= A_diag_data[jj] * u_data[ii];
            }
            else
            {
               // Default do Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            //printf("%d: Au= %e * %e =%e\n",ii,A_diag_data[jj],Vtemp_data[ii], res[j]);
         }
         for (jj = A_offd_i[bidx]; jj < A_offd_i[bidx + 1]; jj++)
         {
            // always do Jacobi for off-diagonal part
            ii = A_offd_j[jj];
            res[j] -= A_offd_data[jj] * Vext_data[ii];
         }
         //printf("%d: res = %e\n",bidx,res[j]);
      }

      for (j = 0; j < blk_size; j++)
      {
         bidx1 = i * blk_size + j;
         for (k = 0; k < blk_size; k++)
         {
            bidx  = i * nb2 + j * blk_size + k;
            u_data[bidx1] += res[k] * diaginv[bidx];
            //printf("u[%d] = %e, diaginv[%d] = %e\n",bidx1,u_data[bidx1],bidx,diaginv[bidx]);
         }
         //printf("u[%d] = %e\n",bidx1,u_data[bidx1]);
      }
   }

   if (num_procs > 1)
   {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(res, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBlockRelaxSolveDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBlockRelaxSolveDevice( hypre_ParCSRMatrix  *B,
                                hypre_ParCSRMatrix  *A,
                                hypre_ParVector     *f,
                                hypre_ParVector     *u,
                                hypre_ParVector     *Vtemp,
                                HYPRE_Real           relax_weight )
{
   hypre_GpuProfilingPushRange("BlockRelaxSolve");

   /* Copy f into temporary vector */
   hypre_ParVectorCopy(f, Vtemp);

   /* Perform Matvec: Vtemp = w * (f - Au) */
   if (hypre_ParVectorAllZeros(u))
   {
#if defined(HYPRE_DEBUG)
      hypre_assert(hypre_ParVectorInnerProd(u, u) == 0.0);
#endif
      hypre_ParVectorScale(relax_weight, Vtemp);
   }
   else
   {
      hypre_ParCSRMatrixMatvec(-relax_weight, A, u, relax_weight, Vtemp);
   }

   /* Update solution: u += B * Vtemp */
   hypre_ParCSRMatrixMatvec(1.0, B, Vtemp, 1.0, u);
   hypre_ParVectorAllZeros(u) = 0;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBlockRelaxSolve
 *
 * Computes a block Jacobi relaxation of matrix A, given the inverse of the
 * diagonal blocks (of A) obtained by calling hypre_MGRBlockRelaxSetup.
 *
 * TODO: Adapt to relax on specific points based on CF_marker information
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBlockRelaxSolve( hypre_ParCSRMatrix *A,
                          hypre_ParVector    *f,
                          hypre_ParVector    *u,
                          HYPRE_Int           blk_size,
                          HYPRE_Int           n_block,
                          HYPRE_Int           left_size,
                          HYPRE_Int           method,
                          HYPRE_Real         *diaginv,
                          hypre_ParVector    *Vtemp )
{
   HYPRE_UNUSED_VAR(left_size);

   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int       *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Real      *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle = NULL;

   HYPRE_Int        n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_Vector    *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real      *u_data  = hypre_VectorData(u_local);

   hypre_Vector    *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Real      *f_data  = hypre_VectorData(f_local);

   hypre_Vector    *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   HYPRE_Real      *Vtemp_data = hypre_VectorData(Vtemp_local);
   HYPRE_Real      *Vext_data = NULL;
   HYPRE_Real      *v_buf_data = NULL;

   HYPRE_Int        i, j, k;
   HYPRE_Int        ii, jj;
   HYPRE_Int        bidx, bidx1, bidxm1;
   HYPRE_Int        num_sends;
   HYPRE_Int        index, start;
   HYPRE_Int        num_procs, my_id;
   HYPRE_Real      *res;

   const HYPRE_Int  nb2 = blk_size * blk_size;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //   HYPRE_Int num_threads = hypre_NumThreads();

   res = hypre_CTAlloc(HYPRE_Real,  blk_size, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   if (num_procs > 1)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(HYPRE_Real,
                                 hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends),
                                 HYPRE_MEMORY_HOST);

      Vext_data = hypre_CTAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_HOST);

      if (num_cols_offd)
      {
         A_offd_j = hypre_CSRMatrixJ(A_offd);
         A_offd_data = hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            v_buf_data[index++]
               = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                                                  Vext_data);
   }

   /*-----------------------------------------------------------------
   * Copy current approximation into temporary vector.
   *-----------------------------------------------------------------*/

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n; i++)
   {
      Vtemp_data[i] = u_data[i];
      //printf("u_old[%d] = %e\n",i,Vtemp_data[i]);
   }
   if (num_procs > 1)
   {
      hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   /*-----------------------------------------------------------------
   * Relax points block by block
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      bidxm1 = i * blk_size;
      for (j = 0; j < blk_size; j++)
      {
         bidx = bidxm1 + j;
         res[j] = f_data[bidx];
         for (jj = A_diag_i[bidx]; jj < A_diag_i[bidx + 1]; jj++)
         {
            ii = A_diag_j[jj];
            if (method == 0)
            {
               // Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            else if (method == 1)
            {
               // Gauss-Seidel for diagonal part
               res[j] -= A_diag_data[jj] * u_data[ii];
            }
            else
            {
               // Default do Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            //printf("%d: Au= %e * %e =%e\n",ii,A_diag_data[jj],Vtemp_data[ii], res[j]);
         }
         for (jj = A_offd_i[bidx]; jj < A_offd_i[bidx + 1]; jj++)
         {
            // always do Jacobi for off-diagonal part
            ii = A_offd_j[jj];
            res[j] -= A_offd_data[jj] * Vext_data[ii];
         }
         //printf("%d: res = %e\n",bidx,res[j]);
      }

      for (j = 0; j < blk_size; j++)
      {
         bidx1 = bidxm1 + j;
         for (k = 0; k < blk_size; k++)
         {
            bidx  = i * nb2 + j * blk_size + k;
            u_data[bidx1] += res[k] * diaginv[bidx];
            //printf("u[%d] = %e, diaginv[%d] = %e\n",bidx1,u_data[bidx1],bidx,diaginv[bidx]);
         }
         //printf("u[%d] = %e\n",bidx1,u_data[bidx1]);
      }
   }
   if (num_procs > 1)
   {
      hypre_TFree(Vext_data, HYPRE_MEMORY_HOST);
      hypre_TFree(v_buf_data, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(res, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BlockDiagInvLapack
 *
 * TODO (VPM): move this function to seq_ls
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BlockDiagInvLapack(HYPRE_Real *diag, HYPRE_Int N, HYPRE_Int blk_size)
{
   HYPRE_Int nblock, left_size, i;
   //HYPRE_Int *IPIV = hypre_CTAlloc(HYPRE_Int, blk_size, HYPRE_MEMORY_HOST);
   HYPRE_Int LWORK = blk_size * blk_size;
   HYPRE_Real *WORK = hypre_CTAlloc(HYPRE_Real, LWORK, HYPRE_MEMORY_HOST);
   HYPRE_Int INFO;

   HYPRE_Real wall_time;
   HYPRE_Int my_id;
   MPI_Comm comm = hypre_MPI_COMM_WORLD;
   hypre_MPI_Comm_rank(comm, &my_id);

   nblock = N / blk_size;
   left_size = N - blk_size * nblock;
   i = nblock;
   HYPRE_Int *IPIV = hypre_CTAlloc(HYPRE_Int, blk_size, HYPRE_MEMORY_HOST);

   wall_time = time_getWallclockSeconds();
   if (blk_size >= 2 && blk_size <= 4)
   {
      for (i = 0; i < nblock; i++)
      {
         hypre_MGRSmallBlkInverse(diag + i * LWORK, blk_size);
         //hypre_blas_smat_inv_n2(diag+i*LWORK);
      }
   }
   else if (blk_size > 4)
   {
      for (i = 0; i < nblock; i++)
      {
         hypre_dgetrf(&blk_size, &blk_size, diag + i * LWORK, &blk_size, IPIV, &INFO);
         hypre_dgetri(&blk_size, diag + i * LWORK, &blk_size, IPIV, WORK, &LWORK, &INFO);
      }
   }

   // Left size
   if (left_size > 0)
   {
      hypre_dgetrf(&left_size, &left_size, diag + i * LWORK, &left_size, IPIV, &INFO);
      hypre_dgetri(&left_size, diag + i * LWORK, &left_size, IPIV, WORK, &LWORK, &INFO);
   }
   wall_time = time_getWallclockSeconds() - wall_time;
   //if (my_id == 0) hypre_printf("Proc = %d, Compute inverse time: %1.5f\n", my_id, wall_time);

   hypre_TFree(IPIV, HYPRE_MEMORY_HOST);
   hypre_TFree(WORK, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixExtractBlockDiagHost
 *
 * Extract the block diagonal part of a A or a principal submatrix of A
 * defined by a marker (point_type) in an associated CF_marker array.
 * The result is an array of (flattened) block diagonals.
 *
 * If CF marker array is NULL, it returns an array of the (flattened)
 * block diagonal of the entire matrix A.
 *
 * Options for diag_type are:
 *   diag_type = 1: return the inverse of the block diagonals
 *   otherwise    : return the block diagonals
 *
 * On return, blk_diag_size contains the size of the returned
 * (flattened) array. (i.e. nnz of extracted block diagonal)
 *
 * Input parameters are:
 *    A          - original ParCSR matrix
 *    blk_size   - Size of diagonal blocks to extract
 *    CF_marker  - Array prescribing submatrix from which to extract
 *                 block diagonals. Ignored if NULL.
 *    point_type - marker tag in CF_marker array to extract diagonal
 *    diag_type  - Type of block diagonal entries to return.
 *                 Currently supports block diagonal or inverse block
 *                 diagonal entries (diag_type = 1).
 *
 * Output parameters are:
 *      diag_ptr - Array of block diagonal entries
 * blk_diag_size - number of entries in extracted block diagonal
 *                 (size of diag_ptr).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixExtractBlockDiagHost( hypre_ParCSRMatrix   *par_A,
                                        HYPRE_Int             blk_size,
                                        HYPRE_Int             num_points,
                                        HYPRE_Int             point_type,
                                        HYPRE_Int            *CF_marker,
                                        HYPRE_Int             diag_size,
                                        HYPRE_Int             diag_type,
                                        HYPRE_Real           *diag_data )
{
   HYPRE_UNUSED_VAR(diag_size);

   hypre_CSRMatrix      *A_diag       = hypre_ParCSRMatrixDiag(par_A);
   HYPRE_Int             nrows        = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Complex        *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);

   HYPRE_Int             i, j;
   HYPRE_Int             ii, jj;
   HYPRE_Int             bidx, bidxm1, bidxp1, ridx, didx;
   HYPRE_Int             row_offset;

   HYPRE_Int             whole_num_points, cnt, bstart;
   HYPRE_Int             bs2 = blk_size * blk_size;
   HYPRE_Int             num_blocks;
   HYPRE_Int             left_size = 0;

   // First count the number of points matching point_type in CF_marker
   num_blocks       = num_points / blk_size;
   whole_num_points = blk_size * num_blocks;
   left_size        = num_points - whole_num_points;
   bstart           = bs2 * num_blocks;

   /*-----------------------------------------------------------------
    * Get all the diagonal sub-blocks
    *-----------------------------------------------------------------*/

   HYPRE_ANNOTATE_REGION_BEGIN("%s", "ExtractDiagSubBlocks");
   if (CF_marker == NULL)
   {
      // CF Marker is NULL. Consider all rows of matrix.
      for (i = 0; i < num_blocks; i++)
      {
         bidxm1 = i * blk_size;
         bidxp1 = (i + 1) * blk_size;

         for (j = 0; j < blk_size; j++)
         {
            for (ii = A_diag_i[bidxm1 + j]; ii < A_diag_i[bidxm1 + j + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if ((jj >= bidxm1) &&
                   (jj < bidxp1)  &&
                   hypre_cabs(A_diag_data[ii]) > HYPRE_REAL_MIN)
               {
                  bidx = j * blk_size + jj - bidxm1;
                  diag_data[i * bs2 + bidx] = A_diag_data[ii];
               }
            }
         }
      }

      // deal with remaining points if any
      if (left_size)
      {
         bidxm1 = whole_num_points;
         bidxp1 = num_points;
         for (j = 0; j < left_size; j++)
         {
            for (ii = A_diag_i[bidxm1 + j]; ii < A_diag_i[bidxm1 + j + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if ((jj >= bidxm1) &&
                   (jj < bidxp1)  &&
                   hypre_cabs(A_diag_data[ii]) > HYPRE_REAL_MIN)
               {
                  bidx = j * left_size + jj - bidxm1;
                  diag_data[bstart + bidx] = A_diag_data[ii];
               }
            }
         }
      }
   }
   else
   {
      // extract only block diagonal of submatrix defined by CF marker
      cnt = 0;
      row_offset = 0;
      for (i = 0; i < nrows; i++)
      {
         if (CF_marker[i] == point_type)
         {
            bidx = cnt / blk_size;
            ridx = cnt % blk_size;
            bidxm1 = bidx * blk_size;
            bidxp1 = (bidx + 1) * blk_size;
            for (ii = A_diag_i[i]; ii < A_diag_i[i + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if (CF_marker[jj] == point_type)
               {
                  if ((jj - row_offset >= bidxm1) &&
                      (jj - row_offset < bidxp1)  &&
                      (hypre_cabs(A_diag_data[ii]) > HYPRE_REAL_MIN))
                  {
                     didx = bidx * bs2 + ridx * blk_size + jj - bidxm1 - row_offset;
                     diag_data[didx] = A_diag_data[ii];
                  }
               }
            }
            if (++cnt == whole_num_points)
            {
               break;
            }
         }
         else
         {
            row_offset++;
         }
      }

      // remaining points
      for (i = whole_num_points; i < num_points; i++)
      {
         if (CF_marker[i] == point_type)
         {
            bidx = num_blocks;
            ridx = cnt - whole_num_points;
            bidxm1 = whole_num_points;
            bidxp1 = num_points;
            for (ii = A_diag_i[i]; ii < A_diag_i[i + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if (CF_marker[jj] == point_type)
               {
                  if ((jj - row_offset >= bidxm1) &&
                      (jj - row_offset < bidxp1)  &&
                      (hypre_cabs(A_diag_data[ii]) > HYPRE_REAL_MIN))
                  {
                     didx = bstart + ridx * left_size + jj - bidxm1 - row_offset;
                     diag_data[didx] = A_diag_data[ii];
                  }
               }
            }
            cnt++;
         }
         else
         {
            row_offset++;
         }
      }
   }
   HYPRE_ANNOTATE_REGION_END("%s", "ExtractDiagSubBlocks");

   /*-----------------------------------------------------------------
    * Compute the inverses of all the diagonal sub-blocks
    *-----------------------------------------------------------------*/

   if (diag_type == 1)
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "InvertDiagSubBlocks");
      if (blk_size > 1)
      {
         hypre_BlockDiagInvLapack(diag_data, num_points, blk_size);
      }
      else
      {
         for (i = 0; i < num_points; i++)
         {
            if (hypre_cabs(diag_data[i]) < HYPRE_REAL_MIN)
            {
               diag_data[i] = 0.0;
            }
            else
            {
               diag_data[i] = 1.0 / diag_data[i];
            }
         }
      }
      HYPRE_ANNOTATE_REGION_END("%s", "InvertDiagSubBlocks");
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlockDiagMatrix
 *
 * Extract the block diagonal part of a A or a principal submatrix of A defined
 * by a marker (point_type) in an associated CF_marker array. The result is
 * a new block diagonal parCSR matrix.
 *
 * If CF marker array is NULL, it returns the block diagonal of the matrix A.
 *
 * Options for diag_type are:
 *    diag_type = 1: return the inverse of the block diagonals
 *    otherwise : return the block diagonals
 *
 * Input parameters are:
 *    par_A      - original ParCSR matrix
 *    blk_size   - Size of diagonal blocks to extract
 *    CF_marker  - Array prescribing submatrix from which to extract block
 *                 diagonals. Ignored if NULL.
 *    point_type - marker tag in CF_marker array to extract diagonal
 *    diag_type  - Type of block diagonal entries to return. Currently supports
 *                 block diagonal or inverse block diagonal entries.
 *
 * Output parameters are:
 *    B_ptr      - New block diagonal matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlockDiagMatrix( hypre_ParCSRMatrix  *A,
                                   HYPRE_Int            blk_size,
                                   HYPRE_Int            point_type,
                                   HYPRE_Int           *CF_marker,
                                   HYPRE_Int            diag_type,
                                   hypre_ParCSRMatrix **B_ptr )
{
#if defined (HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixBlockDiagMatrixDevice(A, blk_size, point_type,
                                              CF_marker, diag_type, B_ptr);
   }
   else
#endif
   {
      hypre_ParCSRMatrixBlockDiagMatrixHost(A, blk_size, point_type,
                                            CF_marker, diag_type, B_ptr);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixBlockDiagMatrixHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixBlockDiagMatrixHost( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int            blk_size,
                                       HYPRE_Int            point_type,
                                       HYPRE_Int           *CF_marker,
                                       HYPRE_Int            diag_type,
                                       hypre_ParCSRMatrix **B_ptr )
{
   /* Input matrix info */
   MPI_Comm              comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt         *row_starts_A    = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt          num_rows_A      = hypre_ParCSRMatrixGlobalNumRows(A);
   hypre_CSRMatrix      *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             A_diag_num_rows = hypre_CSRMatrixNumRows(A_diag);

   /* Global block matrix info */
   hypre_ParCSRMatrix   *par_B;
   HYPRE_BigInt          num_rows_B;
   HYPRE_BigInt          row_starts_B[2];

   /* Diagonal block matrix info */
   hypre_CSRMatrix      *B_diag;
   HYPRE_Int             B_diag_num_rows = 0;
   HYPRE_Int             B_diag_size;
   HYPRE_Int            *B_diag_i;
   HYPRE_Int            *B_diag_j;
   HYPRE_Complex        *B_diag_data;

   /* Local variables */
   HYPRE_BigInt          num_rows_big;
   HYPRE_BigInt          scan_recv;
   HYPRE_Int             num_procs, my_id;
   HYPRE_Int             nb2 = blk_size * blk_size;
   HYPRE_Int             num_blocks, num_left;
   HYPRE_Int             bidx, i, j, k;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   /* Sanity check */
   if ((num_rows_A > 0) && (num_rows_A < blk_size))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error!!! Input matrix is smaller than block size.");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    * Count the number of points matching point_type in CF_marker
    *-----------------------------------------------------------------*/

   if (CF_marker == NULL)
   {
      B_diag_num_rows = A_diag_num_rows;
   }
   else
   {
#if !defined(_MSC_VER) && defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:B_diag_num_rows) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < A_diag_num_rows; i++)
      {
         B_diag_num_rows += (CF_marker[i] == point_type) ? 1 : 0;
      }
   }
   num_blocks  = B_diag_num_rows / blk_size;
   num_left    = B_diag_num_rows - num_blocks * blk_size;
   B_diag_size = blk_size * (blk_size * num_blocks) + num_left * num_left;

   /*-----------------------------------------------------------------
    * Compute global number of rows and partitionings
    *-----------------------------------------------------------------*/

   if (CF_marker)
   {
      num_rows_big = (HYPRE_BigInt) B_diag_num_rows;
      hypre_MPI_Scan(&num_rows_big, &scan_recv, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      /* first point in my range */
      row_starts_B[0] = scan_recv - num_rows_big;

      /* first point in next proc's range */
      row_starts_B[1] = scan_recv;
      if (my_id == (num_procs - 1))
      {
         num_rows_B = row_starts_B[1];
      }
      hypre_MPI_Bcast(&num_rows_B, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      row_starts_B[0] = row_starts_A[0];
      row_starts_B[1] = row_starts_A[1];
      num_rows_B = num_rows_A;
   }

   /* Create matrix B */
   par_B = hypre_ParCSRMatrixCreate(comm,
                                    num_rows_B,
                                    num_rows_B,
                                    row_starts_B,
                                    row_starts_B,
                                    0,
                                    B_diag_size,
                                    0);
   hypre_ParCSRMatrixInitialize_v2(par_B, HYPRE_MEMORY_HOST);
   B_diag      = hypre_ParCSRMatrixDiag(par_B);
   B_diag_i    = hypre_CSRMatrixI(B_diag);
   B_diag_j    = hypre_CSRMatrixJ(B_diag);
   B_diag_data = hypre_CSRMatrixData(B_diag);

   /*-----------------------------------------------------------------------
    * Extract coefficients
    *-----------------------------------------------------------------------*/

   hypre_ParCSRMatrixExtractBlockDiagHost(A, blk_size, B_diag_num_rows,
                                          point_type, CF_marker,
                                          B_diag_size, diag_type,
                                          B_diag_data);

   /*-----------------------------------------------------------------
    * Set row/col indices of diagonal blocks
    *-----------------------------------------------------------------*/

   B_diag_i[B_diag_num_rows] = B_diag_size;
   for (i = 0; i < num_blocks; i++)
   {
      //diag_local = &diag[i * nb2];
      for (k = 0; k < blk_size; k++)
      {
         B_diag_i[i * blk_size + k] = i * nb2 + k * blk_size;

         for (j = 0; j < blk_size; j++)
         {
            bidx = i * nb2 + k * blk_size + j;
            B_diag_j[bidx] = i * blk_size + j;
            //B_diag_data[bidx] = diag_local[k * blk_size + j];
         }
      }
   }

   /*-----------------------------------------------------------------
    * Treat the remaining points
    *-----------------------------------------------------------------*/

   //diag_local = &diag[num_blocks * nb2];
   for (k = 0; k < num_left; k++)
   {
      B_diag_i[num_blocks * blk_size + k] = num_blocks * nb2 + k * num_left;

      for (j = 0; j < num_left; j++)
      {
         bidx = num_blocks * nb2 + k * num_left + j;
         B_diag_j[bidx] = num_blocks * blk_size + j;
         //B_diag_data[bidx] = diag_local[k * num_left + j];
      }
   }

   /* Set output pointer */
   *B_ptr = par_B;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRBlockRelaxSetup
 *
 * Setup block smoother. Computes the entries of the inverse of the block
 * diagonal matrix with blk_size diagonal blocks.
 *
 * Current implementation ignores reserved C-pts and acts on whole matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRBlockRelaxSetup( hypre_ParCSRMatrix *A,
                          HYPRE_Int           blk_size,
                          HYPRE_Real        **diaginvptr )
{
   hypre_CSRMatrix      *A_diag   = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int             num_rows = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             num_blocks;
   HYPRE_Int             diag_size;
   HYPRE_Complex        *diaginv = *diaginvptr;

   num_blocks = 1 + (num_rows - 1) / blk_size;
   diag_size  = blk_size * (blk_size * num_blocks);

   hypre_TFree(diaginv, HYPRE_MEMORY_HOST);
   diaginv = hypre_CTAlloc(HYPRE_Complex, diag_size, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixExtractBlockDiagHost(A, blk_size, num_rows, 0, NULL,
                                          diag_size, 1, diaginv);

   *diaginvptr = diaginv;

#if 0
   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real     *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             i, j, k;
   HYPRE_Int             ii, jj;
   HYPRE_Int             bidx, bidxm1, bidxp1;
   HYPRE_Int         num_procs, my_id;

   const HYPRE_Int     nb2 = blk_size * blk_size;
   HYPRE_Int           n_block;
   HYPRE_Int           left_size, inv_size;
   HYPRE_Real        *diaginv = *diaginvptr;


   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //HYPRE_Int num_threads = hypre_NumThreads();

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size * n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size * n_block;
   }

   n_block = n / blk_size;
   left_size = n - blk_size * n_block;

   inv_size  = nb2 * n_block + left_size * left_size;

   if (diaginv != NULL)
   {
      hypre_TFree(diaginv, HYPRE_MEMORY_HOST);
      diaginv = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);
   }
   else
   {
      diaginv = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------
   * Get all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      bidxm1 = i * blk_size;
      bidxp1 = (i + 1) * blk_size;
      //printf("bidxm1 = %d,bidxp1 = %d\n",bidxm1,bidxp1);

      for (k = 0; k < blk_size; k++)
      {
         for (j = 0; j < blk_size; j++)
         {
            bidx = i * nb2 + k * blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1 + k]; ii < A_diag_i[bidxm1 + k + 1]; ii++)
         {
            jj = A_diag_j[ii];
            if (jj >= bidxm1 && jj < bidxp1 && hypre_cabs(A_diag_data[ii]) > HYPRE_REAL_MIN)
            {
               bidx = i * nb2 + k * blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }
   }

   for (i = 0; i < left_size; i++)
   {
      bidxm1 = n_block * nb2 + i * blk_size;
      bidxp1 = n_block * nb2 + (i + 1) * blk_size;
      for (j = 0; j < left_size; j++)
      {
         bidx = n_block * nb2 + i * blk_size + j;
         diaginv[bidx] = 0.0;
      }

      for (ii = A_diag_i[n_block * blk_size + i]; ii < A_diag_i[n_block * blk_size + i + 1]; ii++)
      {
         jj = A_diag_j[ii];
         if (jj > n_block * blk_size)
         {
            bidx = n_block * nb2 + i * blk_size + jj - n_block * blk_size;
            diaginv[bidx] = A_diag_data[ii];
         }
      }
   }

   /*-----------------------------------------------------------------
   * compute the inverses of all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   if (blk_size > 1)
   {
      for (i = 0; i < n_block; i++)
      {
         hypre_blas_mat_inv(diaginv + i * nb2, blk_size);
      }
      hypre_blas_mat_inv(diaginv + (HYPRE_Int)(blk_size * nb2), left_size);
   }
   else
   {
      for (i = 0; i < n; i++)
      {
         /* TODO: zero-diagonal should be tested previously */
         if (hypre_cabs(diaginv[i]) < HYPRE_REAL_MIN)
         {
            diaginv[i] = 0.0;
         }
         else
         {
            diaginv[i] = 1.0 / diaginv[i];
         }
      }
   }

   *diaginvptr = diaginv;
#endif
   return hypre_error_flag;
}
#if 0
HYPRE_Int
hypre_blockRelax(hypre_ParCSRMatrix *A,
                 hypre_ParVector    *f,
                 hypre_ParVector    *u,
                 HYPRE_Int          blk_size,
                 HYPRE_Int          reserved_coarse_size,
                 HYPRE_Int          method,
                 hypre_ParVector    *Vtemp,
                 hypre_ParVector    *Ztemp)
{
   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real     *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             i, j, k;
   HYPRE_Int             ii, jj;

   HYPRE_Int             bidx, bidxm1, bidxp1;

   HYPRE_Int         num_procs, my_id;

   const HYPRE_Int     nb2 = blk_size * blk_size;
   HYPRE_Int           n_block;
   HYPRE_Int           left_size, inv_size;
   HYPRE_Real          *diaginv;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   //HYPRE_Int num_threads = hypre_NumThreads();

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size * n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size * n_block;
   }

   inv_size  = nb2 * n_block + left_size * left_size;

   diaginv = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);
   /*-----------------------------------------------------------------
   * Get all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      bidxm1 = i * blk_size;
      bidxp1 = (i + 1) * blk_size;
      //printf("bidxm1 = %d,bidxp1 = %d\n",bidxm1,bidxp1);

      for (k = 0; k < blk_size; k++)
      {
         for (j = 0; j < blk_size; j++)
         {
            bidx = i * nb2 + k * blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1 + k]; ii < A_diag_i[bidxm1 + k + 1]; ii++)
         {
            jj = A_diag_j[ii];

            if (jj >= bidxm1 && jj < bidxp1 && hypre_abs(A_diag_data[ii]) > HYPRE_REAL_MIN)
            {
               bidx = i * nb2 + k * blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }
   }

   for (i = 0; i < left_size; i++)
   {
      bidxm1 = n_block * nb2 + i * blk_size;
      bidxp1 = n_block * nb2 + (i + 1) * blk_size;
      for (j = 0; j < left_size; j++)
      {
         bidx = n_block * nb2 + i * blk_size + j;
         diaginv[bidx] = 0.0;
      }

      for (ii = A_diag_i[n_block * blk_size + i]; ii < A_diag_i[n_block * blk_size + i + 1]; ii++)
      {
         jj = A_diag_j[ii];
         if (jj > n_block * blk_size)
         {
            bidx = n_block * nb2 + i * blk_size + jj - n_block * blk_size;
            diaginv[bidx] = A_diag_data[ii];
         }
      }
   }
   /*
   for (i = 0;i < n_block; i++)
   {
     for (j = 0;j < blk_size; j++)
     {
       for (k = 0;k < blk_size; k ++)
       {
         bidx = i*nb2 + j*blk_size + k;
         printf("%e\t",diaginv[bidx]);
       }
       printf("\n");
     }
     printf("\n");
   }
   */
   /*-----------------------------------------------------------------
   * compute the inverses of all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   if (blk_size > 1)
   {
      for (i = 0; i < n_block; i++)
      {
         hypre_blas_mat_inv(diaginv + i * nb2, blk_size);
      }
      hypre_blas_mat_inv(diaginv + (HYPRE_Int)(blk_size * nb2), left_size);
      /*
      for (i = 0;i < n_block; i++)
      {
        for (j = 0;j < blk_size; j++)
        {
          for (k = 0;k < blk_size; k ++)
          {
            bidx = i*nb2 + j*blk_size + k;
            printf("%e\t",diaginv[bidx]);
          }
          printf("\n");
        }
        printf("\n");
      }
      */
   }
   else
   {
      for (i = 0; i < n; i++)
      {
         // FIX-ME: zero-diagonal should be tested previously
         if (hypre_abs(diaginv[i]) < HYPRE_REAL_MIN)
         {
            diaginv[i] = 0.0;
         }
         else
         {
            diaginv[i] = 1.0 / diaginv[i];
         }
      }
   }

   hypre_MGRBlockRelaxSolve(A, f, u, blk_size, n_block, left_size, method, diaginv, Vtemp);

   /*-----------------------------------------------------------------
   * Free temporary memory
   *-----------------------------------------------------------------*/
   hypre_TFree(diaginv, HYPRE_MEMORY_HOST);

   return (hypre_error_flag);
}
#endif

/*--------------------------------------------------------------------------
 * hypre_MGRSetFSolver
 *
 * set F-relaxation solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRSetFSolver( void  *mgr_vdata,
                     HYPRE_Int  (*fine_grid_solver_solve)(void*, void*, void*, void*),
                     HYPRE_Int  (*fine_grid_solver_setup)(void*, void*, void*, void*),
                     void       *fsolver )
{
   hypre_ParMGRData *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   HYPRE_Solver **aff_solver = (mgr_data -> aff_solver);

   if (aff_solver == NULL)
   {
      aff_solver = hypre_CTAlloc(HYPRE_Solver*, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }

   /* only allow to set F-solver for the first level */
   aff_solver[0] = (HYPRE_Solver *) fsolver;

   (mgr_data -> fine_grid_solver_solve) = fine_grid_solver_solve;
   (mgr_data -> fine_grid_solver_setup) = fine_grid_solver_setup;
   (mgr_data -> aff_solver) = aff_solver;
   (mgr_data -> fsolver_mode) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRSetFSolverAtLevel
 *
 * set F-relaxation solver for a given MGR level.
 *
 * Note this function asks for a level identifier and doesn't expect an array
 * of function pointers for each level (as done by SetLevel functions).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRSetFSolverAtLevel( HYPRE_Int   level,
                            void       *mgr_vdata,
                            void       *fsolver )
{
   hypre_ParMGRData *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   HYPRE_Int        max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   HYPRE_Solver   **aff_solver = (mgr_data -> aff_solver);

   /* Check if the requested level makes sense */
   if (level < 0 || level >= max_num_coarse_levels)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   /* Allocate aff_solver if needed */
   if (!aff_solver)
   {
      (mgr_data -> aff_solver) = aff_solver = hypre_CTAlloc(HYPRE_Solver*,
                                                            max_num_coarse_levels,
                                                            HYPRE_MEMORY_HOST);
   }

   aff_solver[level] = (HYPRE_Solver *) fsolver;
   (mgr_data -> fsolver_mode)  = 0;

   return hypre_error_flag;
}

/* set coarse grid solver */
HYPRE_Int
hypre_MGRSetCoarseSolver( void  *mgr_vdata,
                          HYPRE_Int  (*coarse_grid_solver_solve)(void*, void*, void*, void*),
                          HYPRE_Int  (*coarse_grid_solver_setup)(void*, void*, void*, void*),
                          void  *coarse_grid_solver )
{
   hypre_ParMGRData *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   (mgr_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
   (mgr_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
   (mgr_data -> coarse_grid_solver)       = (HYPRE_Solver) coarse_grid_solver;

   (mgr_data -> use_default_cgrid_solver) = 0;

   return hypre_error_flag;
}

/* Set the maximum number of coarse levels.
 * maxcoarselevs = 1 yields the default 2-grid scheme.
*/
HYPRE_Int
hypre_MGRSetMaxCoarseLevels( void *mgr_vdata, HYPRE_Int maxcoarselevs )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> max_num_coarse_levels) = maxcoarselevs;
   return hypre_error_flag;
}

/* Set the system block size */
HYPRE_Int
hypre_MGRSetBlockSize( void *mgr_vdata, HYPRE_Int bsize )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> block_size) = bsize;
   return hypre_error_flag;
}

/* Set the relaxation type for the fine levels of the reduction.
 * Currently supports the following flavors of relaxation types
 * as described in the documentation:
 * relax_types 0 - 8, 13, 14, 18, 19, 98.
 * See par_relax.c and par_relax_more.c for more details.
 * */
HYPRE_Int
hypre_MGRSetRelaxType( void *mgr_vdata, HYPRE_Int relax_type )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> relax_type) = relax_type;
   return hypre_error_flag;
}

/* Set the number of relaxation sweeps */
HYPRE_Int
hypre_MGRSetNumRelaxSweeps( void *mgr_vdata, HYPRE_Int nsweeps )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree(mgr_data -> num_relax_sweeps, HYPRE_MEMORY_HOST);
   HYPRE_Int *num_relax_sweeps = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                               HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      num_relax_sweeps[i] = nsweeps;
   }
   (mgr_data -> num_relax_sweeps) = num_relax_sweeps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRSetLevelNumRelaxSweeps( void *mgr_vdata, HYPRE_Int *level_nsweeps )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree(mgr_data -> num_relax_sweeps, HYPRE_MEMORY_HOST);

   HYPRE_Int *num_relax_sweeps = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                               HYPRE_MEMORY_HOST);
   if (level_nsweeps != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         num_relax_sweeps[i] = level_nsweeps[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         num_relax_sweeps[i] = 0;
      }
   }
   (mgr_data -> num_relax_sweeps) = num_relax_sweeps;

   return hypre_error_flag;
}

/* Set the order of the global smoothing step at each level
 * 1=Down cycle/ Pre-smoothing (default)
 * 2=Up cycle/ Post-smoothing
 */
HYPRE_Int
hypre_MGRSetGlobalSmoothCycle( void *mgr_vdata, HYPRE_Int smooth_cycle )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> global_smooth_cycle) = smooth_cycle;
   return hypre_error_flag;
}

/* Set the F-relaxation strategy: 0=single level, 1=multi level */
HYPRE_Int
hypre_MGRSetFRelaxMethod( void *mgr_vdata, HYPRE_Int relax_method )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree(mgr_data -> Frelax_method, HYPRE_MEMORY_HOST);
   HYPRE_Int *Frelax_method = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      Frelax_method[i] = relax_method;
   }
   (mgr_data -> Frelax_method) = Frelax_method;
   return hypre_error_flag;
}

/* Set the F-relaxation strategy: 0=single level, 1=multi level */
/* This will be removed later. Use SetLevelFrelaxType */
HYPRE_Int
hypre_MGRSetLevelFRelaxMethod( void *mgr_vdata, HYPRE_Int *relax_method )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree(mgr_data -> Frelax_method, HYPRE_MEMORY_HOST);

   HYPRE_Int *Frelax_method = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (relax_method != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_method[i] = relax_method[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_method[i] = 0;
      }
   }
   (mgr_data -> Frelax_method) = Frelax_method;
   return hypre_error_flag;
}

/* Set the F-relaxation type:
 * 0: Jacobi
 * 1: Vcycle smoother
 * 2: AMG
 * Otherwise: use standard BoomerAMGRelax options
*/
HYPRE_Int
hypre_MGRSetLevelFRelaxType( void *mgr_vdata, HYPRE_Int *relax_type )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree(mgr_data -> Frelax_type, HYPRE_MEMORY_HOST);

   HYPRE_Int *Frelax_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (relax_type != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_type[i] = relax_type[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_type[i] = 0;
      }
   }
   (mgr_data -> Frelax_type) = Frelax_type;
   return hypre_error_flag;
}

/* Coarse grid method: 0=Galerkin RAP, 1=non-Galerkin with dropping */
HYPRE_Int
hypre_MGRSetCoarseGridMethod( void *mgr_vdata, HYPRE_Int *cg_method )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);

   hypre_TFree(mgr_data -> mgr_coarse_grid_method, HYPRE_MEMORY_HOST);
   HYPRE_Int *mgr_coarse_grid_method = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                                     HYPRE_MEMORY_HOST);
   if (cg_method != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         mgr_coarse_grid_method[i] = cg_method[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         mgr_coarse_grid_method[i] = 0;
      }
   }
   (mgr_data -> mgr_coarse_grid_method) = mgr_coarse_grid_method;
   return hypre_error_flag;
}

/* Set the F-relaxation number of functions for each level */
HYPRE_Int
hypre_MGRSetLevelFRelaxNumFunctions( void *mgr_vdata, HYPRE_Int *num_functions )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);

   hypre_TFree(mgr_data -> Frelax_num_functions, HYPRE_MEMORY_HOST);

   HYPRE_Int *Frelax_num_functions = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels,
                                                   HYPRE_MEMORY_HOST);
   if (num_functions != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_num_functions[i] = num_functions[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_num_functions[i] = 1;
      }
   }
   (mgr_data -> Frelax_num_functions) = Frelax_num_functions;
   return hypre_error_flag;
}

/* Set the type of the restriction type
 * for computing restriction operator
*/
HYPRE_Int
hypre_MGRSetLevelRestrictType( void *mgr_vdata, HYPRE_Int *restrict_type)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree((mgr_data -> restrict_type), HYPRE_MEMORY_HOST);

   HYPRE_Int *level_restrict_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (restrict_type != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_restrict_type[i] = *(restrict_type + i);
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_restrict_type[i] = 0;
      }
   }
   (mgr_data -> restrict_type) = level_restrict_type;
   return hypre_error_flag;
}

/* Set the type of the restriction type
 * for computing restriction operator
*/
HYPRE_Int
hypre_MGRSetRestrictType( void *mgr_vdata, HYPRE_Int restrict_type)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> restrict_type) != NULL)
   {
      hypre_TFree((mgr_data -> restrict_type), HYPRE_MEMORY_HOST);
      (mgr_data -> restrict_type) = NULL;
   }
   HYPRE_Int *level_restrict_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      level_restrict_type[i] = restrict_type;
   }
   (mgr_data -> restrict_type) = level_restrict_type;
   return hypre_error_flag;
}

/* Set the number of Jacobi interpolation iterations
 * for computing interpolation operator
*/
HYPRE_Int
hypre_MGRSetNumRestrictSweeps( void *mgr_vdata, HYPRE_Int nsweeps )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> num_restrict_sweeps) = nsweeps;
   return hypre_error_flag;
}

/* Set the type of the interpolation
 * for computing interpolation operator
*/
HYPRE_Int
hypre_MGRSetInterpType( void *mgr_vdata, HYPRE_Int interpType)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> interp_type) != NULL)
   {
      hypre_TFree((mgr_data -> interp_type), HYPRE_MEMORY_HOST);
      (mgr_data -> interp_type) = NULL;
   }
   HYPRE_Int *level_interp_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      level_interp_type[i] = interpType;
   }
   (mgr_data -> interp_type) = level_interp_type;
   return hypre_error_flag;
}

/* Set the type of the interpolation
 * for computing interpolation operator
*/
HYPRE_Int
hypre_MGRSetLevelInterpType( void *mgr_vdata, HYPRE_Int *interpType)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree((mgr_data -> interp_type), HYPRE_MEMORY_HOST);

   HYPRE_Int *level_interp_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (interpType != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_interp_type[i] = *(interpType + i);
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_interp_type[i] = 2;
      }
   }
   (mgr_data -> interp_type) = level_interp_type;
   return hypre_error_flag;
}

/* Set the number of Jacobi interpolation iterations
 * for computing interpolation operator
*/
HYPRE_Int
hypre_MGRSetNumInterpSweeps( void *mgr_vdata, HYPRE_Int nsweeps )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> num_interp_sweeps) = nsweeps;
   return hypre_error_flag;
}

/* Set the threshold to truncate the coarse grid at each
 * level of reduction
*/
HYPRE_Int
hypre_MGRSetTruncateCoarseGridThreshold( void *mgr_vdata, HYPRE_Real threshold)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> truncate_coarse_grid_threshold) = threshold;
   return hypre_error_flag;
}

/* Set block size for block Jacobi Interp/Relax */
HYPRE_Int
hypre_MGRSetBlockJacobiBlockSize( void *mgr_vdata, HYPRE_Int blk_size)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> block_jacobi_bsize) = blk_size;
   return hypre_error_flag;
}

/* Set print level for F-relaxation solver */
HYPRE_Int
hypre_MGRSetFrelaxPrintLevel( void *mgr_vdata, HYPRE_Int print_level )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> frelax_print_level) = print_level;
   return hypre_error_flag;
}

/* Set print level for coarse grid solver */
HYPRE_Int
hypre_MGRSetCoarseGridPrintLevel( void *mgr_vdata, HYPRE_Int print_level )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> cg_print_level) = print_level;
   return hypre_error_flag;
}

/* Set print level for mgr solver */
HYPRE_Int
hypre_MGRSetPrintLevel( void *mgr_vdata, HYPRE_Int print_level )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   /* Unset reserved bits if any are active */
   (mgr_data -> print_level) = print_level & ~(HYPRE_MGR_PRINT_RESERVED_A |
                                               HYPRE_MGR_PRINT_RESERVED_B |
                                               HYPRE_MGR_PRINT_RESERVED_C);
   return hypre_error_flag;
}

/* Set logging level for mgr solver */
HYPRE_Int
hypre_MGRSetLogging( void *mgr_vdata, HYPRE_Int logging )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> logging) = logging;
   return hypre_error_flag;
}

/* Set max number of iterations for mgr solver */
HYPRE_Int
hypre_MGRSetMaxIter( void *mgr_vdata, HYPRE_Int max_iter )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> max_iter) = max_iter;
   return hypre_error_flag;
}

/* Set convergence tolerance for mgr solver */
HYPRE_Int
hypre_MGRSetTol( void *mgr_vdata, HYPRE_Real tol )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> tol) = tol;
   return hypre_error_flag;
}

/* Set max number of iterations for mgr global smoother */
HYPRE_Int
hypre_MGRSetMaxGlobalSmoothIters( void *mgr_vdata, HYPRE_Int max_iter )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> level_smooth_iters) != NULL)
   {
      hypre_TFree((mgr_data -> level_smooth_iters), HYPRE_MEMORY_HOST);
      (mgr_data -> level_smooth_iters) = NULL;
   }
   HYPRE_Int *level_smooth_iters = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (max_num_coarse_levels > 0)
   {
      level_smooth_iters[0] = max_iter;
   }
   (mgr_data -> level_smooth_iters) = level_smooth_iters;

   return hypre_error_flag;
}

/* Set global smoothing type for mgr solver */
HYPRE_Int
hypre_MGRSetGlobalSmoothType( void *mgr_vdata, HYPRE_Int gsmooth_type )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> level_smooth_type) != NULL)
   {
      hypre_TFree((mgr_data -> level_smooth_type), HYPRE_MEMORY_HOST);
      (mgr_data -> level_smooth_type) = NULL;
   }
   HYPRE_Int *level_smooth_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (max_num_coarse_levels > 0)
   {
      level_smooth_type[0] = gsmooth_type;
   }
   (mgr_data -> level_smooth_type) = level_smooth_type;

   return hypre_error_flag;
}

/* Set global smoothing type for mgr solver */
HYPRE_Int
hypre_MGRSetLevelSmoothType( void *mgr_vdata, HYPRE_Int *gsmooth_type )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree((mgr_data -> level_smooth_type), HYPRE_MEMORY_HOST);

   HYPRE_Int *level_smooth_type = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (gsmooth_type != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_type[i] = gsmooth_type[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_type[i] = 0;
      }
   }
   (mgr_data -> level_smooth_type) = level_smooth_type;
   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRSetLevelSmoothIters( void *mgr_vdata, HYPRE_Int *gsmooth_iters )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int i;
   HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   hypre_TFree((mgr_data -> level_smooth_iters), HYPRE_MEMORY_HOST);

   HYPRE_Int *level_smooth_iters = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   if (gsmooth_iters != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_iters[i] = gsmooth_iters[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_iters[i] = 0;
      }
   }
   (mgr_data -> level_smooth_iters) = level_smooth_iters;
   return hypre_error_flag;
}

/* Set the maximum number of non-zero entries for interpolation operators */
HYPRE_Int
hypre_MGRSetPMaxElmts(void *mgr_vdata, HYPRE_Int P_max_elmts)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int           max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   HYPRE_Int           i;

   /* Allocate internal P_max_elmts if needed */
   if (!(mgr_data -> P_max_elmts))
   {
      (mgr_data -> P_max_elmts) = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }

   /* Set all P_max_elmts entries to the value passed as input */
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      (mgr_data -> P_max_elmts)[i] = P_max_elmts;
   }

   return hypre_error_flag;
}

/* Set the maximum number of non-zero entries for interpolation operators per level */
HYPRE_Int
hypre_MGRSetLevelPMaxElmts(void *mgr_vdata, HYPRE_Int *P_max_elmts)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int           max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   HYPRE_Int           i;

   /* Allocate internal P_max_elmts if needed */
   if (!(mgr_data -> P_max_elmts))
   {
      (mgr_data -> P_max_elmts) = hypre_CTAlloc(HYPRE_Int, max_num_coarse_levels, HYPRE_MEMORY_HOST);
   }

   /* Set all P_max_elmts entries to the value passed as input */
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      (mgr_data -> P_max_elmts)[i] = (P_max_elmts) ? P_max_elmts[i] : 0;
   }

   return hypre_error_flag;
}

/* Get number of iterations for MGR solver */
HYPRE_Int
hypre_MGRGetNumIterations( void *mgr_vdata, HYPRE_Int *num_iterations )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *num_iterations = mgr_data->num_iterations;

   return hypre_error_flag;
}

/* Get residual norms for MGR solver */
HYPRE_Int
hypre_MGRGetFinalRelativeResidualNorm( void *mgr_vdata, HYPRE_Real *res_norm )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *res_norm = mgr_data->final_rel_residual_norm;

   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRGetCoarseGridConvergenceFactor( void *mgr_vdata, HYPRE_Real *conv_factor )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *conv_factor = (mgr_data -> cg_convergence_factor);

   return hypre_error_flag;
}

/* Build A_FF matrix from A given a CF_marker array */
HYPRE_Int
hypre_MGRGetSubBlock( hypre_ParCSRMatrix   *A,
                      HYPRE_Int            *row_cf_marker,
                      HYPRE_Int            *col_cf_marker,
                      HYPRE_Int             debug_flag,
                      hypre_ParCSRMatrix  **A_block_ptr )
{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd         = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   //HYPRE_Int             *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   hypre_IntArray          *coarse_dof_func_ptr = NULL;
   HYPRE_BigInt            num_row_cpts_global[2];
   HYPRE_BigInt            num_col_cpts_global[2];

   hypre_ParCSRMatrix    *Ablock;
   HYPRE_BigInt         *col_map_offd_Ablock;
   HYPRE_Int       *tmp_map_offd = NULL;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRMatrix    *Ablock_diag;
   hypre_CSRMatrix    *Ablock_offd;

   HYPRE_Real      *Ablock_diag_data;
   HYPRE_Int             *Ablock_diag_i;
   HYPRE_Int             *Ablock_diag_j;
   HYPRE_Real      *Ablock_offd_data;
   HYPRE_Int             *Ablock_offd_i;
   HYPRE_Int             *Ablock_offd_j;

   HYPRE_Int              Ablock_diag_size, Ablock_offd_size;

   HYPRE_Int             *Ablock_marker;

   HYPRE_Int              ii_counter;
   HYPRE_Int              jj_counter, jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd;

   HYPRE_Int              start_indexing = 0; /* start indexing for Aff_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int             *col_coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_BigInt              total_global_row_cpts;
   HYPRE_BigInt              total_global_col_cpts;
   HYPRE_Int              num_cols_Ablock_offd;
   //  HYPRE_BigInt              my_first_row_cpt, my_first_col_cpt;

   HYPRE_Int              i, i1;
   HYPRE_Int              j, jl, jj;
   HYPRE_Int              start;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;
   HYPRE_Int             *int_buf_data;
   HYPRE_Int              local_numrows = hypre_CSRMatrixNumRows(A_diag);

   hypre_IntArray        *wrap_cf;

   //  HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   num_threads = 1;

   /* get the number of coarse rows */
   wrap_cf = hypre_IntArrayCreate(local_numrows);
   hypre_IntArrayMemoryLocation(wrap_cf) = HYPRE_MEMORY_HOST;
   hypre_IntArrayData(wrap_cf) = row_cf_marker;
   hypre_BoomerAMGCoarseParms(comm, local_numrows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_row_cpts_global);
   hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;

   //hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_row_cpts_global[0], num_row_cpts_global[1]);

   //  my_first_row_cpt = num_row_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_row_cpts = num_row_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_row_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* get the number of coarse rows */
   hypre_IntArrayData(wrap_cf) = col_cf_marker;
   hypre_BoomerAMGCoarseParms(comm, local_numrows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_col_cpts_global);
   hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;

   //hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_col_cpts_global[0], num_col_cpts_global[1]);

   //  my_first_col_cpt = num_col_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_col_cpts = num_col_cpts_global[1]; }
   hypre_MPI_Bcast(&total_global_col_cpts, 1, HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   //  if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_cols_A_offd) { CF_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST); }

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                           num_sends), HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         int_buf_data[index++]
            = col_cf_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of Ablock and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   col_coarse_counter = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int, num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int, n_fine, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;

      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a F-point, we loop through the columns and select
          *  the F-columns. Also set up mapping vector.
          *--------------------------------------------------------------------*/

         if (col_cf_marker[i] > 0)
         {
            fine_to_coarse[i] = col_coarse_counter[j];
            col_coarse_counter[j]++;
         }

         if (row_cf_marker[i] > 0)
         {
            //fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (col_cf_marker[i1] > 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (CF_marker_offd[i1] > 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads - 1; i++)
   {
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
      coarse_counter[i + 1] += coarse_counter[i];
      col_coarse_counter[i + 1] += col_coarse_counter[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];
   ii_counter = coarse_counter[i];

   Ablock_diag_size = jj_counter;

   Ablock_diag_i    = hypre_CTAlloc(HYPRE_Int, ii_counter + 1, memory_location);
   Ablock_diag_j    = hypre_CTAlloc(HYPRE_Int, Ablock_diag_size, memory_location);
   Ablock_diag_data = hypre_CTAlloc(HYPRE_Real, Ablock_diag_size, memory_location);

   Ablock_diag_i[ii_counter] = jj_counter;


   Ablock_offd_size = jj_counter_offd;

   Ablock_offd_i    = hypre_CTAlloc(HYPRE_Int, ii_counter + 1, memory_location);
   Ablock_offd_j    = hypre_CTAlloc(HYPRE_Int, Ablock_offd_size, memory_location);
   Ablock_offd_data = hypre_CTAlloc(HYPRE_Real, Ablock_offd_size, memory_location);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   //-----------------------------------------------------------------------
   //  Send and receive fine_to_coarse info.
   //-----------------------------------------------------------------------

   //  if (debug_flag==4) wall_time = time_getWallclockSeconds();
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = col_coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }

   //  if (debug_flag==4) wall_time = time_getWallclockSeconds();
#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   //  for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_col_cpt;

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,jl,i1,jj,ns,ne,size,rest,jj_counter,jj_counter_offd,ii_counter) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }
      ii_counter = 0;
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a F-point, we loop through the columns and select
          *  the F-columns. Also set up mapping vector.
          *--------------------------------------------------------------------*/
         if (row_cf_marker[i] > 0)
         {
            // Diagonal part of Ablock //
            Ablock_diag_i[ii_counter] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (col_cf_marker[i1] > 0)
               {
                  Ablock_diag_j[jj_counter]    = fine_to_coarse[i1];
                  Ablock_diag_data[jj_counter] = A_diag_data[jj];
                  jj_counter++;
               }
            }

            // Off-Diagonal part of Ablock //
            Ablock_offd_i[ii_counter] = jj_counter_offd;
            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (CF_marker_offd[i1] > 0)
                  {
                     Ablock_offd_j[jj_counter_offd]  = i1;
                     Ablock_offd_data[jj_counter_offd] = A_offd_data[jj];
                     jj_counter_offd++;
                  }
               }
            }
            ii_counter++;
         }
      }
      Ablock_offd_i[ii_counter] = jj_counter_offd;
      Ablock_diag_i[ii_counter] = jj_counter;
   }
   Ablock = hypre_ParCSRMatrixCreate(comm,
                                     total_global_row_cpts,
                                     total_global_col_cpts,
                                     num_row_cpts_global,
                                     num_col_cpts_global,
                                     0,
                                     Ablock_diag_i[ii_counter],
                                     Ablock_offd_i[ii_counter]);

   Ablock_diag = hypre_ParCSRMatrixDiag(Ablock);
   hypre_CSRMatrixData(Ablock_diag) = Ablock_diag_data;
   hypre_CSRMatrixI(Ablock_diag) = Ablock_diag_i;
   hypre_CSRMatrixJ(Ablock_diag) = Ablock_diag_j;
   Ablock_offd = hypre_ParCSRMatrixOffd(Ablock);
   hypre_CSRMatrixData(Ablock_offd) = Ablock_offd_data;
   hypre_CSRMatrixI(Ablock_offd) = Ablock_offd_i;
   hypre_CSRMatrixJ(Ablock_offd) = Ablock_offd_j;

   num_cols_Ablock_offd = 0;

   if (Ablock_offd_size)
   {
      Ablock_marker = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         Ablock_marker[i] = 0;
      }
      num_cols_Ablock_offd = 0;
      for (i = 0; i < Ablock_offd_size; i++)
      {
         index = Ablock_offd_j[i];
         if (!Ablock_marker[index])
         {
            num_cols_Ablock_offd++;
            Ablock_marker[index] = 1;
         }
      }

      col_map_offd_Ablock = hypre_CTAlloc(HYPRE_BigInt, num_cols_Ablock_offd, memory_location);
      tmp_map_offd = hypre_CTAlloc(HYPRE_Int, num_cols_Ablock_offd, HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_cols_Ablock_offd; i++)
      {
         while (Ablock_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }
#if 0
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < Ablock_offd_size; i++)
         Ablock_offd_j[i] = hypre_BinarySearch(tmp_map_offd,
                                               Ablock_offd_j[i],
                                               num_cols_Ablock_offd);
      hypre_TFree(Ablock_marker, HYPRE_MEMORY_HOST);
   }

   if (num_cols_Ablock_offd)
   {
      hypre_ParCSRMatrixColMapOffd(Ablock) = col_map_offd_Ablock;
      hypre_CSRMatrixNumCols(Ablock_offd) = num_cols_Ablock_offd;
   }

   hypre_GetCommPkgRTFromCommPkgA(Ablock, A, fine_to_coarse, tmp_map_offd);

   /* Create the assumed partition */
   if (hypre_ParCSRMatrixAssumedPartition(Ablock) == NULL)
   {
      hypre_ParCSRMatrixCreateAssumedPartition(Ablock);
   }

   *A_block_ptr = Ablock;

   hypre_TFree(tmp_map_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
   hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(col_coarse_counter, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
   hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);
   hypre_IntArrayData(wrap_cf) = NULL;
   hypre_IntArrayDestroy(wrap_cf);

   return hypre_error_flag;
}

/* Build A_FF matrix from A given a CF_marker array */
HYPRE_Int
hypre_MGRBuildAff( hypre_ParCSRMatrix   *A,
                   HYPRE_Int            *CF_marker,
                   HYPRE_Int             debug_flag,
                   hypre_ParCSRMatrix  **A_ff_ptr )
{
   HYPRE_Int i;
   HYPRE_Int local_numrows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   /* create a copy of the CF_marker array and switch C-points to F-points */
   HYPRE_Int *CF_marker_copy = hypre_CTAlloc(HYPRE_Int, local_numrows, HYPRE_MEMORY_HOST);

#if 0
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < local_numrows; i++)
   {
      CF_marker_copy[i] = -CF_marker[i];
   }

   hypre_MGRGetSubBlock(A, CF_marker_copy, CF_marker_copy, debug_flag, A_ff_ptr);

   /* Free copy of CF marker */
   hypre_TFree(CF_marker_copy, HYPRE_MEMORY_HOST);
   return (0);
}

/*********************************************************************************
 * This routine assumes that the 'toVector' is larger than the 'fromVector' and
 * the CF_marker is of the same length as the toVector. There must be n 'point_type'
 * values in the CF_marker, where n is the length of the 'fromVector'.
 * It adds the values of the 'fromVector' to the 'toVector' where the marker is the
 * same as the 'point_type'
 *********************************************************************************/
HYPRE_Int
hypre_MGRAddVectorP ( hypre_IntArray  *CF_marker,
                      HYPRE_Int        point_type,
                      HYPRE_Real       a,
                      hypre_ParVector  *fromVector,
                      HYPRE_Real       b,
                      hypre_ParVector  **toVector )
{
   hypre_Vector    *fromVectorLocal = hypre_ParVectorLocalVector(fromVector);
   HYPRE_Real      *fromVectorData  = hypre_VectorData(fromVectorLocal);
   hypre_Vector    *toVectorLocal   = hypre_ParVectorLocalVector(*toVector);
   HYPRE_Real      *toVectorData    = hypre_VectorData(toVectorLocal);
   HYPRE_Int       *CF_marker_data = hypre_IntArrayData(CF_marker);

   //HYPRE_Int       n = hypre_ParVectorActualLocalSize(*toVector);
   HYPRE_Int       n = hypre_IntArraySize(CF_marker);
   HYPRE_Int       i, j;

   j = 0;
   for (i = 0; i < n; i++)
   {
      if (CF_marker_data[i] == point_type)
      {
         toVectorData[i] = b * toVectorData[i] + a * fromVectorData[j];
         j++;
      }
   }
   return 0;
}

/*************************************************************************************
 * This routine assumes that the 'fromVector' is larger than the 'toVector' and
 * the CF_marker is of the same length as the fromVector. There must be n 'point_type'
 * values in the CF_marker, where n is the length of the 'toVector'.
 * It adds the values of the 'fromVector' where the marker is the
 * same as the 'point_type' to the 'toVector'
 *************************************************************************************/
HYPRE_Int
hypre_MGRAddVectorR ( hypre_IntArray *CF_marker,
                      HYPRE_Int        point_type,
                      HYPRE_Real       a,
                      hypre_ParVector  *fromVector,
                      HYPRE_Real       b,
                      hypre_ParVector  **toVector )
{
   hypre_Vector    *fromVectorLocal = hypre_ParVectorLocalVector(fromVector);
   HYPRE_Real      *fromVectorData  = hypre_VectorData(fromVectorLocal);
   hypre_Vector    *toVectorLocal   = hypre_ParVectorLocalVector(*toVector);
   HYPRE_Real      *toVectorData    = hypre_VectorData(toVectorLocal);
   HYPRE_Int       *CF_marker_data = hypre_IntArrayData(CF_marker);

   //HYPRE_Int       n = hypre_ParVectorActualLocalSize(*toVector);
   HYPRE_Int       n = hypre_IntArraySize(CF_marker);
   HYPRE_Int       i, j;

   j = 0;
   for (i = 0; i < n; i++)
   {
      if (CF_marker_data[i] == point_type)
      {
         toVectorData[j] = b * toVectorData[j] + a * fromVectorData[i];
         j++;
      }
   }
   return 0;
}

/*
HYPRE_Int
hypre_MGRBuildAffRAP( MPI_Comm comm, HYPRE_Int local_num_variables, HYPRE_Int num_functions,
  HYPRE_Int *dof_func, HYPRE_Int *CF_marker, HYPRE_Int **coarse_dof_func_ptr, HYPRE_BigInt **coarse_pnts_global_ptr,
  hypre_ParCSRMatrix *A, HYPRE_Int debug_flag, hypre_ParCSRMatrix **P_f_ptr, hypre_ParCSRMatrix **A_ff_ptr )
{
  HYPRE_Int *CF_marker_copy = hypre_CTAlloc(HYPRE_Int,  local_num_variables, HYPRE_MEMORY_HOST);
  HYPRE_Int i;
  for (i = 0; i < local_num_variables; i++) {
    CF_marker_copy[i] = -CF_marker[i];
  }

  hypre_BoomerAMGCoarseParms(comm, local_num_variables, 1, NULL, CF_marker_copy, coarse_dof_func_ptr, coarse_pnts_global_ptr);
  hypre_MGRBuildP(A, CF_marker_copy, (*coarse_pnts_global_ptr), 0, debug_flag, P_f_ptr);
  hypre_BoomerAMGBuildCoarseOperator(*P_f_ptr, A, *P_f_ptr, A_ff_ptr);

  hypre_TFree(CF_marker_copy, HYPRE_MEMORY_HOST);
  return 0;
}
*/

/* Get pointer to coarse grid matrix for MGR solver */
HYPRE_Int
hypre_MGRGetCoarseGridMatrix( void *mgr_vdata, hypre_ParCSRMatrix **RAP )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (mgr_data -> RAP == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        " Coarse grid matrix is NULL. Please make sure MGRSetup() is called \n");
      return hypre_error_flag;
   }
   *RAP = mgr_data->RAP;

   return hypre_error_flag;
}

/* Get pointer to coarse grid solution for MGR solver */
HYPRE_Int
hypre_MGRGetCoarseGridSolution( void *mgr_vdata, hypre_ParVector **sol )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (mgr_data -> U_array == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        " MGR solution array is NULL. Please make sure MGRSetup() and MGRSolve() are called \n");
      return hypre_error_flag;
   }
   *sol = mgr_data->U_array[mgr_data->num_coarse_levels];

   return hypre_error_flag;
}

/* Get pointer to coarse grid solution for MGR solver */
HYPRE_Int
hypre_MGRGetCoarseGridRHS( void *mgr_vdata, hypre_ParVector **rhs )
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (mgr_data -> F_array == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        " MGR RHS array is NULL. Please make sure MGRSetup() and MGRSolve() are called \n");
      return hypre_error_flag;
   }
   *rhs = mgr_data->F_array[mgr_data->num_coarse_levels];

   return hypre_error_flag;
}

/* Print coarse grid linear system (for debugging)*/
HYPRE_Int
hypre_MGRPrintCoarseSystem( void *mgr_vdata, HYPRE_Int print_flag)
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   mgr_data->print_coarse_system = print_flag;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_MGRDataPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MGRDataPrint(void *mgr_vdata)
{
   hypre_ParMGRData     *mgr_data           = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int             print_level        = (mgr_data -> print_level);
   HYPRE_Int             num_coarse_levels  = (mgr_data -> num_coarse_levels);
   hypre_ParCSRMatrix  **A_array            = (mgr_data -> A_array);
   hypre_ParCSRMatrix  **P_array            = (mgr_data -> P_array);
   hypre_ParCSRMatrix  **RT_array           = (mgr_data -> RT_array);
   hypre_ParCSRMatrix   *A_coarsest         = (mgr_data -> RAP);
   hypre_ParVector     **f_array            = (mgr_data -> F_array);
   HYPRE_Int            *point_marker_array = (mgr_data -> point_marker_array);
   HYPRE_Int             block_size         = (mgr_data -> block_size);
   char                 *data_path          = (mgr_data -> data_path);

   char                  topdir[] = "./hypre-data";
   char                 *filename = NULL;
   hypre_IntArray       *dofmap   = NULL;
   MPI_Comm              comm;
   HYPRE_Int             myid, lvl;
   HYPRE_Int             data_path_length = 0;

   /* Sanity check */
   if (!A_array[0])
   {
      return hypre_error_flag;
   }

   /* Get rank ID */
   comm = hypre_ParCSRMatrixComm(A_array[0]);
   hypre_MPI_Comm_rank(comm, &myid);

   /* Create new "ls_" folder (data_path) */
   if (((print_level & HYPRE_MGR_PRINT_INFO_PARAMS) ||
        (print_level & HYPRE_MGR_PRINT_FINE_MATRIX) ||
        (print_level & HYPRE_MGR_PRINT_FINE_RHS)    ||
        (print_level & HYPRE_MGR_PRINT_CRSE_MATRIX) ||
        (print_level & HYPRE_MGR_PRINT_LVLS_MATRIX) )   &&
       (data_path == NULL))
   {
      if (!myid)
      {
         if (!hypre_CheckDirExists(topdir))
         {
            hypre_CreateDir(topdir);
         }

         hypre_CreateNextDirOfSequence(topdir, "ls_", &data_path);
         data_path_length = strlen(data_path) + 1;
      }
      hypre_MPI_Bcast(&data_path_length, 1, HYPRE_MPI_INT, 0, comm);

      if (data_path_length > 0)
      {
         if (myid)
         {
            data_path = hypre_TAlloc(char, data_path_length, HYPRE_MEMORY_HOST);
         }
      }
      else
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unable to create data path!");
         return hypre_error_flag;
      }
      hypre_MPI_Bcast(data_path, data_path_length, hypre_MPI_CHAR, 0, comm);

      /* Save data_path */
      (mgr_data -> data_path) = data_path;
   }
   else
   {
      if (data_path)
      {
         data_path_length = strlen(data_path);
      }
   }

   /* Allocate memory for filename */
   filename = hypre_TAlloc(char, data_path_length + 16, HYPRE_MEMORY_HOST);

   /* Print MGR parameters to file */
   if (print_level & HYPRE_MGR_PRINT_INFO_PARAMS)
   {
      /* TODO (VPM): print internal MGR parameters to file */

      /* Signal that the MGR parameters have already been printed */
      (mgr_data -> print_level) &= ~HYPRE_MGR_PRINT_INFO_PARAMS;
      (mgr_data -> print_level) |= HYPRE_MGR_PRINT_RESERVED_A;
   }

   /* Print linear system matrix at the finest level and dofmap */
   if ((print_level & (HYPRE_MGR_PRINT_FINE_MATRIX + HYPRE_MGR_PRINT_LVLS_MATRIX)) && A_array[0])
   {
      /* Build dofmap array */
      dofmap = hypre_IntArrayCreate(hypre_ParCSRMatrixNumRows(A_array[0]));
      hypre_IntArrayInitialize_v2(dofmap, HYPRE_MEMORY_HOST);
      if (point_marker_array)
      {
         hypre_TMemcpy(hypre_IntArrayData(dofmap), point_marker_array,
                       HYPRE_Int, hypre_ParCSRMatrixNumRows(A_array[0]),
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      else
      {
         hypre_IntArraySetInterleavedValues(dofmap, block_size);
      }

      /* Print dofmap */
      hypre_ParPrintf(comm, "Writing dofmap to path: %s\n", data_path);
      hypre_sprintf(filename, "%s/dofmap.out", data_path);
      hypre_IntArrayPrint(comm, dofmap, filename);

      /* Free memory */
      hypre_IntArrayDestroy(dofmap);

      /* Print Matrix */
      hypre_ParPrintf(comm, "Writing fine level matrix to path: %s\n", data_path);
      hypre_sprintf(filename, "%s/IJ.out.A", data_path);
      if (print_level & HYPRE_MGR_PRINT_MODE_ASCII)
      {
         hypre_ParCSRMatrixPrintIJ(A_array[0], 0, 0, filename);
      }
      else
      {
         hypre_ParCSRMatrixPrintBinaryIJ(A_array[0], 0, 0, filename);
      }

      /* Signal that the matrix has already been printed */
      (mgr_data -> print_level) &= ~HYPRE_MGR_PRINT_FINE_MATRIX;
      (mgr_data -> print_level) |= HYPRE_MGR_PRINT_RESERVED_B;
   }

   /* Print linear system RHS at the finest level */
   if ((print_level & HYPRE_MGR_PRINT_FINE_RHS) && f_array[0])
   {
      /* Print RHS */
      hypre_ParPrintf(comm, "Writing RHS to path: %s\n", data_path);
      hypre_sprintf(filename, "%s/IJ.out.b", data_path);
      if (print_level & HYPRE_MGR_PRINT_MODE_ASCII)
      {
         hypre_ParVectorPrintIJ(f_array[0], 0, filename);
      }
      else
      {
         hypre_ParVectorPrintBinaryIJ(f_array[0], filename);
      }

      /* Free memory */
      hypre_TFree(filename, HYPRE_MEMORY_HOST);

      /* Signal that the vector has already been printed */
      (mgr_data -> print_level) &= ~HYPRE_MGR_PRINT_FINE_RHS;
      (mgr_data -> print_level) |= HYPRE_MGR_PRINT_RESERVED_C;
   }

   /* Print linear system matrix at the coarsest level */
   if ((print_level & (HYPRE_MGR_PRINT_CRSE_MATRIX + HYPRE_MGR_PRINT_LVLS_MATRIX)) && A_coarsest)
   {
      hypre_ParPrintf(comm, "Writing coarsest level matrix to path: %s\n", data_path);
      hypre_sprintf(filename, "%s/IJ.out.A.%02d", data_path, num_coarse_levels);
      if (print_level & HYPRE_MGR_PRINT_MODE_ASCII)
      {
         hypre_ParCSRMatrixPrintIJ(A_coarsest, 0, 0, filename);
      }
      else
      {
         hypre_ParCSRMatrixPrintBinaryIJ(A_coarsest, 0, 0, filename);
      }

      /* Signal that the matrix has already been printed */
      (mgr_data -> print_level) &= ~HYPRE_MGR_PRINT_CRSE_MATRIX;
   }

   /* Print MGR hierarchy */
   if ((print_level & HYPRE_MGR_PRINT_LVLS_MATRIX))
   {
      for (lvl = 0; lvl < num_coarse_levels - 1; lvl++)
      {
         /* Print operator matrix */
         hypre_ParPrintf(comm, "Writing level %d matrix to path: %s\n", lvl + 1, data_path);
         hypre_sprintf(filename, "%s/IJ.out.A.%02d", data_path, lvl + 1);
         if (print_level & HYPRE_MGR_PRINT_MODE_ASCII)
         {
            hypre_ParCSRMatrixPrintIJ(A_array[lvl + 1], 0, 0, filename);
         }
         else
         {
            hypre_ParCSRMatrixPrintBinaryIJ(A_array[lvl + 1], 0, 0, filename);
         }

         /* Print interpolation matrix */
         if (P_array[lvl])
         {
            hypre_ParPrintf(comm, "Writing level %d interpolation to path: %s\n", lvl, data_path);
            hypre_sprintf(filename, "%s/IJ.out.P.%02d", data_path, lvl);
            if (print_level & HYPRE_MGR_PRINT_MODE_ASCII)
            {
               hypre_ParCSRMatrixPrintIJ(P_array[lvl], 0, 0, filename);
            }
            else
            {
               hypre_ParCSRMatrixPrintBinaryIJ(P_array[lvl], 0, 0, filename);
            }
         }

         /* Print restriction matrix */
         if (RT_array[lvl])
         {
            hypre_ParPrintf(comm, "Writing level %d restriction to path: %s\n", lvl, data_path);
            hypre_sprintf(filename, "%s/IJ.out.RT.%02d", data_path, lvl);
            if (print_level & HYPRE_MGR_PRINT_MODE_ASCII)
            {
               hypre_ParCSRMatrixPrintIJ(RT_array[lvl], 0, 0, filename);
            }
            else
            {
               hypre_ParCSRMatrixPrintBinaryIJ(RT_array[lvl], 0, 0, filename);
            }
         }
      }

      /* Signal that the data has already been printed */
      (mgr_data -> print_level) &= ~HYPRE_MGR_PRINT_LVLS_MATRIX;
   }

   /* Free memory */
   hypre_TFree(filename, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/***************************************************************************
 ***************************************************************************/

#ifdef HYPRE_USING_DSUPERLU
void *
hypre_MGRDirectSolverCreate()
{
   //   hypre_DSLUData *dslu_data = hypre_CTAlloc(hypre_DSLUData, 1, HYPRE_MEMORY_HOST);
   //   return (void *) dslu_data;
   return NULL;
}

HYPRE_Int
hypre_MGRDirectSolverSetup( void                *solver,
                            hypre_ParCSRMatrix  *A,
                            hypre_ParVector     *f,
                            hypre_ParVector     *u )
{
   HYPRE_Int ierr;
   ierr = hypre_SLUDistSetup( solver, A, 0);

   return ierr;
}
HYPRE_Int
hypre_MGRDirectSolverSolve( void                *solver,
                            hypre_ParCSRMatrix  *A,
                            hypre_ParVector     *f,
                            hypre_ParVector     *u )
{
   hypre_SLUDistSolve(solver, f, u);

   return hypre_error_flag;
}

HYPRE_Int
hypre_MGRDirectSolverDestroy( void *solver )
{
   hypre_SLUDistDestroy(solver);

   return hypre_error_flag;
}
#endif
