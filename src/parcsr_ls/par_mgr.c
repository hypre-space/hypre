/*BHEADER**********************************************************************
 * Copyright (c) 2015,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_mgr.h"
#include <assert.h>

/* Create */
void *
hypre_MGRCreate()
{
  hypre_ParMGRData  *mgr_data;

  mgr_data = hypre_CTAlloc(hypre_ParMGRData,  1, HYPRE_MEMORY_HOST);

  /* block data */
  (mgr_data -> block_size) = 1;
  (mgr_data -> num_coarse_indexes) = 1;
  (mgr_data -> block_num_coarse_indexes) = NULL;
  (mgr_data -> block_cf_marker) = NULL;

  /* general data */
  (mgr_data -> max_num_coarse_levels) = 10;
  (mgr_data -> A_array) = NULL;
  (mgr_data -> P_array) = NULL;
  (mgr_data -> RT_array) = NULL;
  (mgr_data -> RAP) = NULL;
  (mgr_data -> CF_marker_array) = NULL;
  (mgr_data -> coarse_indices_lvls) = NULL;

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
  (mgr_data -> S_commpkg_switch) = 1.0;
  (mgr_data -> P_max_elmts) = 0;

  (mgr_data -> coarse_grid_solver) = NULL;
  (mgr_data -> coarse_grid_solver_setup) = NULL;
  (mgr_data -> coarse_grid_solver_solve) = NULL;

  (mgr_data -> global_smoother) = NULL;

  (mgr_data -> use_default_cgrid_solver) = 1;
  (mgr_data -> omega) = 1.;
  (mgr_data -> max_iter) = 20;
  (mgr_data -> tol) = 1.0e-7;
  (mgr_data -> relax_type) = 0;
  (mgr_data -> relax_order) = 1;
  (mgr_data -> interp_type) = 2;
  (mgr_data -> restrict_type) = 0;
  (mgr_data -> num_relax_sweeps) = 1;
  (mgr_data -> relax_weight) = 1.0;

  (mgr_data -> logging) = 0;
  (mgr_data -> print_level) = 0;

  (mgr_data -> l1_norms) = NULL;

  (mgr_data -> reserved_coarse_size) = 0;
  (mgr_data -> reserved_coarse_indexes) = NULL;
  (mgr_data -> reserved_Cpoint_local_indexes) = NULL;

  (mgr_data -> diaginv) = NULL;
  (mgr_data -> global_smooth_iters) = 1;
  (mgr_data -> global_smooth_type) = 0;

  (mgr_data -> set_non_Cpoints_to_F) = 0;

  (mgr_data -> Frelax_method) = 0;
  (mgr_data -> FrelaxVcycleData) = NULL;
  (mgr_data -> max_local_lvls) = 10;

  (mgr_data -> print_coarse_system) = 0;

  return (void *) mgr_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
HYPRE_Int
hypre_MGRDestroy( void *data )
{
  hypre_ParMGRData * mgr_data = (hypre_ParMGRData*) data;

  HYPRE_Int i;
  HYPRE_Int num_coarse_levels = (mgr_data -> num_coarse_levels);

  /* block info data */
  if ((mgr_data -> block_cf_marker))
  {
    for (i=0; i < (mgr_data -> max_num_coarse_levels); i++)
    {
       if ((mgr_data -> block_cf_marker)[i])
       {
         hypre_TFree((mgr_data -> block_cf_marker)[i], HYPRE_MEMORY_HOST);
       }
    }
    hypre_TFree((mgr_data -> block_cf_marker), HYPRE_MEMORY_HOST);
    (mgr_data -> block_cf_marker) = NULL;
  }

  if(mgr_data -> block_num_coarse_indexes)
  {
     hypre_TFree(mgr_data -> block_num_coarse_indexes, HYPRE_MEMORY_HOST);
     (mgr_data -> block_num_coarse_indexes) = NULL;
  }

  /* final residual vector */
  if((mgr_data -> residual))
  {
    hypre_ParVectorDestroy( (mgr_data -> residual) );
    (mgr_data -> residual) = NULL;
  }
  if((mgr_data -> rel_res_norms))
  {
    hypre_TFree( (mgr_data -> rel_res_norms) , HYPRE_MEMORY_HOST);
    (mgr_data -> rel_res_norms) = NULL;
  }
  /* temp vectors for solve phase */
  if((mgr_data -> Vtemp))
  {
    hypre_ParVectorDestroy( (mgr_data -> Vtemp) );
    (mgr_data -> Vtemp) = NULL;
  }
  if((mgr_data -> Ztemp))
  {
    hypre_ParVectorDestroy( (mgr_data -> Ztemp) );
    (mgr_data -> Ztemp) = NULL;
  }
  if((mgr_data -> Utemp))
  {
    hypre_ParVectorDestroy( (mgr_data -> Utemp) );
    (mgr_data -> Utemp) = NULL;
  }
  if((mgr_data -> Ftemp))
  {
    hypre_ParVectorDestroy( (mgr_data -> Ftemp) );
    (mgr_data -> Ftemp) = NULL;
  }
  /* coarse grid solver */
  if((mgr_data -> use_default_cgrid_solver))
  {
    if((mgr_data -> coarse_grid_solver))
       hypre_BoomerAMGDestroy( (mgr_data -> coarse_grid_solver) );
    (mgr_data -> coarse_grid_solver) = NULL;
  }
  /* l1_norms */
  if ((mgr_data -> l1_norms))
  {
    for (i=0; i < (num_coarse_levels); i++)
       if ((mgr_data -> l1_norms)[i])
         hypre_TFree((mgr_data -> l1_norms)[i], HYPRE_MEMORY_HOST);
    hypre_TFree((mgr_data -> l1_norms), HYPRE_MEMORY_HOST);
  }

  /* coarse_indices_lvls */
  if ((mgr_data -> coarse_indices_lvls))
  {
    for (i=0; i < (num_coarse_levels); i++)
       if ((mgr_data -> coarse_indices_lvls)[i])
         hypre_TFree((mgr_data -> coarse_indices_lvls)[i], HYPRE_MEMORY_HOST);
    hypre_TFree((mgr_data -> coarse_indices_lvls), HYPRE_MEMORY_HOST);
  }

  /* linear system and cf marker array */
  if(mgr_data -> A_array || mgr_data -> P_array || mgr_data -> RT_array || mgr_data -> CF_marker_array)
  {
    for (i=1; i < num_coarse_levels+1; i++) {
      hypre_ParVectorDestroy((mgr_data -> F_array)[i]);
      hypre_ParVectorDestroy((mgr_data -> U_array)[i]);

      if ((mgr_data -> P_array)[i-1])
          hypre_ParCSRMatrixDestroy((mgr_data -> P_array)[i-1]);

      if ((mgr_data -> RT_array)[i-1])
          hypre_ParCSRMatrixDestroy((mgr_data -> RT_array)[i-1]);

      hypre_TFree((mgr_data -> CF_marker_array)[i-1], HYPRE_MEMORY_HOST);
    }
    for (i=1; i < (num_coarse_levels); i++) {
      if ((mgr_data -> A_array)[i])
        hypre_ParCSRMatrixDestroy((mgr_data -> A_array)[i]);
    }
  }

  if((mgr_data -> F_array))
  {
     hypre_TFree((mgr_data -> F_array), HYPRE_MEMORY_HOST);
     (mgr_data -> F_array) = NULL;
  }
  if((mgr_data -> U_array))
  {
     hypre_TFree((mgr_data -> U_array), HYPRE_MEMORY_HOST);
     (mgr_data -> U_array) = NULL;
  }
  if((mgr_data -> A_array))
  {
     hypre_TFree((mgr_data -> A_array), HYPRE_MEMORY_HOST);
     (mgr_data -> A_array) = NULL;
  }
  if((mgr_data -> P_array))
  {
     hypre_TFree((mgr_data -> P_array), HYPRE_MEMORY_HOST);
     (mgr_data -> P_array) = NULL;
  }
  if((mgr_data -> RT_array))
  {
     hypre_TFree((mgr_data -> RT_array), HYPRE_MEMORY_HOST);
     (mgr_data -> RT_array) = NULL;
  }
  if((mgr_data -> CF_marker_array))
  {
     hypre_TFree((mgr_data -> CF_marker_array), HYPRE_MEMORY_HOST);
     (mgr_data -> CF_marker_array) = NULL;
  }
  if((mgr_data -> reserved_Cpoint_local_indexes))
  {
     hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), HYPRE_MEMORY_HOST);
     (mgr_data -> reserved_Cpoint_local_indexes) = NULL;
  }

  /* data for V-cycle F-relaxation */
  if (mgr_data -> FrelaxVcycleData) {
    for (i = 0; i < num_coarse_levels; i++) {
      if ((mgr_data -> FrelaxVcycleData)[i]) {
        hypre_MGRDestroyFrelaxVcycleData((mgr_data -> FrelaxVcycleData)[i]);
        (mgr_data -> FrelaxVcycleData)[i] = NULL;
      }
    }
    hypre_TFree(mgr_data -> FrelaxVcycleData, HYPRE_MEMORY_HOST);
    mgr_data -> FrelaxVcycleData = NULL;
  }
  /* data for reserved coarse nodes */
  if(mgr_data -> reserved_coarse_indexes)
  {
     hypre_TFree(mgr_data -> reserved_coarse_indexes, HYPRE_MEMORY_HOST);
     (mgr_data -> reserved_coarse_indexes) = NULL;
  }
  /* coarse level matrix - RAP */
  if ((mgr_data -> RAP))
    hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
  if ((mgr_data -> diaginv))
    hypre_TFree((mgr_data -> diaginv), HYPRE_MEMORY_HOST);
  /* mgr data */
  hypre_TFree(mgr_data, HYPRE_MEMORY_HOST);

  return hypre_error_flag;
}

/* Create data for V-cycle F-relaxtion */
void *
hypre_MGRCreateFrelaxVcycleData()
{
  hypre_ParAMGData  *vdata = hypre_CTAlloc(hypre_ParAMGData,  1, HYPRE_MEMORY_HOST);

   hypre_ParAMGDataAArray(vdata) = NULL;
   hypre_ParAMGDataPArray(vdata) = NULL;
   hypre_ParAMGDataFArray(vdata) = NULL;
   hypre_ParAMGDataCFMarkerArray(vdata) = NULL;
   hypre_ParAMGDataVtemp(vdata)  = NULL;
   hypre_ParAMGDataAMat(vdata)  = NULL;
   hypre_ParAMGDataBVec(vdata)  = NULL;
   hypre_ParAMGDataZtemp(vdata)  = NULL;
   hypre_ParAMGDataCommInfo(vdata) = NULL;
   hypre_ParAMGDataUArray(vdata) = NULL;
   hypre_ParAMGDataNewComm(vdata) = hypre_MPI_COMM_NULL;
   hypre_ParAMGDataNumLevels(vdata) = 0;
   hypre_ParAMGDataMaxLevels(vdata) = 10;

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

  for (i=1; i < num_levels; i++)
  {
   hypre_ParVectorDestroy(hypre_ParAMGDataFArray(vdata)[i]);
   hypre_ParVectorDestroy(hypre_ParAMGDataUArray(vdata)[i]);

        if (hypre_ParAMGDataAArray(vdata)[i])
           hypre_ParCSRMatrixDestroy(hypre_ParAMGDataAArray(vdata)[i]);

        if (hypre_ParAMGDataPArray(vdata)[i-1])
           hypre_ParCSRMatrixDestroy(hypre_ParAMGDataPArray(vdata)[i-1]);

 hypre_TFree(hypre_ParAMGDataCFMarkerArray(vdata)[i-1], HYPRE_MEMORY_HOST);
  }

  /* see comments in par_coarsen.c regarding special case for CF_marker */
  if (num_levels == 1)
  {
     hypre_TFree(hypre_ParAMGDataCFMarkerArray(vdata)[0], HYPRE_MEMORY_HOST);
  }

/* Points to vtemp of mgr_data, which is already destroyed */
//  hypre_ParVectorDestroy(hypre_ParAMGDataVtemp(vdata));
  hypre_TFree(hypre_ParAMGDataFArray(vdata), HYPRE_MEMORY_HOST);
  hypre_TFree(hypre_ParAMGDataUArray(vdata), HYPRE_MEMORY_HOST);
  hypre_TFree(hypre_ParAMGDataAArray(vdata), HYPRE_MEMORY_HOST);
  hypre_TFree(hypre_ParAMGDataPArray(vdata), HYPRE_MEMORY_HOST);
  hypre_TFree(hypre_ParAMGDataCFMarkerArray(vdata), HYPRE_MEMORY_HOST);

/* Points to ztemp of mgr_data, which is already destroyed */
/*
  if (hypre_ParAMGDataZtemp(vdata))
      hypre_ParVectorDestroy(hypre_ParAMGDataZtemp(vdata));
*/

  if (hypre_ParAMGDataAMat(vdata)) hypre_TFree(hypre_ParAMGDataAMat(vdata), HYPRE_MEMORY_HOST);
  if (hypre_ParAMGDataBVec(vdata)) hypre_TFree(hypre_ParAMGDataBVec(vdata), HYPRE_MEMORY_HOST);
  if (hypre_ParAMGDataCommInfo(vdata)) hypre_TFree(hypre_ParAMGDataCommInfo(vdata), HYPRE_MEMORY_HOST);

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

/* Initialize/ set block data information */
HYPRE_Int
hypre_MGRSetCpointsByBlock( void      *mgr_vdata,
                         HYPRE_Int  block_size,
                         HYPRE_Int  max_num_levels,
                         HYPRE_Int  *block_num_coarse_points,
                         HYPRE_Int  **block_coarse_indexes)
{
  HYPRE_Int  i,j;
  HYPRE_Int  **block_cf_marker = NULL;
  HYPRE_Int *block_num_coarse_indexes = NULL;

  hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;

  /* free block cf_marker data if not previously destroyed */
  if((mgr_data -> block_cf_marker) != NULL)
  {
    for (i=0; i < (mgr_data -> max_num_coarse_levels); i++)
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
   if((mgr_data -> block_num_coarse_indexes))
   {
      hypre_TFree((mgr_data -> block_num_coarse_indexes), HYPRE_MEMORY_HOST);
      (mgr_data -> block_num_coarse_indexes) = NULL;
   }

   /* store block cf_marker */
   block_cf_marker = hypre_CTAlloc(HYPRE_Int *, max_num_levels, HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_levels; i++)
   {
     block_cf_marker[i] = hypre_CTAlloc(HYPRE_Int, block_size, HYPRE_MEMORY_HOST);
     memset(block_cf_marker[i], FMRK, block_size*sizeof(HYPRE_Int));
   }
   for (i = 0; i < max_num_levels; i++)
   {
     for(j=0; j<block_num_coarse_points[i]; j++)
     {
       (block_cf_marker[i])[block_coarse_indexes[i][j]] = CMRK;
     }
   }

   /* store block_num_coarse_points */
   if(max_num_levels > 0)
   {
      block_num_coarse_indexes = hypre_CTAlloc(HYPRE_Int,  max_num_levels, HYPRE_MEMORY_HOST);
      for(i=0; i<max_num_levels; i++)
         block_num_coarse_indexes[i] = block_num_coarse_points[i];
   }
   /* set block data */
   (mgr_data -> max_num_coarse_levels) = max_num_levels;
   (mgr_data -> block_size) = block_size;
   (mgr_data -> block_num_coarse_indexes) = block_num_coarse_indexes;
   (mgr_data -> block_cf_marker) = block_cf_marker;

   return hypre_error_flag;
}

/*Set number of points that remain part of the coarse grid throughout the hierarchy */
HYPRE_Int
hypre_MGRSetReservedCoarseNodes(void      *mgr_vdata,
                  HYPRE_Int reserved_coarse_size,
                  HYPRE_Int *reserved_cpt_index)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   HYPRE_Int *reserved_coarse_indexes = NULL;
   HYPRE_Int i;

   if (!mgr_data)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! MGR object empty!\n");
      return hypre_error_flag;
   }

   if(reserved_coarse_size < 0)
   {
     hypre_error_in_arg(2);
          return hypre_error_flag;
   }
   /* free data not previously destroyed */
   if((mgr_data -> reserved_coarse_indexes))
   {
      hypre_TFree((mgr_data -> reserved_coarse_indexes), HYPRE_MEMORY_HOST);
      (mgr_data -> reserved_coarse_indexes) = NULL;
   }

        /* set reserved coarse nodes */
        if(reserved_coarse_size > 0)
        {
           reserved_coarse_indexes = hypre_CTAlloc(HYPRE_Int,  reserved_coarse_size, HYPRE_MEMORY_HOST);
           for(i=0; i<reserved_coarse_size; i++)
              reserved_coarse_indexes[i] = reserved_cpt_index[i];
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
               HYPRE_Int **CF_marker,
               HYPRE_Int cflag)
{
  HYPRE_Int *cf_marker, i, row, nc;
  HYPRE_Int *cindexes = fixed_coarse_indexes;

  HYPRE_Int nloc =  hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

  /* If this is the last level, coarsen onto fixed coarse set */
  if(cflag)
  {
    if(*CF_marker != NULL)
    {
       hypre_TFree(*CF_marker, HYPRE_MEMORY_HOST);
    }
    cf_marker = hypre_CTAlloc(HYPRE_Int, nloc, HYPRE_MEMORY_HOST);
    memset(cf_marker, FMRK, nloc*sizeof(HYPRE_Int));

    /* first mark fixed coarse set */
    nc = fixed_coarse_size;
    for(i = 0; i < nc; i++)
    {
       cf_marker[cindexes[i]] = CMRK;
    }
  }
  else {
    /* First coarsen to get initial CF splitting.
     * This is then followed by updating the CF marker to pass
     * coarse information to the next levels. NOTE: It may be
     * convenient to implement this way (allows the use of multiple
     * coarsening strategies without changing too much code),
     * but not necessarily the best option, compared to initializing
     * CF_marker first and then coarsening on subgraph which excludes
     * the initialized coarse nodes.
    */
    hypre_BoomerAMGCoarsen(S, A, 0, debug_flag, &cf_marker);

    /* Update CF_marker to correct Cpoints marked as Fpoints. */
    nc = fixed_coarse_size;
    for(i = 0; i < nc; i++)
    {
       cf_marker[cindexes[i]] = CMRK;
    }
    /* set F-points to FMRK. This is necessary since the different coarsening schemes differentiate
     * between type of F-points (example Ruge coarsening). We do not need that distinction here.
    */
    for (row = 0; row <nloc; row++)
    {
       if(cf_marker[row] == CMRK) continue;
       cf_marker[row] = FMRK;
    }
#if 0
    /* IMPORTANT: Update coarse_indexes array to define the positions of the fixed coarse points
     * in the next level.
     */
    nc = 0;
    index_i = 0;
    for (row = 0; row <nloc; row++)
    {
       /* loop through new c-points */
       if(cf_marker[row] == CMRK) nc++;
       else if(cf_marker[row] == S_CMRK)
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
          cf_marker[row] = FMRK;
       }
    }
    /* check if this should be last level */
    if( nc == fixed_coarse_size)
       last_level = 1;
    //printf(" nc = %d and fixed coarse size = %d \n", nc, fixed_coarse_size);
#endif
  }
  /* set CF_marker */
  *CF_marker = cf_marker;

  return hypre_error_flag;
}

/* Interpolation for MGR - Adapted from BoomerAMGBuildInterp */
HYPRE_Int
hypre_MGRBuildP( hypre_ParCSRMatrix   *A,
               HYPRE_Int            *CF_marker,
               HYPRE_Int            *num_cpts_global,
               HYPRE_Int             method,
               HYPRE_Int             debug_flag,
               hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd         = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Real       *a_diag;

   hypre_ParCSRMatrix    *P;
   HYPRE_Int            *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;

   HYPRE_Int             *P_marker, *P_marker_offd;

   HYPRE_Int              jj_counter,jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd;
//   HYPRE_Int              jj_begin_row,jj_begin_row_offd;
//   HYPRE_Int              jj_end_row,jj_end_row_offd;

   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_Int              total_global_cpts;
   HYPRE_Int              num_cols_P_offd,my_first_cpt;

   HYPRE_Int              i,i1;
   HYPRE_Int              j,jl,jj;
   HYPRE_Int              start;

   HYPRE_Real       one  = 1.0;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;

   HYPRE_Int             *int_buf_data;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_cols_A_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                         num_sends), HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                    CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

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
      size = n_fine/num_threads;
      rest = n_fine - size*num_threads;

      if (j < rest)
      {
         ns = j*size+j;
         ne = (j+1)*size+j+1;
      }
      else
      {
         ns = j*size+rest;
         ne = (j+1)*size+rest;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
          *--------------------------------------------------------------------*/
          else
          {
             for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
             {
                i1 = A_diag_j[jj];
                if (CF_marker[i1] >= 0)
                {
                   jj_count[j]++;
                }
             }

             if (num_procs > 1)
             {
                for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                {
                   i1 = A_offd_j[jj];
                   if (CF_marker_offd[i1] >= 0)
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
   for (i=0; i < num_threads-1; i++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads-1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_SHARED);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_SHARED);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, HYPRE_MEMORY_SHARED);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_SHARED);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_SHARED);
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, HYPRE_MEMORY_SHARED);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) coarse_shift = coarse_counter[j-1];
      size = n_fine/num_threads;
      rest = n_fine - size*num_threads;
      if (j < rest)
      {
         ns = j*size+j;
         ne = (j+1)*size+j+1;
      }
      else
      {
         ns = j*size+rest;
         ne = (j+1)*size+rest;
      }
      for (i = ns; i < ne; i++)
         fine_to_coarse[i] += my_first_cpt+coarse_shift;
   }

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         int_buf_data[index++]
            = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                    fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   a_diag = hypre_CTAlloc(HYPRE_Real,  n_fine, HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
      {
         i1 = A_diag_j[jj];
         if ( i==i1 )  /* diagonal of A only */
         {
            a_diag[i] = 1.0/A_diag_data[jj];
         }
      }
   }

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine/num_threads;
      rest = n_fine - size*num_threads;
      if (jl < rest)
      {
         ns = jl*size+jl;
         ne = (jl+1)*size+jl+1;
      }
      else
      {
         ns = jl*size+rest;
         ne = (jl+1)*size+rest;
      }
      jj_counter = 0;
      if (jl > 0) jj_counter = jj_count[jl-1];
      jj_counter_offd = 0;
      if (jl > 0) jj_counter_offd = jj_count_offd[jl-1];
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      if (num_cols_A_offd)
         P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
      else
         P_marker_offd = NULL;

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/
         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  if(method == 0)
                  {
                     P_diag_data[jj_counter] = 0.0;
                  }
                  else if (method == 1)
                  {
                     P_diag_data[jj_counter] = - A_diag_data[jj];
                  }
                  else if (method == 2)
                  {
                        P_diag_data[jj_counter] = - A_diag_data[jj]*a_diag[i];

                  }
                  jj_counter++;
               }
            }

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     if(method == 0)
                  {
                      P_offd_data[jj_counter_offd] = 0.0;
                  }
                  else if (method == 1)
                  {
                     P_offd_data[jj_counter_offd] = - A_offd_data[jj];
                  }
                  else if (method == 2)
                  {
                     P_offd_data[jj_counter_offd] = - A_offd_data[jj]*a_diag[i];
                  }

                     jj_counter_offd++;
                  }
               }
            }
         }
         P_offd_i[i+1] = jj_counter_offd;
      }
    hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
    hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }
 hypre_TFree(a_diag, HYPRE_MEMORY_HOST);
   P = hypre_ParCSRMatrixCreate(comm,
                         hypre_ParCSRMatrixGlobalNumRows(A),
                         total_global_cpts,
                         hypre_ParCSRMatrixColStarts(A),
                         num_cpts_global,
                         0,
                         P_diag_i[n_fine],
                         P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   num_cols_P_offd = 0;

   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i=0; i < num_cols_A_offd; i++)
         P_marker[i] = 0;
      num_cols_P_offd = 0;
      for (i=0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index]==0) index++;
         col_map_offd_P[i] = index++;
      }

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i=0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
                                  P_offd_j[i],
                                  num_cols_P_offd);
    hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i=0; i < n_fine; i++)
      if (CF_marker[i] == -3) CF_marker[i] = -1;
   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }
   hypre_GetCommPkgRTFromCommPkgA(P,A, fine_to_coarse_offd);

   *P_ptr = P;

 hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
 hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
 hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
 hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
 hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
 hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
 hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   return(0);
}


/* Interpolation for MGR - Dynamic Row Sum method */

HYPRE_Int
hypre_MGRBuildPDRS( hypre_ParCSRMatrix   *A,
                 HYPRE_Int            *CF_marker,
                 HYPRE_Int            *num_cpts_global,
                 HYPRE_Int             blk_size,
                 HYPRE_Int             reserved_coarse_size,
                 HYPRE_Int             debug_flag,
                 hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd         = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data    = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Real       *a_diag;

   hypre_ParCSRMatrix    *P;
   HYPRE_Int            *col_map_offd_P;

   HYPRE_Int             *CF_marker_offd = NULL;

   hypre_CSRMatrix    *P_diag;
   hypre_CSRMatrix    *P_offd;

   HYPRE_Real      *P_diag_data;
   HYPRE_Int             *P_diag_i;
   HYPRE_Int             *P_diag_j;
   HYPRE_Real      *P_offd_data;
   HYPRE_Int             *P_offd_i;
   HYPRE_Int             *P_offd_j;

   HYPRE_Int              P_diag_size, P_offd_size;

   HYPRE_Int             *P_marker, *P_marker_offd;

   HYPRE_Int              jj_counter,jj_counter_offd;
   HYPRE_Int             *jj_count, *jj_count_offd;
//   HYPRE_Int              jj_begin_row,jj_begin_row_offd;
//   HYPRE_Int              jj_end_row,jj_end_row_offd;

   HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   HYPRE_Int              n_fine  = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             *fine_to_coarse;
   HYPRE_Int             *fine_to_coarse_offd;
   HYPRE_Int             *coarse_counter;
   HYPRE_Int              coarse_shift;
   HYPRE_Int              total_global_cpts;
   HYPRE_Int              num_cols_P_offd,my_first_cpt;

   HYPRE_Int              i,i1;
   HYPRE_Int              j,jl,jj;
   HYPRE_Int              start;

   HYPRE_Real       one  = 1.0;

   HYPRE_Int              my_id;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;
   HYPRE_Int              num_sends;
   HYPRE_Int              index;
   HYPRE_Int              ns, ne, size, rest;

   HYPRE_Int             *int_buf_data;

   HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   num_threads = hypre_NumThreads();

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs -1)) total_global_cpts = num_cpts_global[1];
   hypre_MPI_Bcast(&total_global_cpts, 1, HYPRE_MPI_INT, num_procs-1, comm);
#else
   my_first_cpt = num_cpts_global[my_id];
   total_global_cpts = num_cpts_global[num_procs];
#endif

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_cols_A_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                         num_sends), HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         int_buf_data[index++]
            = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                    CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);
   jj_count_offd = hypre_CTAlloc(HYPRE_Int,  num_threads, HYPRE_MEMORY_HOST);

   fine_to_coarse = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] = -1;

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
      size = n_fine/num_threads;
      rest = n_fine - size*num_threads;

      if (j < rest)
      {
         ns = j*size+j;
         ne = (j+1)*size+j+1;
      }
      else
      {
         ns = j*size+rest;
         ne = (j+1)*size+rest;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
          *--------------------------------------------------------------------*/
          else
          {
             for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
             {
                i1 = A_diag_j[jj];
                if (CF_marker[i1] >= 0)
                {
                   jj_count[j]++;
                }
             }

             if (num_procs > 1)
             {
                for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                {
                   i1 = A_offd_j[jj];
                   if (CF_marker_offd[i1] >= 0)
                   {
                      jj_count_offd[j]++;
                   }
                }
             }
          }
         /*--------------------------------------------------------------------
          *  Set up the indexes for the DRS method
          *--------------------------------------------------------------------*/

      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i=0; i < num_threads-1; i++)
   {
      coarse_counter[i+1] += coarse_counter[i];
      jj_count[i+1] += jj_count[i];
      jj_count_offd[i+1] += jj_count_offd[i];
   }
   i = num_threads-1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_HOST);
   P_diag_j    = hypre_CTAlloc(HYPRE_Int,  P_diag_size, HYPRE_MEMORY_HOST);
   P_diag_data = hypre_CTAlloc(HYPRE_Real,  P_diag_size, HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = hypre_CTAlloc(HYPRE_Int,  n_fine+1, HYPRE_MEMORY_HOST);
   P_offd_j    = hypre_CTAlloc(HYPRE_Int,  P_offd_size, HYPRE_MEMORY_HOST);
   P_offd_data = hypre_CTAlloc(HYPRE_Real,  P_offd_size, HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   fine_to_coarse_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) coarse_shift = coarse_counter[j-1];
      size = n_fine/num_threads;
      rest = n_fine - size*num_threads;
      if (j < rest)
      {
         ns = j*size+j;
         ne = (j+1)*size+j+1;
      }
      else
      {
         ns = j*size+rest;
         ne = (j+1)*size+rest;
      }
      for (i = ns; i < ne; i++)
         fine_to_coarse[i] += my_first_cpt+coarse_shift;
   }

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         int_buf_data[index++]
            = fine_to_coarse[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                    fine_to_coarse_offd);

   hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag==4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag==4) wall_time = time_getWallclockSeconds();

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   a_diag = hypre_CTAlloc(HYPRE_Real,  n_fine, HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
      {
         i1 = A_diag_j[jj];
         if ( i==i1 )  /* diagonal of A only */
         {
            a_diag[i] = 1.0/A_diag_data[jj];
         }
      }
   }

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine/num_threads;
      rest = n_fine - size*num_threads;
      if (jl < rest)
      {
         ns = jl*size+jl;
         ne = (jl+1)*size+jl+1;
      }
      else
      {
         ns = jl*size+rest;
         ne = (jl+1)*size+rest;
      }
      jj_counter = 0;
      if (jl > 0) jj_counter = jj_count[jl-1];
      jj_counter_offd = 0;
      if (jl > 0) jj_counter_offd = jj_count_offd[jl-1];
      P_marker = hypre_CTAlloc(HYPRE_Int,  n_fine, HYPRE_MEMORY_HOST);
      if (num_cols_A_offd)
         P_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
      else
         P_marker_offd = NULL;

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/
         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = - A_diag_data[jj]*a_diag[i];

                  jj_counter++;
               }
            }

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     P_offd_data[jj_counter_offd] = - A_offd_data[jj]*a_diag[i];

                     jj_counter_offd++;
                  }
               }
            }
         }
         P_offd_i[i+1] = jj_counter_offd;
      }
    hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
    hypre_TFree(P_marker_offd, HYPRE_MEMORY_HOST);
   }
 hypre_TFree(a_diag, HYPRE_MEMORY_HOST);
   P = hypre_ParCSRMatrixCreate(comm,
                         hypre_ParCSRMatrixGlobalNumRows(A),
                         total_global_cpts,
                         hypre_ParCSRMatrixColStarts(A),
                         num_cpts_global,
                         0,
                         P_diag_i[n_fine],
                         P_offd_i[n_fine]);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixData(P_offd) = P_offd_data;
   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixJ(P_offd) = P_offd_j;
   hypre_ParCSRMatrixOwnsRowStarts(P) = 0;

   num_cols_P_offd = 0;

   if (P_offd_size)
   {
      P_marker = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_HOST);
#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i=0; i < num_cols_A_offd; i++)
         P_marker[i] = 0;
      num_cols_P_offd = 0;
      for (i=0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd, HYPRE_MEMORY_HOST);
      index = 0;
      for (i=0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index]==0) index++;
         col_map_offd_P[i] = index++;
      }

#if 0
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i=0; i < P_offd_size; i++)
         P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
                                  P_offd_j[i],
                                  num_cols_P_offd);
    hypre_TFree(P_marker, HYPRE_MEMORY_HOST);
   }

   for (i=0; i < n_fine; i++)
      if (CF_marker[i] == -3) CF_marker[i] = -1;
   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }
   hypre_GetCommPkgRTFromCommPkgA(P,A, fine_to_coarse_offd);

   *P_ptr = P;

 hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
 hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
 hypre_TFree(fine_to_coarse, HYPRE_MEMORY_HOST);
 hypre_TFree(fine_to_coarse_offd, HYPRE_MEMORY_HOST);
 hypre_TFree(coarse_counter, HYPRE_MEMORY_HOST);
 hypre_TFree(jj_count, HYPRE_MEMORY_HOST);
 hypre_TFree(jj_count_offd, HYPRE_MEMORY_HOST);

   return(0);
}

/* Setup interpolation operator */
HYPRE_Int
hypre_MGRBuildInterp(hypre_ParCSRMatrix     *A,
         HYPRE_Int             *CF_marker,
                         hypre_ParCSRMatrix   *S,
                         HYPRE_Int            *num_cpts_global,
                         HYPRE_Int            num_functions,
                         HYPRE_Int            *dof_func,
                         HYPRE_Int            debug_flag,
                         HYPRE_Real           trunc_factor,
                         HYPRE_Int         max_elmts,
                         HYPRE_Int          *col_offd_S_to_A,
         hypre_ParCSRMatrix    **P,
         HYPRE_Int         last_level,
         HYPRE_Int         method,
         HYPRE_Int         numsweeps)
{
//   HYPRE_Int i;
   hypre_ParCSRMatrix    *P_ptr = NULL;
//   HYPRE_Real       jac_trunc_threshold = trunc_factor;
//   HYPRE_Real       jac_trunc_threshold_minus = 0.5*jac_trunc_threshold;

   /* Build interpolation operator using (hypre default) */
   if(!last_level)
   {
           hypre_MGRBuildP( A,CF_marker,num_cpts_global,2,debug_flag,&P_ptr);
   }
   /* Do Jacobi interpolation for last level */
   else
   {
      if (method <3)
      {
         hypre_MGRBuildP( A,CF_marker,num_cpts_global,method,debug_flag,&P_ptr);
         /* Could do a few sweeps of Jacobi to further improve P */
                    //for(i=0; i<numsweeps; i++)
         //    hypre_BoomerAMGJacobiInterp(A, &P_ptr, S,1, NULL, CF_marker, 0, jac_trunc_threshold, jac_trunc_threshold_minus );
      }
      else
      {

         /* Classical modified interpolation */
         hypre_BoomerAMGBuildInterp(A, CF_marker, S, num_cpts_global,1, NULL,debug_flag,
                          trunc_factor, max_elmts, col_offd_S_to_A, &P_ptr);

              /* Do k steps of Jacobi build W for P = [-W I].
          * Note that BoomerAMGJacobiInterp assumes you have some initial P,
          * hence we need to initialize P as above, before calling this routine.
          * If numsweeps = 0, the following step is skipped and P is returned as is.
          * Looping here is equivalent to improving P by Jacobi interpolation
                   */
//         for(i=0; i<numsweeps; i++)
//            hypre_BoomerAMGJacobiInterp(A, &P_ptr, S,1, NULL, CF_marker,
//                                 0, jac_trunc_threshold,
//                                 jac_trunc_threshold_minus );
      }

   }
   /* set pointer to P */
   *P = P_ptr;

   return hypre_error_flag;
}

void hypre_blas_smat_inv_n4 (HYPRE_Real *a)
{
    const HYPRE_Real a11 = a[0],  a12 = a[1],  a13 = a[2],  a14 = a[3];
    const HYPRE_Real a21 = a[4],  a22 = a[5],  a23 = a[6],  a24 = a[7];
    const HYPRE_Real a31 = a[8],  a32 = a[9],  a33 = a[10], a34 = a[11];
    const HYPRE_Real a41 = a[12], a42 = a[13], a43 = a[14], a44 = a[15];

    const HYPRE_Real M11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 - a24*a33*a42;
    const HYPRE_Real M12 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 - a12*a33*a44 - a13*a34*a42 - a14*a32*a43;
    const HYPRE_Real M13 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 - a14*a23*a42;
    const HYPRE_Real M14 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 - a12*a23*a34 - a13*a24*a32 - a14*a22*a33;
    const HYPRE_Real M21 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 - a21*a33*a44 - a23*a34*a41 - a24*a31*a43;
    const HYPRE_Real M22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 - a14*a33*a41;
    const HYPRE_Real M23 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 - a11*a23*a44 - a13*a24*a41 - a14*a21*a43;
    const HYPRE_Real M24 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 - a14*a23*a31;
    const HYPRE_Real M31 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 - a24*a32*a41;
    const HYPRE_Real M32 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 - a11*a32*a44 - a12*a34*a41 - a14*a31*a42;
    const HYPRE_Real M33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 - a14*a22*a41;
    const HYPRE_Real M34 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 - a11*a22*a34 - a12*a24*a31 - a14*a21*a32;
    const HYPRE_Real M41 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 - a21*a32*a43 - a22*a33*a41 - a23*a31*a42;
    const HYPRE_Real M42 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 - a13*a32*a41;
    const HYPRE_Real M43 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 - a11*a22*a43 - a12*a23*a41 - a13*a21*a42;
    const HYPRE_Real M44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31;

    const HYPRE_Real det = a11*M11 + a12*M21 + a13*M31 + a14*M41;
    HYPRE_Real det_inv;

    //if ( fabs(det) < 1e-22 ) {
        /* there should be no print statements that can't be turned off. Is this an error? */
        //hypre_fprintf(stderr, "### WARNING: Matrix is nearly singular! det = %e\n", det);
        /*
         printf("##----------------------------------------------\n");
         printf("## %12.5e %12.5e %12.5e \n", a0, a1, a2);
         printf("## %12.5e %12.5e %12.5e \n", a3, a4, a5);
         printf("## %12.5e %12.5e %12.5e \n", a5, a6, a7);
         printf("##----------------------------------------------\n");
         getchar();
         */
    //}

    det_inv = 1.0/det;

    a[0] = M11*det_inv;  a[1] = M12*det_inv;  a[2] = M13*det_inv;  a[3] = M14*det_inv;
    a[4] = M21*det_inv;  a[5] = M22*det_inv;  a[6] = M23*det_inv;  a[7] = M24*det_inv;
    a[8] = M31*det_inv;  a[9] = M32*det_inv;  a[10] = M33*det_inv; a[11] = M34*det_inv;
    a[12] = M41*det_inv; a[13] = M42*det_inv; a[14] = M43*det_inv; a[15] = M44*det_inv;

}

void hypre_blas_mat_inv(HYPRE_Real *a,
                  HYPRE_Int n)
{
   HYPRE_Int i,j,k,l,u,kn,in;
   HYPRE_Real alinv;
   if (n == 4)
   {
      hypre_blas_smat_inv_n4(a);
   }
   else
   {
      for (k=0; k<n; ++k) {
         kn = k*n;
         l  = kn+k;

         //if (fabs(a[l]) < SMALLREAL) {
         //   printf("### WARNING: Diagonal entry is close to zero!");
         //   printf("### WARNING: diag_%d=%e\n", k, a[l]);
         //   a[l] = SMALLREAL;
         //}
         alinv = 1.0/a[l];
         a[l] = alinv;

         for (j=0; j<k; ++j) {
            u = kn+j; a[u] *= alinv;
         }

         for (j=k+1; j<n; ++j) {
            u = kn+j; a[u] *= alinv;
         }

         for (i=0; i<k; ++i) {
            in = i*n;
            for (j=0; j<n; ++j)
               if (j!=k) {
                  u = in+j; a[u] -= a[in+k]*a[kn+j];
               } // end if (j!=k)
         }

         for (i=k+1; i<n; ++i) {
            in = i*n;
            for (j=0; j<n; ++j)
               if (j!=k) {
                  u = in+j; a[u] -= a[in+k]*a[kn+j];
               } // end if (j!=k)
         }

         for (i=0; i<k; ++i) {
            u=i*n+k; a[u] *= -alinv;
         }

         for (i=k+1; i<n; ++i) {
            u=i*n+k; a[u] *= -alinv;
         }
      } // end for (k=0; k<n; ++k)
   }// end if
}

HYPRE_Int hypre_block_jacobi_scaling(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **B_ptr,
                 void *mgr_vdata, HYPRE_Int debug_flag)
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   hypre_ParMGRData   *mgr_data =  (hypre_ParMGRData*) mgr_vdata;

   HYPRE_Int         num_procs,  my_id;

   HYPRE_Int    blk_size  = (mgr_data -> block_size);
   HYPRE_Int    reserved_coarse_size = (mgr_data -> reserved_coarse_size);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_ParCSRMatrix    *B;

   hypre_CSRMatrix *B_diag;
   HYPRE_Real      *B_diag_data;
   HYPRE_Int       *B_diag_i;
   HYPRE_Int       *B_diag_j;

   hypre_CSRMatrix *B_offd;
   HYPRE_Int              i,ii;
   HYPRE_Int              j,jj;
   HYPRE_Int              k;

   HYPRE_Int              n = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int n_block, left_size,inv_size;

//   HYPRE_Real       wall_time;  /* for debugging instrumentation  */
   HYPRE_Int        bidx,bidxm1,bidxp1;
   HYPRE_Real       * diaginv;

   const HYPRE_Int     nb2 = blk_size*blk_size;

   HYPRE_Int block_scaling_error = 0;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
//   HYPRE_Int num_threads = hypre_NumThreads();

   //printf("n = %d\n",n);

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size*n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size*n_block;
   }

   inv_size  = nb2*n_block + left_size*left_size;

   //printf("inv_size = %d\n",inv_size);

   hypre_blockRelax_setup(A,blk_size,reserved_coarse_size,&(mgr_data -> diaginv));

//   if (debug_flag==4) wall_time = time_getWallclockSeconds();

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of B and fill in
    *-----------------------------------------------------------------------*/

   B_diag_i    = hypre_CTAlloc(HYPRE_Int,  n+1, HYPRE_MEMORY_HOST);
   B_diag_j    = hypre_CTAlloc(HYPRE_Int,  inv_size, HYPRE_MEMORY_HOST);
   B_diag_data = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);

   B_diag_i[n] = inv_size;

   //B_offd_i    = hypre_CTAlloc(HYPRE_Int,  n+1, HYPRE_MEMORY_HOST);
   //B_offd_j    = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
   //B_offd_data = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_HOST);

   //B_offd_i[n] = 1;
    /*-----------------------------------------------------------------
    * Get all the diagonal sub-blocks
    *-----------------------------------------------------------------*/
   diaginv = hypre_CTAlloc(HYPRE_Real,  nb2, HYPRE_MEMORY_HOST);
   //printf("n_block = %d\n",n_block);
   for (i = 0;i < n_block; i++)
   {
      bidxm1 = i*blk_size;
      bidxp1 = (i+1)*blk_size;

      for (k = 0;k < blk_size; k++)
      {
         for (j = 0;j < blk_size; j++)
         {
            bidx = k*blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1+k]; ii < A_diag_i[bidxm1+k+1]; ii++)
         {

            jj = A_diag_j[ii];

            if (jj >= bidxm1 && jj < bidxp1 && fabs(A_diag_data[ii]) > SMALLREAL)
            {
               bidx = k*blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }

      /* for (k = 0;k < blk_size; k++) */
      /* { */
      /*    for (j = 0;j < blk_size; j++) */
      /*    { */
      /*       bidx = k*blk_size + j; */
      /*       printf("diaginv[%d] = %e\n",bidx,diaginv[bidx]); */
      /*    } */
      /* } */

      hypre_blas_mat_inv(diaginv, blk_size);

      for (k = 0;k < blk_size; k++)
      {
         B_diag_i[i*blk_size+k] = i*nb2 + k*blk_size;
         //B_offd_i[i*nb2+k] = 0;

         for (j = 0;j < blk_size; j++)
         {
            bidx = i*nb2 + k*blk_size + j;
            B_diag_j[bidx] = i*blk_size + j;
            B_diag_data[bidx] = diaginv[k*blk_size + j];
         }
      }
   }

   //printf("Before create\n");
   B = hypre_ParCSRMatrixCreate(comm,
                          hypre_ParCSRMatrixGlobalNumRows(A),
                          hypre_ParCSRMatrixGlobalNumCols(A),
                          hypre_ParCSRMatrixRowStarts(A),
                          hypre_ParCSRMatrixColStarts(A),
                          0,
                          inv_size,
                         0);
   //printf("After create\n");
   B_diag = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrixData(B_diag) = B_diag_data;
   hypre_CSRMatrixI(B_diag) = B_diag_i;
   hypre_CSRMatrixJ(B_diag) = B_diag_j;
   B_offd = hypre_ParCSRMatrixOffd(B);
   hypre_CSRMatrixData(B_offd) = NULL;
   hypre_CSRMatrixI(B_offd) = NULL;
   hypre_CSRMatrixJ(B_offd) = NULL;
   /* hypre_ParCSRMatrixOwnsRowStarts(B) = 0; */

   *B_ptr = B;

   return(block_scaling_error);
}

HYPRE_Int hypre_block_jacobi (hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u,
                       HYPRE_Real         blk_size,
                       HYPRE_Int           n_block,
                       HYPRE_Int           left_size,
                              HYPRE_Real *diaginv,
                       hypre_ParVector    *Vtemp)
{
   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real     *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int            *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Real     *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int            *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int             num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Real     *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Real     *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   HYPRE_Real     *Vtemp_data = hypre_VectorData(Vtemp_local);
   HYPRE_Real      *Vext_data = NULL;
   HYPRE_Real      *v_buf_data;

   HYPRE_Int             i, j, k;
   HYPRE_Int             ii, jj;
   HYPRE_Int             bidx,bidx1;
   HYPRE_Int             relax_error = 0;
   HYPRE_Int         num_sends;
   HYPRE_Int         index, start;
   HYPRE_Int         num_procs, my_id;
   HYPRE_Real       *res;

    const HYPRE_Int     nb2 = blk_size*blk_size;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
//   HYPRE_Int num_threads = hypre_NumThreads();

   res = hypre_CTAlloc(HYPRE_Real,  blk_size, HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = hypre_CTAlloc(HYPRE_Real,
                           hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

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
           for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            v_buf_data[index++]
                    = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
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
   for (i = 0;i < n_block; i++)
   {
      for (j = 0;j < blk_size; j++)
      {
         bidx = i*blk_size +j;
         res[j] = f_data[bidx];
         for (jj = A_diag_i[bidx]; jj < A_diag_i[bidx+1]; jj++)
         {
            ii = A_diag_j[jj];
            res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            //printf("%d: Au= %e * %e =%e\n",ii,A_diag_data[jj],Vtemp_data[ii], res[j]);
         }
         for (jj = A_offd_i[bidx]; jj < A_offd_i[bidx+1]; jj++)
         {
            ii = A_offd_j[jj];
            res[j] -= A_offd_data[jj] * Vext_data[ii];
         }
         //printf("%d: res = %e\n",bidx,res[j]);
      }

      for (j = 0;j < blk_size; j++)
      {
         bidx1 = i*blk_size +j;
         for (k = 0;k < blk_size; k++)
         {
            bidx  = i*nb2 +j*blk_size+k;
            u_data[bidx1] += res[k]*diaginv[bidx];
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
   return(relax_error);
}

/*Block smoother*/
HYPRE_Int
hypre_blockRelax_setup(hypre_ParCSRMatrix *A,
                  HYPRE_Int          blk_size,
                  HYPRE_Int          reserved_coarse_size,
                  HYPRE_Real        **diaginvptr)
{
   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real     *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             i, j,k;
   HYPRE_Int             ii, jj;
   HYPRE_Int             bidx,bidxm1,bidxp1;
   HYPRE_Int         num_procs, my_id;

    const HYPRE_Int     nb2 = blk_size*blk_size;
   HYPRE_Int           n_block;
   HYPRE_Int           left_size,inv_size;
   HYPRE_Real        *diaginv = *diaginvptr;


   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
//   HYPRE_Int num_threads = hypre_NumThreads();

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size*n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size*n_block;
   }

   inv_size  = nb2*n_block + left_size*left_size;

   if (diaginv !=NULL)
   {
    hypre_TFree(diaginv, HYPRE_MEMORY_HOST);
      diaginv = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);
   }
   else {
      diaginv = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------
    * Get all the diagonal sub-blocks
    *-----------------------------------------------------------------*/
   for (i = 0;i < n_block; i++)
   {
      bidxm1 = i*blk_size;
      bidxp1 = (i+1)*blk_size;
      //printf("bidxm1 = %d,bidxp1 = %d\n",bidxm1,bidxp1);

      for (k = 0;k < blk_size; k++)
      {
         for (j = 0;j < blk_size; j++)
         {
            bidx = i*nb2 + k*blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1+k]; ii < A_diag_i[bidxm1+k+1]; ii++)
         {

            jj = A_diag_j[ii];

            if (jj >= bidxm1 && jj < bidxp1 && fabs(A_diag_data[ii]) > SMALLREAL)
            {
               bidx = i*nb2 + k*blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }
   }



   for (i = 0;i < left_size; i++)
   {
      bidxm1 =n_block*nb2 + i*blk_size;
      bidxp1 =n_block*nb2 + (i+1)*blk_size;
      for (j = 0;j < left_size; j++)
      {
         bidx = n_block*nb2 + i*blk_size +j;
         diaginv[bidx] = 0.0;
      }

      for (ii = A_diag_i[n_block*blk_size + i]; ii < A_diag_i[n_block*blk_size+i+1]; ii++)
      {
         jj = A_diag_j[ii];
         if (jj > n_block*blk_size)
         {
            bidx = n_block*nb2 + i*blk_size + jj - n_block*blk_size;
            diaginv[bidx] = A_diag_data[ii];
         }
      }
   }


   /*-----------------------------------------------------------------
    * compute the inverses of all the diagonal sub-blocks
    *-----------------------------------------------------------------*/
   if (blk_size > 1)
   {
      for (i = 0;i < n_block; i++)
      {
         hypre_blas_mat_inv(diaginv+i*nb2, blk_size);
      }
      hypre_blas_mat_inv(diaginv+(HYPRE_Int)(blk_size*nb2),left_size);
   }
   else
   {
      for (i = 0;i < n; i++)
      {
         // FIX-ME: zero-diagonal should be tested previously
         if (fabs(diaginv[i]) < SMALLREAL)
            diaginv[i] = 0.0;
         else
            diaginv[i] = 1.0 / diaginv[i];
      }
   }

   *diaginvptr = diaginv;

   return 1;
}

HYPRE_Int
hypre_blockRelax(hypre_ParCSRMatrix *A,
             hypre_ParVector    *f,
             hypre_ParVector    *u,
             HYPRE_Int          blk_size,
             HYPRE_Int          reserved_coarse_size,
             hypre_ParVector    *Vtemp,
             hypre_ParVector    *Ztemp)
{
   MPI_Comm      comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real     *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int            *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int             n       = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Int             i, j,k;
   HYPRE_Int             ii, jj;

   HYPRE_Int             bidx,bidxm1,bidxp1;
   HYPRE_Int             relax_error = 0;

   HYPRE_Int         num_procs, my_id;

    const HYPRE_Int     nb2 = blk_size*blk_size;
   HYPRE_Int           n_block;
   HYPRE_Int           left_size,inv_size;
   HYPRE_Real          *diaginv;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

//   HYPRE_Int num_threads = hypre_NumThreads();

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size*n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size*n_block;
   }

   inv_size  = nb2*n_block + left_size*left_size;

   diaginv = hypre_CTAlloc(HYPRE_Real,  inv_size, HYPRE_MEMORY_HOST);
   /*-----------------------------------------------------------------
    * Get all the diagonal sub-blocks
    *-----------------------------------------------------------------*/
   for (i = 0;i < n_block; i++)
   {
      bidxm1 = i*blk_size;
      bidxp1 = (i+1)*blk_size;
      //printf("bidxm1 = %d,bidxp1 = %d\n",bidxm1,bidxp1);

      for (k = 0;k < blk_size; k++)
      {
         for (j = 0;j < blk_size; j++)
         {
            bidx = i*nb2 + k*blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1+k]; ii < A_diag_i[bidxm1+k+1]; ii++)
         {

            jj = A_diag_j[ii];

            if (jj >= bidxm1 && jj < bidxp1 && fabs(A_diag_data[ii]) > SMALLREAL)
            {
               bidx = i*nb2 + k*blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }

   }

   for (i = 0;i < left_size; i++)
   {
      bidxm1 =n_block*nb2 + i*blk_size;
      bidxp1 =n_block*nb2 + (i+1)*blk_size;
      for (j = 0;j < left_size; j++)
      {
         bidx = n_block*nb2 + i*blk_size +j;
         diaginv[bidx] = 0.0;
      }

      for (ii = A_diag_i[n_block*blk_size + i]; ii < A_diag_i[n_block*blk_size+i+1]; ii++)
      {
         jj = A_diag_j[ii];
         if (jj > n_block*blk_size)
         {
            bidx = n_block*nb2 + i*blk_size + jj - n_block*blk_size;
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
      for (i = 0;i < n_block; i++)
      {
         hypre_blas_mat_inv(diaginv+i*nb2, blk_size);
      }
      hypre_blas_mat_inv(diaginv+(HYPRE_Int)(blk_size*nb2),left_size);
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
      for (i = 0;i < n; i++)
      {
         // FIX-ME: zero-diagonal should be tested previously
         if (fabs(diaginv[i]) < SMALLREAL)
            diaginv[i] = 0.0;
         else
            diaginv[i] = 1.0 / diaginv[i];
      }

   }

   hypre_block_jacobi(A,f,u,blk_size,n_block,left_size,diaginv,Vtemp);

   /*-----------------------------------------------------------------
    * Free temperary memeory
    *-----------------------------------------------------------------*/
 hypre_TFree(diaginv, HYPRE_MEMORY_HOST);

   return(relax_error);
}

/* set coarse grid solver */
HYPRE_Int
hypre_MGRSetCoarseSolver( void  *mgr_vdata,
                     HYPRE_Int  (*coarse_grid_solver_solve)(void*,void*,void*,void*),
                     HYPRE_Int  (*coarse_grid_solver_setup)(void*,void*,void*,void*),
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
 *
*/
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
   (mgr_data -> num_relax_sweeps) = nsweeps;
   return hypre_error_flag;
}

/* Set the F-relaxation strategy: 0=single level, 1=multi level
*/
HYPRE_Int
hypre_MGRSetFRelaxMethod( void *mgr_vdata, HYPRE_Int relax_method )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> Frelax_method) = relax_method;
   return hypre_error_flag;
}
/* Set the type of the restriction type
 * for computing restriction operator
*/
HYPRE_Int
hypre_MGRSetRestrictType( void *mgr_vdata, HYPRE_Int restrict_type)
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> restrict_type) = restrict_type;
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
   (mgr_data -> interp_type) = interpType;
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
/* Set print level for mgr solver */
HYPRE_Int
hypre_MGRSetPrintLevel( void *mgr_vdata, HYPRE_Int print_level )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> print_level) = print_level;
   return hypre_error_flag;
}
/* Set print level for mgr solver */
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
/* Set max number of iterations for mgr solver */
HYPRE_Int
hypre_MGRSetMaxGlobalsmoothIters( void *mgr_vdata, HYPRE_Int max_iter )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> global_smooth_iters) = max_iter;
   return hypre_error_flag;
}
/* Set max number of iterations for mgr solver */

HYPRE_Int
hypre_MGRSetGlobalsmoothType( void *mgr_vdata, HYPRE_Int iter_type )
{
   hypre_ParMGRData   *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> global_smooth_type) = iter_type;
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
hypre_MGRBuildAff( MPI_Comm comm, HYPRE_Int local_num_variables, HYPRE_Int num_functions,
  HYPRE_Int *dof_func, HYPRE_Int *CF_marker, HYPRE_Int **coarse_dof_func_ptr, HYPRE_Int **coarse_pnts_global_ptr,
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC," Coarse grid matrix is NULL. Please make sure MGRSetup() is called \n");
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC," MGR solution array is NULL. Please make sure MGRSetup() and MGRSolve() are called \n");
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC," MGR RHS array is NULL. Please make sure MGRSetup() and MGRSolve() are called \n");
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

/* Print solver params */
HYPRE_Int
hypre_MGRWriteSolverParams(void *mgr_vdata)
{
   hypre_ParMGRData  *mgr_data = (hypre_ParMGRData*) mgr_vdata;
   hypre_printf("MGR Setup parameters: \n");
   hypre_printf("Max number of coarse levels: %d\n", (mgr_data -> max_num_coarse_levels));
   hypre_printf("Block size: %d\n", (mgr_data -> block_size));
   hypre_printf("Number of coarse indexes: %d\n", (mgr_data -> num_coarse_indexes));
   hypre_printf("reserved coarse nodes size: %d\n", (mgr_data -> reserved_coarse_size));

   hypre_printf("\n MGR Solver Parameters: \n");
   hypre_printf("F-relaxation Method: %d\n", (mgr_data -> Frelax_method));
   hypre_printf("Relax type: %d\n", (mgr_data -> relax_type));
   hypre_printf("Number of relax sweeps: %d\n", (mgr_data -> num_relax_sweeps));
   hypre_printf("Interpolation type: %d\n", (mgr_data -> interp_type));
   hypre_printf("Number of interpolation sweeps: %d\n", (mgr_data -> num_interp_sweeps));
   hypre_printf("Restriction type: %d\n", (mgr_data -> restrict_type));
   hypre_printf("Number of restriction sweeps: %d\n", (mgr_data -> num_restrict_sweeps));
   hypre_printf("Global smoother type: %d\n", (mgr_data ->global_smooth_type));
   hypre_printf("Number of global smoother sweeps: %d\n", (mgr_data ->global_smooth_iters));
   hypre_printf("Max number of iterations: %d\n", (mgr_data -> max_iter));
   hypre_printf("Stopping tolerance: %e\n", (mgr_data -> tol));

   return hypre_error_flag;
}
