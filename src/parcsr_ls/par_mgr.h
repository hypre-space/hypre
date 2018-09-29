/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef hypre_ParMGR_DATA_HEADER
#define hypre_ParMGR_DATA_HEADER
/*--------------------------------------------------------------------------
 * hypre_ParMGRData
 *--------------------------------------------------------------------------*/
typedef struct
{
  // block data
  HYPRE_Int  block_size;
  HYPRE_Int  num_coarse_indexes;
  HYPRE_Int  *block_num_coarse_indexes;
  HYPRE_Int  **block_cf_marker;

  // initial setup data (user provided)
  HYPRE_Int num_coarse_levels;
  HYPRE_Int *num_coarse_per_level;
  HYPRE_Int **level_coarse_indexes;

  //general data
  HYPRE_Int max_num_coarse_levels;
  hypre_ParCSRMatrix **A_array;
  hypre_ParCSRMatrix **P_array;
  hypre_ParCSRMatrix **RT_array;
  hypre_ParCSRMatrix *RAP;
  HYPRE_Int **CF_marker_array;
  HYPRE_Int **coarse_indices_lvls;
  hypre_ParVector    **F_array;
  hypre_ParVector    **U_array;
  hypre_ParVector    *residual;
  HYPRE_Real    *rel_res_norms;

  HYPRE_Real   max_row_sum;
  HYPRE_Int    num_interp_sweeps;
  HYPRE_Int    num_restrict_sweeps;
  HYPRE_Int    interp_type;
  HYPRE_Int    restrict_type;
  HYPRE_Real   strong_threshold;
  HYPRE_Real   trunc_factor;
  HYPRE_Real   S_commpkg_switch;
  HYPRE_Int    P_max_elmts;
  HYPRE_Int    num_iterations;

  HYPRE_Real   **l1_norms;
  HYPRE_Real    final_rel_residual_norm;
  HYPRE_Real    tol;
  HYPRE_Real    relax_weight;
  HYPRE_Int     relax_type;
  HYPRE_Int     logging;
  HYPRE_Int     print_level;
  HYPRE_Int     max_iter;
  HYPRE_Int     relax_order;
  HYPRE_Int     num_relax_sweeps;

  HYPRE_Solver coarse_grid_solver;
  HYPRE_Int     (*coarse_grid_solver_setup)(void*,void*,void*,void*);
  HYPRE_Int     (*coarse_grid_solver_solve)(void*,void*,void*,void*);

  HYPRE_Int     use_default_cgrid_solver;
  HYPRE_Real    omega;

  /* temp vectors for solve phase */
  hypre_ParVector   *Vtemp;
  hypre_ParVector   *Ztemp;
  hypre_ParVector   *Utemp;
  hypre_ParVector   *Ftemp;

  HYPRE_Real          *diaginv;
  HYPRE_Int           n_block;
  HYPRE_Int           left_size;
  HYPRE_Int           global_smooth_iters;
  HYPRE_Int           global_smooth_type;
  HYPRE_Solver global_smoother;
  /*
   Number of points that remain part of the coarse grid throughout the hierarchy.
   For example, number of well equations
   */
  HYPRE_Int reserved_coarse_size;
  HYPRE_Int *reserved_coarse_indexes;
  HYPRE_Int *reserved_Cpoint_local_indexes;

  HYPRE_Int set_non_Cpoints_to_F;

  /* F-relaxation method */
  HYPRE_Int Frelax_method;
  /* V-cycle F relaxation method */
  hypre_ParAMGData    **FrelaxVcycleData;

  HYPRE_Int   max_local_lvls;

  HYPRE_Int   print_coarse_system;

} hypre_ParMGRData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define SMALLREAL 1e-20

#endif
