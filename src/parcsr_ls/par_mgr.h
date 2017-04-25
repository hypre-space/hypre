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
  HYPRE_Int  *tmp_num_coarse_points;
  HYPRE_Int  num_additional_coarse_indices;
  HYPRE_Int  *block_cf_marker;
  HYPRE_Int  **tmp_block_cf_marker;
  HYPRE_Int  *additional_coarse_indices;

   //general data
  HYPRE_Int num_coarse_levels;
  HYPRE_Int max_num_coarse_levels;
  hypre_ParCSRMatrix **A_array;
  hypre_ParCSRMatrix ** A_ff_array;
  hypre_ParCSRMatrix **P_array;
  hypre_ParCSRMatrix **P_f_array;
  hypre_ParCSRMatrix **RT_array;
  hypre_ParCSRMatrix *P_f;
  hypre_ParCSRMatrix *RAP;
  hypre_ParCSRMatrix *A_ff;
  HYPRE_Int **CF_marker_array;
  HYPRE_Int *final_coarse_indexes;
  HYPRE_Int **coarse_indices_lvls;
  hypre_ParVector    **F_array;
  hypre_ParVector    **U_array;
  hypre_ParVector    *residual;
  HYPRE_Real    *rel_res_norms;

  HYPRE_Real   max_row_sum;
  HYPRE_Real	num_interp_sweeps;
  HYPRE_Int    interp_type;
  HYPRE_Int   restrict_type;
  HYPRE_Real   strong_threshold;
  HYPRE_Real   trunc_factor;
  HYPRE_Real   S_commpkg_switch;
  HYPRE_Int	P_max_elmts;
  HYPRE_Int  	num_iterations;

  HYPRE_Real   **l1_norms;
  HYPRE_Real	final_rel_residual_norm;
  HYPRE_Real	conv_tol;
  HYPRE_Real	relax_weight;
  HYPRE_Int	relax_type;
  HYPRE_Int	logging;
  HYPRE_Int	print_level;
  HYPRE_Int	max_iter;
  HYPRE_Int	relax_order;
  HYPRE_Int	num_relax_sweeps;
  HYPRE_Int relax_method;

  HYPRE_Solver *coarse_grid_solver;
  HYPRE_Int	(*coarse_grid_solver_setup)(void*,void*,void*,void*);
  HYPRE_Int	(*coarse_grid_solver_solve)(void*,void*,void*,void*);

  HYPRE_Solver *fine_grid_solver;
  HYPRE_Int   (*fine_grid_solver_setup)(void*,void*,void*,void*);
  HYPRE_Int   (*fine_grid_solver_solve)(void*,void*,void*,void*);

  HYPRE_Solver global_smoother;
  HYPRE_Solver *aff_solver;

  HYPRE_Int	use_default_cgrid_solver;
  HYPRE_Real	omega;

  /* temp vectors for solve phase */
  hypre_ParVector   *Vtemp;
  hypre_ParVector   *Ztemp;
  hypre_ParVector   *Utemp;
  hypre_ParVector   *Ftemp;
  hypre_ParVector   **U_fine_array;
  hypre_ParVector   **F_fine_array;

  HYPRE_Real          *diaginv;
  HYPRE_Int           n_block;
  HYPRE_Int           left_size;
  HYPRE_Int           global_smooth;
  HYPRE_Int           global_smooth_type;

  /* 
   Number of points that remain part of the coarse grid throughout the hierarchy.
   For example, number of well equations
   */
  HYPRE_Int reserved_coarse_size;
  
  HYPRE_Int block_form;

  HYPRE_Int build_aff;
  HYPRE_Int splitting_strategy;
} hypre_ParMGRData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define SMALLREAL 1e-20

#endif
