/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParMGR_DATA_HEADER
#define hypre_ParMGR_DATA_HEADER
/*--------------------------------------------------------------------------
 * hypre_ParMGRData
 *--------------------------------------------------------------------------*/
typedef struct
{
  // block data
  HYPRE_Int  block_size;
  HYPRE_Int  *block_num_coarse_indexes;
  HYPRE_Int  *point_marker_array;
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

  hypre_ParCSRMatrix  **A_ff_array;
  hypre_ParVector    **F_fine_array;
  hypre_ParVector    **U_fine_array;
  HYPRE_Solver **aff_solver;
  HYPRE_Int   (*fine_grid_solver_setup)(void*,void*,void*,void*);
  HYPRE_Int   (*fine_grid_solver_solve)(void*,void*,void*,void*);

  HYPRE_Real   max_row_sum;
  HYPRE_Int    num_interp_sweeps;
  HYPRE_Int    num_restrict_sweeps;
  //HYPRE_Int    interp_type;
  HYPRE_Int    *interp_type;
  HYPRE_Int    *restrict_type;
  HYPRE_Real   strong_threshold;
  HYPRE_Real   trunc_factor;
  HYPRE_Real   S_commpkg_switch;
  HYPRE_Int    P_max_elmts;
  HYPRE_Int    num_iterations;

  hypre_Vector **l1_norms;
  HYPRE_Real    final_rel_residual_norm;
  HYPRE_Real    tol;
  HYPRE_Real    relax_weight;
  HYPRE_Int     relax_type;
  HYPRE_Int     logging;
  HYPRE_Int     print_level;
  HYPRE_Int     frelax_print_level;
  HYPRE_Int     cg_print_level;
  HYPRE_Int     max_iter;
  HYPRE_Int     relax_order;
  HYPRE_Int     num_relax_sweeps;

  HYPRE_Solver coarse_grid_solver;
  HYPRE_Int     (*coarse_grid_solver_setup)(void*,void*,void*,void*);
  HYPRE_Int     (*coarse_grid_solver_solve)(void*,void*,void*,void*);

  HYPRE_Int     use_default_cgrid_solver;
  HYPRE_Int     use_default_fsolver;
//  HYPRE_Int     fsolver_type;
  HYPRE_Real    omega;

  /* temp vectors for solve phase */
  hypre_ParVector   *Vtemp;
  hypre_ParVector   *Ztemp;
  hypre_ParVector   *Utemp;
  hypre_ParVector   *Ftemp;

  HYPRE_Real          *diaginv;
  hypre_ParCSRMatrix  *A_ff_inv;
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
  HYPRE_BigInt *reserved_coarse_indexes;
  HYPRE_Int *reserved_Cpoint_local_indexes;

  HYPRE_Int set_non_Cpoints_to_F;
  HYPRE_BigInt *idx_array;

  /* F-relaxation method */
  HYPRE_Int *Frelax_method;
  HYPRE_Int *Frelax_num_functions;

  /* Non-Galerkin coarse grid */
  HYPRE_Int *use_non_galerkin_cg;

  /* V-cycle F relaxation method */
  hypre_ParAMGData    **FrelaxVcycleData;
  hypre_ParVector   *VcycleRelaxVtemp;
  hypre_ParVector   *VcycleRelaxZtemp;

  HYPRE_Int   max_local_lvls;

  HYPRE_Int   print_coarse_system;
  HYPRE_Real  truncate_coarse_grid_threshold;

  /* how to set C points */
  HYPRE_Int   set_c_points_method;

  /* reduce reserved C-points before coarse grid solve? */
  /* this might be necessary for some applications, e.g. phase transitions */
  HYPRE_Int   lvl_to_keep_cpoints;

  HYPRE_Real  cg_convergence_factor;

} hypre_ParMGRData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

#define SMALLREAL 1e-20
#define DIVIDE_TOL 1e-32

#endif
