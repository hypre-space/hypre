/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
#if defined(HYPRE_USING_GPU)
   hypre_ParCSRMatrix **P_FF_array;
#endif
   hypre_ParCSRMatrix **P_array;
   hypre_ParCSRMatrix **RT_array;
   hypre_ParCSRMatrix *RAP;
   hypre_IntArray    **CF_marker_array;
   HYPRE_Int **coarse_indices_lvls;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;
   hypre_ParVector    *residual;
   HYPRE_Real    *rel_res_norms;

   hypre_ParCSRMatrix  **A_ff_array;
   hypre_ParVector    **F_fine_array;
   hypre_ParVector    **U_fine_array;
   HYPRE_Solver **aff_solver;
   HYPRE_Int   (*fine_grid_solver_setup)(void*, void*, void*, void*);
   HYPRE_Int   (*fine_grid_solver_solve)(void*, void*, void*, void*);

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
   HYPRE_Int     *num_relax_sweeps;

   HYPRE_Solver coarse_grid_solver;
   HYPRE_Int     (*coarse_grid_solver_setup)(void*, void*, void*, void*);
   HYPRE_Int     (*coarse_grid_solver_solve)(void*, void*, void*, void*);

   HYPRE_Int     use_default_cgrid_solver;
   // Mode to use an external AMG solver for F-relaxation
   // 0: use an external AMG solver that is already setup
   // 1: use an external AMG solver but do setup inside MGR
   // 2: use default internal AMG solver
   HYPRE_Int     fsolver_mode;
   //  HYPRE_Int     fsolver_type;
   HYPRE_Real    omega;

   /* temp vectors for solve phase */
   hypre_ParVector   *Vtemp;
   hypre_ParVector   *Ztemp;
   hypre_ParVector   *Utemp;
   hypre_ParVector   *Ftemp;

   HYPRE_Real          **level_diaginv;
   HYPRE_Real          **frelax_diaginv;
   HYPRE_Int           n_block;
   HYPRE_Int           left_size;
   HYPRE_Int           *blk_size;
   HYPRE_Int           *level_smooth_iters;
   HYPRE_Int           *level_smooth_type;
   HYPRE_Solver        *level_smoother;
   HYPRE_Int           global_smooth_cycle;

   /*
    Number of points that remain part of the coarse grid throughout the hierarchy.
    For example, number of well equations
    */
   HYPRE_Int reserved_coarse_size;
   HYPRE_BigInt *reserved_coarse_indexes;
   HYPRE_Int *reserved_Cpoint_local_indexes;

   HYPRE_Int set_non_Cpoints_to_F;
   HYPRE_BigInt *idx_array;

   /* F-relaxation type */
   HYPRE_Int *Frelax_method;
   HYPRE_Int *Frelax_type;

   HYPRE_Int *Frelax_num_functions;

   /* Non-Galerkin coarse grid */
   HYPRE_Int *mgr_coarse_grid_method;

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

   /* block size for block Jacobi interpolation and relaxation */
   HYPRE_Int  block_jacobi_bsize;

   HYPRE_Real  cg_convergence_factor;

   /* Data for Gaussian elimination F-relaxation */
   hypre_ParAMGData    **GSElimData;

} hypre_ParMGRData;

// F-relaxation struct for future refactoring of F-relaxation in MGR
typedef struct
{
   HYPRE_Int relax_type;
   HYPRE_Int relax_nsweeps;

   hypre_ParCSRMatrix *A;
   hypre_ParVector    *b;

   // for hypre's smoother options
   HYPRE_Int *CF_marker;

   // for block Jacobi/GS option
   HYPRE_Complex *diaginv;

   // for ILU option
   HYPRE_Solver frelax_solver;

} hypre_MGRRelaxData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

//#define SMALLREAL 1e-20
//#define DIVIDE_TOL 1e-32

#endif
