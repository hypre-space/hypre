/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParMGR_DATA_HEADER
#define hypre_ParMGR_DATA_HEADER

/*--------------------------------------------------------------------------
 * MGR print level codes
 *--------------------------------------------------------------------------*/

#define HYPRE_MGR_PRINT_INFO_SETUP  0x01       /*   1 (1st bit) */
#define HYPRE_MGR_PRINT_INFO_SOLVE  0x02       /*   2 (2nd bit) */
#define HYPRE_MGR_PRINT_INFO_PARAMS 0x04       /*   4 (3rd bit) */
#define HYPRE_MGR_PRINT_MODE_ASCII  0x08       /*   8 (4th bit) */
#define HYPRE_MGR_PRINT_FINE_MATRIX 0x10       /*  16 (5th bit) */
#define HYPRE_MGR_PRINT_FINE_RHS    0x20       /*  32 (6th bit) */
#define HYPRE_MGR_PRINT_CRSE_MATRIX 0x40       /*  64 (7th bit) */
#define HYPRE_MGR_PRINT_LVLS_MATRIX 0x80       /* 128 (8th bit) */
/* ... */
/* Reserved codes */
#define HYPRE_MGR_PRINT_RESERVED_C  0x10000000 /*  268435456 (29th bit) */
#define HYPRE_MGR_PRINT_RESERVED_B  0x20000000 /*  536870912 (30th bit) */
#define HYPRE_MGR_PRINT_RESERVED_A  0x40000000 /* 1073741824 (31th bit) */

/*--------------------------------------------------------------------------
 * hypre_ParMGRData
 *--------------------------------------------------------------------------*/

typedef struct
{
   /* block data */
   HYPRE_Int             block_size;
   HYPRE_Int            *block_num_coarse_indexes;
   HYPRE_Int            *point_marker_array;
   HYPRE_Int           **block_cf_marker;

   /* initial setup data (user provided) */
   HYPRE_Int             num_coarse_levels;
   HYPRE_Int            *num_coarse_per_level;
   HYPRE_Int           **level_coarse_indexes;

   /* general data */
   HYPRE_Int             max_num_coarse_levels;
   hypre_ParCSRMatrix  **A_array;
   hypre_ParCSRMatrix  **B_array;    /* block diagonal inverse matrices */
   hypre_ParCSRMatrix  **B_FF_array; /* block-FF diagonal inverse matrices */
   hypre_ParCSRMatrix  **A_ff_array;
#if defined(HYPRE_USING_GPU)
   hypre_ParCSRMatrix  **P_FF_array;
#endif
   hypre_ParCSRMatrix  **P_array;
   hypre_ParCSRMatrix  **R_array;
   hypre_ParCSRMatrix  **RT_array;
   hypre_ParCSRMatrix   *RAP;
   hypre_IntArray      **CF_marker_array;
   HYPRE_Int           **coarse_indices_lvls;
   hypre_ParVector     **F_array;
   hypre_ParVector     **U_array;
   hypre_ParVector      *residual;
   HYPRE_Real           *rel_res_norms;

   hypre_ParVector     **F_fine_array;
   hypre_ParVector     **U_fine_array;
   HYPRE_Solver        **aff_solver;
   HYPRE_Int           (*fine_grid_solver_setup)(void*, void*, void*, void*);
   HYPRE_Int           (*fine_grid_solver_solve)(void*, void*, void*, void*);

   HYPRE_Real            max_row_sum;
   HYPRE_Int             num_interp_sweeps;
   HYPRE_Int             num_restrict_sweeps;
   HYPRE_Int            *interp_type;
   HYPRE_Int            *restrict_type;
   HYPRE_Real            strong_threshold;
   HYPRE_Real            trunc_factor;
   HYPRE_Real            S_commpkg_switch;
   HYPRE_Int            *P_max_elmts;
   HYPRE_Int             num_iterations;

   hypre_Vector        **l1_norms;
   HYPRE_Real            final_rel_residual_norm;
   HYPRE_Real            tol;
   HYPRE_Real            relax_weight;
   HYPRE_Int             relax_type;
   HYPRE_Int             logging;
   HYPRE_Int             print_level;
   HYPRE_Int             frelax_print_level;
   HYPRE_Int             cg_print_level;
   HYPRE_Int             max_iter;
   HYPRE_Int             relax_order;
   HYPRE_Int            *num_relax_sweeps;
   char                 *data_path;

   HYPRE_Solver          coarse_grid_solver;
   HYPRE_Int           (*coarse_grid_solver_setup)(void*, void*, void*, void*);
   HYPRE_Int           (*coarse_grid_solver_solve)(void*, void*, void*, void*);

   HYPRE_Int             use_default_cgrid_solver;
   // Mode to use an external AMG solver for F-relaxation
   // 0: use an external AMG solver that is already setup
   // 1: use an external AMG solver but do setup inside MGR
   // 2: use default internal AMG solver
   HYPRE_Int             fsolver_mode;
   //  HYPRE_Int          fsolver_type;
   HYPRE_Real            omega;

   /* temp vectors for solve phase */
   hypre_ParVector      *Vtemp;
   hypre_ParVector      *Ztemp;
   hypre_ParVector      *Utemp;
   hypre_ParVector      *Ftemp;

   HYPRE_Real          **level_diaginv;
   HYPRE_Real          **frelax_diaginv;
   HYPRE_Int             n_block;
   HYPRE_Int             left_size;
   HYPRE_Int            *blk_size;
   HYPRE_Int            *level_smooth_iters;
   HYPRE_Int            *level_smooth_type;
   HYPRE_Solver         *level_smoother;
   HYPRE_Int             global_smooth_cycle;

   /*
    Number of points that remain part of the coarse grid throughout the hierarchy.
    For example, number of well equations
    */
   HYPRE_Int             reserved_coarse_size;
   HYPRE_BigInt         *reserved_coarse_indexes;
   HYPRE_Int            *reserved_Cpoint_local_indexes;

   HYPRE_Int             set_non_Cpoints_to_F;
   HYPRE_BigInt         *idx_array;

   /* F-relaxation type */
   HYPRE_Int            *Frelax_method;
   HYPRE_Int            *Frelax_type;
   HYPRE_Int            *Frelax_num_functions;

   /* Non-Galerkin coarse grid */
   HYPRE_Int            *mgr_coarse_grid_method; /* TODO (VPM): Change name? remove mgr_?*/

   /* V-cycle F relaxation method */
   hypre_ParAMGData    **FrelaxVcycleData;
   hypre_ParVector      *VcycleRelaxVtemp;
   hypre_ParVector      *VcycleRelaxZtemp;

   HYPRE_Int             max_local_lvls;

   HYPRE_Int             print_coarse_system;
   HYPRE_Real            truncate_coarse_grid_threshold;

   /* how to set C points */
   HYPRE_Int             set_c_points_method;

   /* reduce reserved C-points before coarse grid solve? */
   /* this might be necessary for some applications, e.g. phase transitions */
   HYPRE_Int             lvl_to_keep_cpoints;

   /* block size for block Jacobi interpolation and relaxation */
   HYPRE_Int             block_jacobi_bsize;

   HYPRE_Real            cg_convergence_factor;

   /* Data for Gaussian elimination F-relaxation */
   hypre_ParAMGData    **GSElimData;
} hypre_ParMGRData;

/*--------------------------------------------------------------------------
 * hypre_MGRRelaxData
 *--------------------------------------------------------------------------*/

/* F-relaxation struct for future refactoring of F-relaxation in MGR */
typedef struct
{
   HYPRE_Int             relax_type;
   HYPRE_Int             relax_nsweeps;

   hypre_ParCSRMatrix   *A;
   hypre_ParVector      *b;

   /* for hypre's smoother options */
   HYPRE_Int            *CF_marker;

   /* for block Jacobi/GS option */
   HYPRE_Complex        *diaginv;

   /* for ILU option */
   HYPRE_Solver          frelax_solver;
} hypre_MGRRelaxData;

#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

/*--------------------------------------------------------------------------
 * Acessor macros
 *--------------------------------------------------------------------------*/

/* TODO (VPM): add remaining acessor macros */
#define hypre_ParMGRDataNumCoarseLevels(data)       ((data) -> num_coarse_levels)     /* TODO (VPM): change to num_levels ? */
#define hypre_ParMGRDataMaxCoarseLevels(data)       ((data) -> max_num_coarse_levels) /* TODO (VPM): change to max_levels ? */

#define hypre_ParMGRDataAArray(data)                ((data) -> A_array)
#define hypre_ParMGRDataA(data, i)                  ((data) -> A_array[i])
#define hypre_ParMGRDataBArray(data)                ((data) -> B_array)
#define hypre_ParMGRDataB(data, i)                  ((data) -> B_array[i])
#define hypre_ParMGRDataPArray(data)                ((data) -> P_array)
#define hypre_ParMGRDataP(data, i)                  ((data) -> P_array[i])
#define hypre_ParMGRDataRTArray(data)               ((data) -> RT_array)
#define hypre_ParMGRDataRT(data, i)                 ((data) -> RT_array[i])
#define hypre_ParMGRDataBFFArray(data)              ((data) -> B_FF_array)
#define hypre_ParMGRDataBFF(data, i)                ((data) -> B_FF_array[i])
#define hypre_ParMGRDataRAP(data)                   ((data) -> RAP)

#define hypre_ParMGRDataInterpType(data)            ((data) -> interp_type)
#define hypre_ParMGRDataInterpTypeI(data, i)        ((data) -> interp_type[i])
#define hypre_ParMGRDataRestrictType(data)          ((data) -> restrict_type)
#define hypre_ParMGRDataRestrictTypeI(data, i)      ((data) -> restrict_type[i])

#define hypre_ParMGRDataLevelSmoothType(data)       ((data) -> level_smooth_type)
#define hypre_ParMGRDataLevelSmoothTypeI(data, i)   ((data) -> level_smooth_type[i])
#define hypre_ParMGRDataLevelSmoother(data)         ((data) -> level_smoother)
#define hypre_ParMGRDataLevelSmootherI(data, i)     ((data) -> level_smoother[i])

#define hypre_ParMGRDataRelaxType(data)             ((data) -> relax_type)
#define hypre_ParMGRDataFRelaxType(data)            ((data) -> Frelax_type)
#define hypre_ParMGRDataFRelaxTypeI(data, i)        ((data) -> Frelax_type[i])
#define hypre_ParMGRDataAFFsolver(data)             ((data) -> aff_solver)
#define hypre_ParMGRDataAFFsolverI(data)            ((data) -> aff_solver[i])

#define hypre_ParMGRDataCoarseGridMethod(data)      ((data) -> mgr_coarse_grid_method)
#define hypre_ParMGRDataCoarseGridMethodI(data, i)  ((data) -> mgr_coarse_grid_method[i])
#define hypre_ParMGRDataCoarseGridSolver(data)      ((data) -> coarse_grid_solver)
#define hypre_ParMGRDataCoarseGridSolverSetup(data) ((data) -> coarse_grid_solver_setup)

#endif
