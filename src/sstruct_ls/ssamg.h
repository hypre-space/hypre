/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_SSAMG_HEADER
#define hypre_SSAMG_HEADER

//#define DEBUG_SETUP
//#define DEBUG_SOLVE

/*--------------------------------------------------------------------------
 * hypre_SSAMGData
 *--------------------------------------------------------------------------*/

typedef struct hypre_SSAMGData_struct
{
   hypre_MPI_Comm          comm;
   HYPRE_Int               nparts;

   HYPRE_Real              tol;                /* relative tolerance for convergence */
   HYPRE_Int               max_iter;           /* max. number of iterations */
   HYPRE_Int               max_levels;         /* max_level <= 0 means no limit */
   HYPRE_Int               rel_change;         /* tests convergence with rel change of x */
   HYPRE_Int               zero_guess;         /* initial guess is vector of zeros */
   HYPRE_Int               non_galerkin;       /* controls choice of RAP codes */
   HYPRE_Int               num_levels;         /* number of levels of the multigrid hierarchy */
   HYPRE_Int               num_pre_relax;      /* number of pre relaxation sweeps */
   HYPRE_Int               num_post_relax;     /* number of post relaxation sweeps */
   HYPRE_Int               skip_relax;         /* skip relaxation flag */
   HYPRE_Int               relax_type;         /* relaxation type flag */
   HYPRE_Real              usr_relax_weight;   /* user relax weight */
   HYPRE_Real              usr_set_rweight;    /* user sets relax weight */
   HYPRE_Real             *dxyz[HYPRE_MAXDIM]; /* nparts array used to determine cdir */

   /* Coarse solver data */
   HYPRE_Solver            csolver;
   HYPRE_IJMatrix          ij_Ac;
   hypre_ParVector        *par_b;
   hypre_ParVector        *par_x;
   HYPRE_Int               csolver_type;     /* coarse solver type */
   HYPRE_Int               num_coarse_relax; /* number of coarse relaxation sweeps */
   HYPRE_Int               max_coarse_size;  /* maximum size for the coarse grid */

   /* (nlevels x nparts) arrays */
   HYPRE_Int             **active_l;         /* active parts for relaxation */
   HYPRE_Int             **cdir_l;           /* coarsening directions */
   HYPRE_Real            **relax_weights;    /* relaxation weights */
   hypre_SStructGrid     **grid_l;           /* grids */

   /* work matrices and vectors */
   hypre_SStructMatrix   **A_l;
   hypre_SStructMatrix   **P_l;
   hypre_SStructMatrix   **RT_l;
   hypre_SStructVector   **b_l;
   hypre_SStructVector   **x_l;
   hypre_SStructVector   **r_l;
   hypre_SStructVector   **e_l;
   hypre_SStructVector   **tx_l;

   /* data structures for performing relaxation, interpolation,
      restriction and matrix-vector multiplication */
   void                  **relax_data_l;
   void                  **matvec_data_l;
   void                  **restrict_data_l;
   void                  **interp_data_l;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;
   HYPRE_Int               print_level;
   HYPRE_Int               print_freq;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int               logging;
   HYPRE_Real             *norms;
   HYPRE_Real             *rel_norms;

} hypre_SSAMGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the hypre_SSAMGData structure
 *--------------------------------------------------------------------------*/

#define hypre_SSAMGDataComm(ssamg_data)            ((ssamg_data) -> comm)
#define hypre_SSAMGDataNParts(ssamg_data)          ((ssamg_data) -> nparts)
#define hypre_SSAMGDataTol(ssamg_data)             ((ssamg_data) -> tol)
#define hypre_SSAMGDataNumLevels(ssamg_data)       ((ssamg_data) -> num_levels)
#define hypre_SSAMGDataMaxLevels(ssamg_data)       ((ssamg_data) -> max_levels)
#define hypre_SSAMGDataMaxIter(ssamg_data)         ((ssamg_data) -> max_iter)
#define hypre_SSAMGDataRelChange(ssamg_data)       ((ssamg_data) -> rel_change)
#define hypre_SSAMGDataZeroGuess(ssamg_data)       ((ssamg_data) -> zero_guess)
#define hypre_SSAMGDataNonGalerkin(ssamg_data)     ((ssamg_data) -> non_galerkin)
#define hypre_SSAMGDataNumIterations(ssamg_data)   ((ssamg_data) -> num_iterations)
#define hypre_SSAMGDataSkipRelax(ssamg_data)       ((ssamg_data) -> skip_relax)
#define hypre_SSAMGDataRelaxType(ssamg_data)       ((ssamg_data) -> relax_type)
#define hypre_SSAMGDataUsrRelaxWeight(ssamg_data)  ((ssamg_data) -> usr_relax_weight)
#define hypre_SSAMGDataUsrSetRWeight(ssamg_data)   ((ssamg_data) -> usr_set_rweight)
#define hypre_SSAMGDataRelaxWeights(ssamg_data)    ((ssamg_data) -> relax_weights)
#define hypre_SSAMGDataNumPreRelax(ssamg_data)     ((ssamg_data) -> num_pre_relax)
#define hypre_SSAMGDataNumPosRelax(ssamg_data)     ((ssamg_data) -> num_post_relax)
#define hypre_SSAMGDataNumCoarseRelax(ssamg_data)  ((ssamg_data) -> num_coarse_relax)
#define hypre_SSAMGDataMaxCoarseSize(ssamg_data)   ((ssamg_data) -> max_coarse_size)
#define hypre_SSAMGDataCSolverType(ssamg_data)     ((ssamg_data) -> csolver_type)
#define hypre_SSAMGDataTimeIndex(ssamg_data)       ((ssamg_data) -> time_index)
#define hypre_SSAMGDataPrintLevel(ssamg_data)      ((ssamg_data) -> print_level)
#define hypre_SSAMGDataPrintFreq(ssamg_data)       ((ssamg_data) -> print_freq)
#define hypre_SSAMGDataLogging(ssamg_data)         ((ssamg_data) -> logging)
#define hypre_SSAMGDataDxyz(ssamg_data)            ((ssamg_data) -> dxyz)
#define hypre_SSAMGDataDxyzD(ssamg_data, d)        ((ssamg_data) -> dxyz[d])
#define hypre_SSAMGDataActivel(ssamg_data)         ((ssamg_data) -> active_l)
#define hypre_SSAMGDataGridl(ssamg_data)           ((ssamg_data) -> grid_l)
#define hypre_SSAMGDataAl(ssamg_data)              ((ssamg_data) -> A_l)
#define hypre_SSAMGDataCdir(ssamg_data)            ((ssamg_data) -> cdir_l)
#define hypre_SSAMGDataNorms(ssamg_data)           ((ssamg_data) -> norms)
#define hypre_SSAMGDataRelNorms(ssamg_data)        ((ssamg_data) -> rel_norms)

#endif
