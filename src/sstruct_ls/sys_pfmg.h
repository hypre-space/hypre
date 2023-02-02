/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the PFMG solver
 *
 *****************************************************************************/

#ifndef hypre_SYS_PFMG_HEADER
#define hypre_SYS_PFMG_HEADER

/*--------------------------------------------------------------------------
 * hypre_SysPFMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   HYPRE_Real            tol;
   HYPRE_Int             max_iter;
   HYPRE_Int             rel_change;
   HYPRE_Int             zero_guess;
   HYPRE_Int             max_levels;  /* max_level <= 0 means no limit */

   HYPRE_Int             relax_type;     /* type of relaxation to use */
   HYPRE_Real            jacobi_weight;  /* weighted jacobi weight */
   HYPRE_Int             usr_jacobi_weight; /* indicator flag for user weight */

   HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */
   HYPRE_Int             skip_relax;     /* flag to allow skipping relaxation */
   HYPRE_Real            dxyz[3];     /* parameters used to determine cdir */

   HYPRE_Int             num_levels;

   HYPRE_Int            *cdir_l;  /* coarsening directions */
   HYPRE_Int            *active_l;  /* flags to relax on level l*/

   hypre_SStructPGrid    **grid_l;
   hypre_SStructPGrid    **P_grid_l;

   HYPRE_Real             *data;
   hypre_SStructPMatrix  **A_l;
   hypre_SStructPMatrix  **P_l;
   hypre_SStructPMatrix  **RT_l;
   hypre_SStructPVector  **b_l;
   hypre_SStructPVector  **x_l;

   /* temp vectors */
   hypre_SStructPVector  **tx_l;
   hypre_SStructPVector  **r_l;
   hypre_SStructPVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   /* log info (always logged) */
   HYPRE_Int             num_iterations;
   HYPRE_Int             time_index;
   HYPRE_Int             print_level;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int             logging;
   HYPRE_Real           *norms;
   HYPRE_Real           *rel_norms;

} hypre_SysPFMGData;

#endif
