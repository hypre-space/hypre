/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the SparseMSG solver
 *
 *****************************************************************************/

#ifndef hypre_SparseMSG_HEADER
#define hypre_SparseMSG_HEADER

/*--------------------------------------------------------------------------
 * hypre_SparseMSGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   HYPRE_Real            tol;
   HYPRE_Int             max_iter;
   HYPRE_Int             rel_change;
   HYPRE_Int             zero_guess;
   HYPRE_Int             jump;

   HYPRE_Int             relax_type;     /* type of relaxation to use */
   HYPRE_Real            jacobi_weight;  /* weighted jacobi weight */
   HYPRE_Int             usr_jacobi_weight; /* indicator flag for user weight */

   HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */
   HYPRE_Int             num_fine_relax; /* number of fine relaxation sweeps */

   HYPRE_Int             num_grids[3];   /* number of grids in each dim */
   HYPRE_Int          num_all_grids;
   HYPRE_Int          num_levels;

   hypre_StructGrid    **grid_array;
   hypre_StructGrid    **Px_grid_array;
   hypre_StructGrid    **Py_grid_array;
   hypre_StructGrid    **Pz_grid_array;

   HYPRE_MemoryLocation  memory_location; /* memory location of data */
   HYPRE_Real           *data;
   hypre_StructMatrix  **A_array;
   hypre_StructMatrix  **Px_array;
   hypre_StructMatrix  **Py_array;
   hypre_StructMatrix  **Pz_array;
   hypre_StructMatrix  **RTx_array;
   hypre_StructMatrix  **RTy_array;
   hypre_StructMatrix  **RTz_array;
   hypre_StructVector  **b_array;
   hypre_StructVector  **x_array;

   /* temp vectors */
   hypre_StructVector  **t_array;
   hypre_StructVector  **r_array;
   hypre_StructVector  **e_array;

   hypre_StructVector  **visitx_array;
   hypre_StructVector  **visity_array;
   hypre_StructVector  **visitz_array;
   HYPRE_Int            *grid_on;

   void                **relax_array;
   void                **matvec_array;
   void                **restrictx_array;
   void                **restricty_array;
   void                **restrictz_array;
   void                **interpx_array;
   void                **interpy_array;
   void                **interpz_array;

   /* log info (always logged) */
   HYPRE_Int             num_iterations;
   HYPRE_Int             time_index;
   HYPRE_Int             print_level;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int             logging;
   HYPRE_Real           *norms;
   HYPRE_Real           *rel_norms;

} hypre_SparseMSGData;

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define hypre_SparseMSGMapIndex(lx, ly, lz, nl, index) \
index = (lx) + ((ly) * nl[0]) + ((lz) * nl[0] * nl[1])

#endif
