/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
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
                      
   double                tol;
   int                   max_iter;
   int                   rel_change;
   int                   zero_guess;
   int                   jump;

   int                   relax_type;     /* type of relaxation to use */
   int                   num_pre_relax;  /* number of pre relaxation sweeps */
   int                   num_post_relax; /* number of post relaxation sweeps */

   int                   num_levels[3];  /* number of levels in each direction */
   int    		 total_num_levels;  /* actual number of levels */
   int                   num_grids;      /* number of total grids */
                      
   double               *restrict_weights; /* restriction weights and coarsening directions */
   double               *interp_weights;   /* interpolation weights */

   hypre_StructGrid    **grid_array;
   hypre_StructGrid    **P_grid_array;
                    
   hypre_StructMatrix  **A_array;
   hypre_StructMatrix  **P_array;
   hypre_StructMatrix  **RT_array;
   hypre_StructVector  **b_array;
   hypre_StructVector  **x_array;

   /* temp vectors */
   hypre_StructVector  **tx_array;
   hypre_StructVector  **r_array;
   hypre_StructVector  **e_array;

   void                **relax_data_array;
   void                **matvec_data_array;
   void                **restrict_data_array;
   void                **interp_data_array;

   /* log info (always logged) */
   int                   num_iterations;
   int                   time_index;

   /* additional log info (logged when `logging' > 0) */
   int                   logging;
   double               *norms;
   double               *rel_norms;

} hypre_SparseMSGData;

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define hypre_SparseMSGSetArrayIndex(lx, ly, lz, nl, index) \
{\
   index = lx + (ly * nl[0]) + (lz * nl[0] * nl[1]);\
}

#endif
