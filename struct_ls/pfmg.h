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
 * Header info for the PFMG solver
 *
 *****************************************************************************/

#ifndef hypre_PFMG_HEADER
#define hypre_PFMG_HEADER

/*--------------------------------------------------------------------------
 * hypre_PFMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
                      
   double                tol;
   int                   max_iter;
   int                   rel_change;
   int                   zero_guess;
   int                   max_levels;  /* max_level <= 0 means no limit */
                      
   int                   relax_type;     /* type of relaxation to use */
   int                   num_pre_relax;  /* number of pre relaxation sweeps */
   int                   num_post_relax; /* number of post relaxation sweeps */
   int                   skip_relax;     /* flag to allow skipping relaxation */
   double                dxyz[3];     /* parameters used to determine cdir */

   int                   num_levels;
                      
   int                  *cdir_l;  /* coarsening directions */
   int                  *active_l;  /* flags to relax on level l*/

   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **P_grid_l;
                    
   double               *data;
   hypre_StructMatrix  **A_l;
   hypre_StructMatrix  **P_l;
   hypre_StructMatrix  **RT_l;
   hypre_StructVector  **b_l;
   hypre_StructVector  **x_l;

   /* temp vectors */
   hypre_StructVector  **tx_l;
   hypre_StructVector  **r_l;
   hypre_StructVector  **e_l;

   void                **relax_data_l;
   void                **matvec_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   /* log info (always logged) */
   int                   num_iterations;
   int                   time_index;

   /* additional log info (logged when `logging' > 0) */
   int                   logging;
   double               *norms;
   double               *rel_norms;

} hypre_PFMGData;

#endif
