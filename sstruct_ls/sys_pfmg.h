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

#ifndef hypre_SYS_PFMG_HEADER
#define hypre_SYS_PFMG_HEADER

/*--------------------------------------------------------------------------
 * hypre_SysPFMGData:
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

   hypre_SStructPGrid    **grid_l;
   hypre_SStructPGrid    **P_grid_l;
                    
   double                 *data;
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
   int                   num_iterations;
   int                   time_index;

   /* additional log info (logged when `logging' > 0) */
   int                   logging;
   double               *norms;
   double               *rel_norms;

} hypre_SysPFMGData;

#endif
