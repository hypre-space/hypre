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
   double                dxyz[3];     /* parameters used to determine cdir */

   int                   num_levels;
                      
   int                  *cdir_l;  /* coarsening directions */

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

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define hypre_PFMGMapFineToCoarse(index_in, cfindex, stride, index_out) \
{\
   hypre_IndexX(index_out) =\
      (hypre_IndexX(index_in) - hypre_IndexX(cfindex)) / hypre_IndexX(stride);\
   hypre_IndexY(index_out) =\
      (hypre_IndexY(index_in) - hypre_IndexY(cfindex)) / hypre_IndexY(stride);\
   hypre_IndexZ(index_out) =\
      (hypre_IndexZ(index_in) - hypre_IndexZ(cfindex)) / hypre_IndexZ(stride);\
}

#define hypre_PFMGMapCoarseToFine(index_in, cfindex, stride, index_out) \
{\
   hypre_IndexX(index_out) =\
      hypre_IndexX(index_in) * hypre_IndexX(stride) + hypre_IndexX(cfindex);\
   hypre_IndexY(index_out) =\
      hypre_IndexY(index_in) * hypre_IndexY(stride) + hypre_IndexY(cfindex);\
   hypre_IndexZ(index_out) =\
      hypre_IndexZ(index_in) * hypre_IndexZ(stride) + hypre_IndexZ(cfindex);\
}

#define hypre_PFMGSetCIndex(cdir, cindex) \
{\
   hypre_SetIndex(cindex, 0, 0, 0);\
   hypre_IndexD(cindex, cdir) = 0;\
}

#define hypre_PFMGSetFIndex(cdir, findex) \
{\
   hypre_SetIndex(findex, 0, 0, 0);\
   hypre_IndexD(findex, cdir) = 1;\
}

#define hypre_PFMGSetStride(cdir, stride) \
{\
   hypre_SetIndex(stride, 1, 1, 1);\
   hypre_IndexD(stride, cdir) = 2;\
}

#endif
