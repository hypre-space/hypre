/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header info for the SMG solver
 *
 *****************************************************************************/

#ifndef hypre_SMG_HEADER
#define hypre_SMG_HEADER

/*--------------------------------------------------------------------------
 * hypre_SMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
                      
   int                   memory_use;
   double                tol;
   int                   max_iter;
   int                   zero_guess;
   int                   max_levels;  /* max_level <= 0 means no limit */
                      
   int                   num_levels;
                      
   int                   num_pre_relax;  /* number of pre relaxation sweeps */
   int                   num_post_relax; /* number of post relaxation sweeps */

   /* base coarsening info */
   int                   cdir;  /* coarsening direction */
   int                   ci;    /* 1st coarse index in coarsening direction */
   int                   fi;    /* 1st fine index in coarsening direction */
   int                   cs;    /* coarse index stride */
   int                   fs;    /* fine index stride */

   /* base index space info */
   hypre_Index           base_index;
   hypre_Index           base_stride;

   /* base index space info for each grid level */
   hypre_Index          *base_index_l;
   hypre_Index          *base_stride_l;

   /* coarsening info for each grid level */
   hypre_Index          *cindex_l;
   hypre_Index          *findex_l;
   hypre_Index          *cstride_l;
   hypre_Index          *fstride_l;

   hypre_StructGrid    **grid_l;
                    
   hypre_StructMatrix  **A_l;
   hypre_StructMatrix  **PT_l;
   hypre_StructMatrix  **R_l;
                    
   hypre_StructVector  **b_l;
   hypre_StructVector  **x_l;

   /* temp vectors */
   hypre_StructVector  **tb_l;
   hypre_StructVector  **tx_l;
   hypre_StructVector  **r_l;
   hypre_StructVector  **e_l;

   void                **relax_data_l;
   void                **residual_data_l;
   void                **restrict_data_l;
   void                **intadd_data_l;

   /* log info (always logged) */
   int                   num_iterations;
   int                   time_index;

   /* additional log info (logged when `logging' > 0) */
   int                   logging;
   double               *norms;
   double               *rel_norms;

} hypre_SMGData;

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define hypre_SMGMapFineToCoarse(index1, index2, cindex, cstride) \
{\
   hypre_IndexX(index2) =\
      (hypre_IndexX(index1) - hypre_IndexX(cindex)) / hypre_IndexX(cstride);\
   hypre_IndexY(index2) =\
      (hypre_IndexY(index1) - hypre_IndexY(cindex)) / hypre_IndexY(cstride);\
   hypre_IndexZ(index2) =\
      (hypre_IndexZ(index1) - hypre_IndexZ(cindex)) / hypre_IndexZ(cstride);\
}

#define hypre_SMGMapCoarseToFine(index1, index2, cindex, cstride) \
{\
   hypre_IndexX(index2) =\
      hypre_IndexX(index1) * hypre_IndexX(cstride) + hypre_IndexX(cindex);\
   hypre_IndexY(index2) =\
      hypre_IndexY(index1) * hypre_IndexY(cstride) + hypre_IndexY(cindex);\
   hypre_IndexZ(index2) =\
      hypre_IndexZ(index1) * hypre_IndexZ(cstride) + hypre_IndexZ(cindex);\
}

#endif
