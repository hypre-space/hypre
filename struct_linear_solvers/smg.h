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
   int                   rel_change;
   int                   zero_guess;
   int                   max_levels;  /* max_level <= 0 means no limit */
                      
   int                   num_levels;
                      
   int                   num_pre_relax;  /* number of pre relaxation sweeps */
   int                   num_post_relax; /* number of post relaxation sweeps */

   int                   cdir;  /* coarsening direction */

   /* base index space info */
   hypre_Index           base_index;
   hypre_Index           base_stride;

   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **PT_grid_l;
                    
   double               *data;
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
   void                **interp_data_l;

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

#define hypre_SMGSetBIndex(base_index, base_stride, level, bindex) \
{\
   if (level > 0)\
      hypre_SetIndex(bindex, 0, 0, 0);\
   else\
      hypre_CopyIndex(base_index, bindex);\
}

#define hypre_SMGSetBStride(base_index, base_stride, level, bstride) \
{\
   if (level > 0)\
      hypre_SetIndex(bstride, 1, 1, 1);\
   else\
      hypre_CopyIndex(base_stride, bstride);\
}

#define hypre_SMGSetCIndex(base_index, base_stride, level, cdir, cindex) \
{\
   hypre_SMGSetBIndex(base_index, base_stride, level, cindex);\
   hypre_IndexD(cindex, cdir) += 0;\
}

#define hypre_SMGSetFIndex(base_index, base_stride, level, cdir, findex) \
{\
   hypre_SMGSetBIndex(base_index, base_stride, level, findex);\
   hypre_IndexD(findex, cdir) += 1;\
}

#define hypre_SMGSetStride(base_index, base_stride, level, cdir, stride) \
{\
   hypre_SMGSetBStride(base_index, base_stride, level, stride);\
   hypre_IndexD(stride, cdir) *= 2;\
}

#endif
