/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/

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
                      
   HYPRE_Int             memory_use;
   double                tol;
   HYPRE_Int             max_iter;
   HYPRE_Int             rel_change;
   HYPRE_Int             zero_guess;
   HYPRE_Int             max_levels;  /* max_level <= 0 means no limit */
                      
   HYPRE_Int             num_levels;
                      
   HYPRE_Int             num_pre_relax;  /* number of pre relaxation sweeps */
   HYPRE_Int             num_post_relax; /* number of post relaxation sweeps */

   HYPRE_Int             cdir;  /* coarsening direction */

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
   HYPRE_Int             num_iterations;
   HYPRE_Int             time_index;

   HYPRE_Int            print_level;

   /* additional log info (logged when `logging' > 0) */
   HYPRE_Int             logging;
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
