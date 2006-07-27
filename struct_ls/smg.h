/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

   int                  print_level;

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
