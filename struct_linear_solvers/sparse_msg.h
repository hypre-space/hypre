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

#define hypre_SparseMSGMapFineToCoarse(index_in, cfindex, stride, index_out) \
{\
   hypre_IndexX(index_out) =\
      (hypre_IndexX(index_in) - hypre_IndexX(cfindex)) / hypre_IndexX(stride);\
   hypre_IndexY(index_out) =\
      (hypre_IndexY(index_in) - hypre_IndexY(cfindex)) / hypre_IndexY(stride);\
   hypre_IndexZ(index_out) =\
      (hypre_IndexZ(index_in) - hypre_IndexZ(cfindex)) / hypre_IndexZ(stride);\
}

#define hypre_SparseMSGMapCoarseToFine(index_in, cfindex, stride, index_out) \
{\
   hypre_IndexX(index_out) =\
      hypre_IndexX(index_in) * hypre_IndexX(stride) + hypre_IndexX(cfindex);\
   hypre_IndexY(index_out) =\
      hypre_IndexY(index_in) * hypre_IndexY(stride) + hypre_IndexY(cfindex);\
   hypre_IndexZ(index_out) =\
      hypre_IndexZ(index_in) * hypre_IndexZ(stride) + hypre_IndexZ(cfindex);\
}

#define hypre_SparseMSGSetCIndex(cdir, cindex) \
{\
   hypre_SetIndex(cindex, 0, 0, 0);\
   hypre_IndexD(cindex, cdir) = 0;\
}

#define hypre_SparseMSGSetFIndex(cdir, findex) \
{\
   hypre_SetIndex(findex, 0, 0, 0);\
   hypre_IndexD(findex, cdir) = 1;\
}

#define hypre_SparseMSGSetStride(cdir, stride) \
{\
   hypre_SetIndex(stride, 1, 1, 1);\
   hypre_IndexD(stride, cdir) = 2;\
}

#define hypre_SparseMSGComputeArrayIndex(lx, ly, lz, num_levels_x, num_levels_y,index) \
{\
   index = lx + (ly * num_levels_x) + (lz * num_levels_x * num_levels_y);\
}

#endif
