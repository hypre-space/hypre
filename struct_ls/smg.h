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

#ifndef zzz_SMG_HEADER
#define zzz_SMG_HEADER


/*--------------------------------------------------------------------------
 * zzz_SMGData:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm           *comm;

   int                 memory_use;
   double              tol;
   int                 max_iter;
   int                 zero_guess;
   int                 max_levels;  /* max_level <= 0 means no limit */
                    
   int                 num_levels;

   int                 num_pre_relax;  /* number of pre relaxation sweeps */
   int                 num_post_relax; /* number of post relaxation sweeps */

   /* base coarsening info */
   int                 cdir;    /* coarsening direction */
   int                 ci;      /* 1st coarse index in coarsening direction */
   int                 fi;      /* 1st fine index in coarsening direction */
   int                 cs;      /* coarse index stride */
   int                 fs;      /* fine index stride */

   /* base index space info */
   zzz_Index          *base_index;
   zzz_Index          *base_stride;

   /* base index space info for each grid level */
   zzz_Index         **base_index_l;
   zzz_Index         **base_stride_l;

   /* coarsening info for each grid level */
   zzz_Index         **cindex_l;
   zzz_Index         **findex_l;
   zzz_Index         **cstride_l;
   zzz_Index         **fstride_l;

   zzz_StructGrid    **grid_l;
                    
   zzz_StructMatrix  **A_l;
   zzz_StructMatrix  **PT_l;
   zzz_StructMatrix  **R_l;
                    
   zzz_StructVector  **b_l;
   zzz_StructVector  **x_l;

   /* temp vectors */
   zzz_StructVector  **tb_l;
   zzz_StructVector  **tx_l;
   zzz_StructVector  **r_l;
   zzz_StructVector  **e_l;

   void              **relax_data_l;
   void              **residual_data_l;
   void              **restrict_data_l;
   void              **intadd_data_l;

   /* log info (always logged) */
   int                 num_iterations;
   int                 time_index;

   /* additional log info (logged when `logging' > 0) */
   int                 logging;
   double             *norms;
   double             *rel_norms;

} zzz_SMGData;

/*--------------------------------------------------------------------------
 * Utility routines:
 *--------------------------------------------------------------------------*/

#define zzz_SMGMapFineToCoarse(index1, index2, cindex, cstride) \
{\
   zzz_IndexX(index2) =\
      (zzz_IndexX(index1) - zzz_IndexX(cindex)) / zzz_IndexX(cstride);\
   zzz_IndexY(index2) =\
      (zzz_IndexY(index1) - zzz_IndexY(cindex)) / zzz_IndexY(cstride);\
   zzz_IndexZ(index2) =\
      (zzz_IndexZ(index1) - zzz_IndexZ(cindex)) / zzz_IndexZ(cstride);\
}

#define zzz_SMGMapCoarseToFine(index1, index2, cindex, cstride) \
{\
   zzz_IndexX(index2) =\
      zzz_IndexX(index1) * zzz_IndexX(cstride) + zzz_IndexX(cindex);\
   zzz_IndexY(index2) =\
      zzz_IndexY(index1) * zzz_IndexY(cstride) + zzz_IndexY(cindex);\
   zzz_IndexZ(index2) =\
      zzz_IndexZ(index1) * zzz_IndexZ(cstride) + zzz_IndexZ(cindex);\
}

#endif
