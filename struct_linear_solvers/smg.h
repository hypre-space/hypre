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

   int                 max_iter;
   int                 max_levels;  /* max_level <= 0 means no limit */
                    
   int                 num_levels;

   zzz_Index          *cindex;
   zzz_Index          *cstride;
   zzz_Index          *findex;
   zzz_Index          *fstride;

   zzz_StructGrid    **grid_l;
                    
   zzz_StructMatrix  **A_l;
   zzz_StructMatrix  **PT_l;
   zzz_StructMatrix  **R_l;
                    
   zzz_StructVector  **b_l;
   zzz_StructVector  **x_l;
   zzz_StructVector  **r_l;

   zzz_SBoxArrayArray *coarse_points_l;
   zzz_SBoxArrayArray *fine_points_l;

   void               *pre_relax_data_initial;
   void              **pre_relax_data_l;
   void               *coarse_relax_data;
   void              **post_relax_data_l;
   void              **residual_data_l;
   void              **restrict_data_l;
   void              **intadd_data_l;

   /* log info (always logged) */
   int                 num_iterations;

   /* additional log info (logged when `logging' > 0) */
   int                 logging;
   int                *norms;
   int                *rel_norms;

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

#define zzz_SMGMapCoarseToFine(index1, index2, coarse_dir) \
{\
   zzz_IndexX(index2) =\
      zzz_IndexX(index1) * zzz_IndexX(cstride) + zzz_IndexX(cindex);\
   zzz_IndexY(index2) =\
      zzz_IndexY(index1) * zzz_IndexY(cstride) + zzz_IndexY(cindex);\
   zzz_IndexZ(index2) =\
      zzz_IndexZ(index1) * zzz_IndexZ(cstride) + zzz_IndexZ(cindex);\
}

#endif
