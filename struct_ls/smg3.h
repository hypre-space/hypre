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
 * Header info for the SMG3 solver
 *
 *****************************************************************************/

#ifndef zzz_SMG3_HEADER
#define zzz_SMG3_HEADER


/*--------------------------------------------------------------------------
 * zzz_SMG3Data:
 *--------------------------------------------------------------------------*/

typedef struct
{
   int        max_iter;
   int        max_levels;
   int        min_NX, min_NY, min_NZ;
	     
   int        time_index;

   PFModule  	   **smooth_l;
   PFModule  	    *solve;

   /* InitInstanceXtra arguments */
   double           *temp_data;

   /* instance data */
   int               num_levels;

   int              *coarsen_l;

   Grid            **grid_l;

   SubregionArray  **f_sra_l;
   SubregionArray  **c_sra_l;

   ComputePkg      **restrict_compute_pkg_l;
   ComputePkg      **prolong_compute_pkg_l;

   Matrix          **A_l;
   Matrix          **P_l;

   /* temp data */
   Vector          **x_l;
   Vector          **b_l;
   Vector          **temp_vec_l;

} zzz_SMG3Data;

/*--------------------------------------------------------------------------
 * Accessor macros:
 *--------------------------------------------------------------------------*/

#define zzz_SMG3Data(smg3_data)      ((smg3_data) -> )

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
