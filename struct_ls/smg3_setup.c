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
 *
 *****************************************************************************/

#include "headers.h"
#include "smg3.h"

/*--------------------------------------------------------------------------
 * zzz_SMG3Setup
 *--------------------------------------------------------------------------*/

int
zzz_SMG3Setup( zzz_SMG3Data     *smg3_data,
               zzz_StructMatrix *A,
               zzz_StructVector *b,
               zzz_StructVector *x         )
{
   int ierr;

   /*-----------------------------------------------------
    * Setup coarse grids
    *-----------------------------------------------------*/

   SetIndex(cindex, 0, 0, 0);
   SetIndex(findex, 0, 0, 1);
   SetIndex(cstride, 1, 1, 2);
   SetIndex(fstride, 1, 1, 2);

   /* Compute a new max_levels value based on the grid */
   all_boxes = zzz_StructGridAllBoxes(zzz_StructMatrixGrid(A));
   izmin = zzz_BoxIMinZ(zzz_BoxArrayBox(all_boxes, 0));
   izmax = zzz_BoxIMaxZ(zzz_BoxArrayBox(all_boxes, 0));
   zzz_ForBoxI(i, all_boxes)
   {
      izmin = min(izmin, zzz_BoxIMinZ(zzz_BoxArrayBox(all_boxes, i)));
      izmax = max(izmax, zzz_BoxIMaxZ(zzz_BoxArrayBox(all_boxes, i)));
   }
   max_levels = zzz_Log2(izmax - izmin + 2) + 1;
   if (SMG3DataMaxLevels(smg3_data) > 0)
      max_levels = min(max_levels, SMG3DataMaxLevels(smg3_data));
   SMG3DataMaxLevels(smg3_data) = max_levels;

   grid_l = zzz_TAlloc(Grid *, max_levels);
   grid_l[0] = grid;

   l = 0;
   still_coarsening = 1;
   while ( still_coarsening && (l < (max_levels - 1)) )
   {
      /* coarsen the grid */
      coarse_points = zzz_ProjectBoxArray(zzz_StructGridAllBoxes(grid_l[l]),
                                          cindex, cstride);
      all_boxes = zzz_NewBoxArray();
      processes = zzz_TAlloc(int, zzz_SBoxArraySize(coarse_points));
      zzz_ForSBoxI(i, coarse_points)
      {
         sbox = zzz_SBoxArraySBox(coarse_points, i);
         box = zzz_DuplicateBox(zzz_SBoxBox(sbox));
         zzz_SMGMapFineToCoarse(zzz_BoxIMin(box), zzz_BoxIMin(box),
                                cindex, cstride);
         zzz_SMGMapFineToCoarse(zzz_BoxIMax(box), zzz_BoxIMax(box),
                                cindex, cstride);
         zzz_AppendBox(all_boxes, box);
         processes[i] = zzz_StructGridProcess(grid, i);
      }
      grid_l[l+1] =
         zzz_NewAssembledStructGrid(zzz_StructGridComm(grid_l[l]),
                                    zzz_StructGridDim(grid_l[l]),
                                    all_boxes, processes);

      /* do we continue coarsening? */
      izmin = zzz_BoxIMinZ(zzz_BoxArrayBox(all_boxes, 0));
      izmax = zzz_BoxIMaxZ(zzz_BoxArrayBox(all_boxes, 0));
      zzz_ForBoxI(i, all_boxes)
      {
         izmin = min(izmin, zzz_BoxIMinZ(zzz_BoxArrayBox(all_boxes, i)));
         izmax = max(izmax, zzz_BoxIMaxZ(zzz_BoxArrayBox(all_boxes, i)));
      }
      if ( izmin == izmax )
         still_coarsening = 0;
      else
         still_coarsening = 1;

      l++;
   }
   num_levels = l + 1;

   zzz_SMG3DataGridL(smg3_data)         = grid_l;
   zzz_SMG3DataNumLevels(smg3_data)     = num_levels;

   /*-----------------------------------------------------
    * Create coarse_points and fine_points
    *-----------------------------------------------------*/

   coarse_points_l = zzz_NewSBoxArrayArray(num_levels - 1);
   fine_points_l   = zzz_NewSBoxArrayArray(num_levels - 1);

   for (l = 0; l < (num_levels - 1); l++)
   {
      coarse_points = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid_l[l]),
                                          cindex, cstride);
      fine_points   = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid_l[l]),
                                          findex, fstride);
      zzz_FreeSBoxArray(zzz_SBoxArrayArraySBoxArray(coarse_points_l, l));
      zzz_FreeSBoxArray(zzz_SBoxArrayArraySBoxArray(fine_points_l, l));
      zzz_SBoxArrayArraySBoxArray(coarse_points_l, l) = coarse_points;
      zzz_SBoxArrayArraySBoxArray(fine_points_l, l)   = fine_points;
   }

   zzz_SMG3DataCoarsePointsL(smg3_data) = coarse_points_l;
   zzz_SMG3DataFinePointsL(smg3_data)   = fine_points_l;

   /*-----------------------------------------------------
    * Setup matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = zzz_TAlloc(zzz_StructMatrix *, num_levels);
   PT_l = zzz_TAlloc(zzz_StructMatrix *, num_levels - 1);
   x_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   b_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   r_l  = zzz_TAlloc(zzz_StructVector *, num_levels);

   r_l[0] = zzz_NewStructVector(grid_l[0]);
   zzz_SetStructVectorNumGhost(r_l[0], r_num_ghost);
   zzz_InitializeStructVector(r_l[0]);
   zzz_AssembleStructVector(r_l[0]);
   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = zzz_SMG3NewInterpOp(A_l[l], grid_l[l+1]);

      if (zzz_StructMatrixSymmetric(A))
         R_l[l] = PT_l[l];
      else
         R_l[l]   = zzz_SMG3NewRestrictOp(A_l[l], grid_l[l+1]);

      A_l[l+1] = zzz_SMG3NewRAPOp(R_l[l], A_l[l], PT_l[l]);

      x_l[l+1] = zzz_NewStructVector(grid_l[l+1]);
      zzz_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      zzz_InitializeStructVector(x_l[l+1]);
      zzz_AssembleStructVector(x_l[l+1]);

      b_l[l+1] = zzz_NewStructVector(grid_l[l+1]);
      zzz_SetStructVectorNumGhost(b_l[l+1], b_num_ghost);
      zzz_InitializeStructVector(b_l[l+1]);
      zzz_AssembleStructVector(b_l[l+1]);

      r_l[l+1] = zzz_NewStructVector(grid_l[l+1]);
      zzz_SetStructVectorNumGhost(r_l[l+1], r_num_ghost);
      zzz_InitializeStructVectorShell(r_l[l+1]);
      zzz_InitializeStructVectorData(r_l[l+1], zzz_StructVectorData(r_l[0]));
      zzz_AssembleStructVector(r_l[l+1]);
   }

   zzz_SMG3DataAL(smg3_data)  = A_l;
   zzz_SMG3DataPTL(smg3_data) = PT_l;
   zzz_SMG3DataRL(smg3_data)  = R_l;
   zzz_SMG3DataXL(smg3_data)  = x_l;
   zzz_SMG3DataBL(smg3_data)  = b_l;
   zzz_SMG3DataRL(smg3_data)  = r_l;

   /*-----------------------------------------------------
    * Setup coarse grid operators and transfer operators
    *-----------------------------------------------------*/

   zzz_SMG3SetupInterpOp(A_l[l], PT_l[l]);

   if (!zzz_StructMatrixSymmetric(A))
      zzz_SMG3SetupRestrictOp(A_l[l], R_l[l]);

   zzz_SMG3SetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1]);

   /*-----------------------------------------------------
    * Call various setup routines
    *-----------------------------------------------------*/

   relax_data_l    = zzz_TAlloc(SMG3RelaxData *, num_levels);
   residual_data_l = zzz_TAlloc(SMGResidualData *, num_levels);
   restrict_data_l = zzz_TAlloc(SMG3RestrictData *, num_levels);
   intadd_data_l   = zzz_TAlloc(SMG3Intadd *, num_levels);

   for (l = 0; l < (num_levels - 1); l++)
   {
   }
   relax_data_initial;
   relax_data_l;
   residual_data_l;
   restrict_data_l;
   intadd_data_l;

   /*-----------------------------------------------------------------------
    * Initialize data associated with argument `temp_data'
    *-----------------------------------------------------------------------*/

   if ( temp_data != NULL )
   {
      (instance_xtra -> temp_data) = temp_data;

      for (l = 0; l < (instance_xtra -> num_levels); l++)
	 SetTempVectorData((instance_xtra -> temp_vec_l[l]), temp_data);
      temp_data += SizeOfVector(instance_xtra -> temp_vec_l[0]);

      for (l = 1; l < (instance_xtra -> num_levels); l++)
      {
	 SetTempVectorData((instance_xtra -> x_l[l]), temp_data);
	 temp_data += SizeOfVector(instance_xtra -> x_l[l]);
	 SetTempVectorData((instance_xtra -> b_l[l]), temp_data);
	 temp_data += SizeOfVector(instance_xtra -> b_l[l]);
      }

      /* Set up comm_pkg's */
      for (l = 0; l < (instance_xtra -> num_levels) - 1; l++)
      {
	 (instance_xtra -> restrict_comm_pkg_l[l]) =
	    NewVectorCommPkg((instance_xtra -> temp_vec_l[l]),
			     (instance_xtra -> restrict_compute_pkg_l[l]));
	 (instance_xtra -> prolong_comm_pkg_l[l]) =
	    NewVectorCommPkg((instance_xtra -> temp_vec_l[l]),
			     (instance_xtra -> prolong_compute_pkg_l[l]));
      }
   }

   /*-----------------------------------------------------------------------
    * Initialize module instances
    *-----------------------------------------------------------------------*/

   /* if null `grid', pass null `grid_l' to other modules */
   if ( grid == NULL)
      grid_l    = zzz_CTAlloc(Grid *, (instance_xtra -> num_levels));
   else
      grid_l = (instance_xtra -> grid_l);

   /* if null `A', pass null `A_l' to other modules */
   if ( A == NULL)
      A_l = zzz_CTAlloc(Matrix *, (instance_xtra -> num_levels));
   else
      A_l    = (instance_xtra -> A_l);

   if ( PFModuleInstanceXtra(this_module) == NULL )
   {
      (instance_xtra -> smooth_l) =
	 zzz_TAlloc(PFModule *, (instance_xtra -> num_levels));
      for (l = 0; l < ((instance_xtra -> num_levels) - 1); l++)
      {
	 (instance_xtra -> smooth_l[l])  =
	    PFModuleNewInstance((public_xtra -> smooth),
				(problem, grid_l[l], problem_data, A_l[l],
				 temp_data));
      }
      (instance_xtra -> solve)   =
	 PFModuleNewInstance((public_xtra -> solve),
			     (problem, grid_l[l], problem_data, A_l[l],
			      temp_data));
   }
   else
   {
      for (l = 0; l < ((instance_xtra -> num_levels) - 1); l++)
      {
	 PFModuleReNewInstance((instance_xtra -> smooth_l[l]),
			       (problem, grid_l[l], problem_data, A_l[l],
				temp_data));
      }
      PFModuleReNewInstance((instance_xtra -> solve),
			    (problem, grid_l[l], problem_data, A_l[l],
			     temp_data));
   }








   return ierr;
}
