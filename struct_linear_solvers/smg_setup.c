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
#include "smg.h"

/*--------------------------------------------------------------------------
 * zzz_SMGSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetup( zzz_SMGData      *smg_data,
              zzz_StructMatrix *A,
              zzz_StructVector *b,
              zzz_StructVector *x        )
{
   int ierr;

   MPI_Comm           *comm;

   int                 max_iter;
   int                 max_levels;
                    
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

   int                 num_iterations;

   int                 logging;
   int                *norms;
   int                *rel_norms;

   /*-----------------------------------------------------
    * Setup coarse grids
    *-----------------------------------------------------*/

   cindex  = (smg_data -> cindex);
   cstride = (smg_data -> cstride);
   findex  = (smg_data -> findex);
   fstride = (smg_data -> fstride);

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
   if (SMGDataMaxLevels(smg_data) > 0)
      max_levels = min(max_levels, SMGDataMaxLevels(smg_data));

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

      l++;
   }
   num_levels = l + 1;

   (smg_data -> max_levels) = max_levels;
   (smg_data -> num_levels) = num_levels;
   (smg_data -> grid_l)     = grid_l;

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

   (smg_data -> coarse_points_l) = coarse_points_l;
   (smg_data -> fine_points_l)   = fine_points_l;

   /*-----------------------------------------------------
    * Setup matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = zzz_TAlloc(zzz_StructMatrix *, num_levels);
   PT_l = zzz_TAlloc(zzz_StructMatrix *, num_levels - 1);
   R_l  = zzz_TAlloc(zzz_StructMatrix *, num_levels - 1);
   x_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   b_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   r_l  = zzz_TAlloc(zzz_StructVector *, num_levels);

   A_l[0] = A;
   x_l[0] = x;
   b_l[0] = b;

   r_l[0] = zzz_NewStructVector(grid_l[0]);
   zzz_SetStructVectorNumGhost(r_l[0], r_num_ghost);
   zzz_InitializeStructVector(r_l[0]);
   zzz_AssembleStructVector(r_l[0]);
   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = zzz_SMGNewInterpOp(A_l[l], grid_l[l+1]);

      if (zzz_StructMatrixSymmetric(A))
         R_l[l] = PT_l[l];
      else
         R_l[l]   = zzz_SMGNewRestrictOp(A_l[l], grid_l[l+1]);

      A_l[l+1] = zzz_SMGNewRAPOp(R_l[l], A_l[l], PT_l[l]);

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

   (smg_data -> A_l)  = A_l;
   (smg_data -> PT_l) = PT_l;
   (smg_data -> R_l)  = R_l;
   (smg_data -> x_l)  = x_l;
   (smg_data -> b_l)  = b_l;
   (smg_data -> r_l)  = r_l;

   /*-----------------------------------------------------
    * Setup coarse grid operators and transfer operators
    *-----------------------------------------------------*/

   zzz_SMGSetupInterpOp(A_l[l], PT_l[l]);

   if (!zzz_StructMatrixSymmetric(A))
      zzz_SMGSetupRestrictOp(A_l[l], R_l[l]);

   zzz_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1]);

   /*-----------------------------------------------------
    * Call various setup routines
    *-----------------------------------------------------*/

   pre_relax_data_l  = zzz_TAlloc(SMGRelaxData *, num_levels);
   post_relax_data_l = zzz_TAlloc(SMGRelaxData *, num_levels);
   residual_data_l   = zzz_TAlloc(SMGResidualData *, num_levels);
   restrict_data_l   = zzz_TAlloc(SMGRestrictData *, num_levels);
   intadd_data_l     = zzz_TAlloc(SMGIntadd *, num_levels);

   pre_relax_data_initial = zzz_SMGRelaxInitialize();
   zzz_SMGRelaxSetTol(pre_relax_data_initial, 0.0);
   zzz_SMGRelaxSetMaxIter(pre_relax_data_initial, 1);
   if (smg_data -> zero_guess)
      zzz_SMGRelaxSetZeroGuess(pre_relax_data_initial);
   zzz_SMGRelaxSetup(pre_relax_data_initial, x_l[l], b_l[l]);

   for (l = 0; l <= (num_levels - 2); l++)
   {
      pre_relax_data_l[l] = zzz_SMGRelaxInitialize();
      zzz_SMGRelaxSetTol(pre_relax_data_l[l], 0.0);
      zzz_SMGRelaxSetMaxIter(pre_relax_data_l[l], 1);
      if (l > 0)
         zzz_SMGRelaxSetZeroGuess(pre_relax_data_l[l]);
      zzz_SMGRelaxSetup(pre_relax_data_l[l], x_l[l], b_l[l]);

      residual_data_l[l] = zzz_SMGResidualInitialize();
      zzz_SMGResidualSetup(residual_data_l[l], A_l[l], x_l[l], b_l[l], r_l[l]);

      restrict_data_l[l] = zzz_SMGRestrictInitialize();
      zzz_SMGRestrictSetup(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1]);
   }

   coarse_relax_data = zzz_SMGRelaxInitialize();
   zzz_SMGRelaxSetTol(coarse_relax_data, 0.0);
   zzz_SMGRelaxSetMaxIter(coarse_relax_data_l[l], 1);
   zzz_SMGRelaxSetZeroGuess(coarse_relax_data_l[l]);
   zzz_SMGRelaxSetup(coarse_relax_data, x_l[l], b_l[l]);

   for (l = (num_levels - 2); l >= 0; l--)
   {
      intadd_data_l[l] = zzz_SMGIntAddInitialize();
      zzz_SMGIntAddSetup(intadd_data_l[l], PT_l[l], x_l[l+1], x_l[l]);

      post_relax_data_l[l] = zzz_SMGRelaxInitialize();
      zzz_SMGRelaxSetTol(post_relax_data_l[l], 0.0);
      zzz_SMGRelaxSetMaxIter(post_relax_data_l[l], 1);
      zzz_SMGRelaxSetup(post_relax_data_l[l], x_l[l], b_l[l]);
   }

   (smg_data -> pre_relax_data_initial) = pre_relax_data_initial;
   (smg_data -> pre_relax_data_l)       = pre_relax_data_l;
   (smg_data -> coarse_relax_data)      = coarse_relax_data;
   (smg_data -> post_relax_data_l)      = post_relax_data_l;
   (smg_data -> residual_data_l)        = residual_data_l;
   (smg_data -> restrict_data_l)        = restrict_data_l;
   (smg_data -> intadd_data_l)          = intadd_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((smg_data -> logging) > 0)
   {
      (smg_data -> norms)     = zzz_TAlloc(double, max_iter);
      (smg_data -> rel_norms) = zzz_TAlloc(double, max_iter);
   }

   return ierr;
}
