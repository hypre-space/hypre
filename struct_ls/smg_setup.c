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
 * zzz_SMGSetPreRelaxParams
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetPreRelaxParams( void *smg_vdata,
                          int   zero_guess, 
                          void *pre_relax_data )
{
   zzz_SMGData *smg_data = smg_vdata;
   int ierr = 0;

   zzz_SMGRelaxSetTol(pre_relax_data, 0.0);
   zzz_SMGRelaxSetMaxIter(pre_relax_data, 1);
   if (zero_guess)
      zzz_SMGRelaxSetZeroGuess(pre_relax_data);

   zzz_SMGRelaxSetNumSpaces(pre_relax_data, 2);
   /* relax coarse points */
   zzz_SMGRelaxSetSpace(pre_relax_data, 0,
                        (smg_data -> ci), (smg_data -> cs));
   /* relax fine points */
   zzz_SMGRelaxSetSpace(pre_relax_data, 1,
                        (smg_data -> fi), (smg_data -> fs));

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetCoarseRelaxParams
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetCoarseRelaxParams( void *smg_vdata,
                             void *coarse_relax_data )
{
   int ierr = 0;

   zzz_SMGRelaxSetTol(coarse_relax_data, 0.0);
   zzz_SMGRelaxSetMaxIter(coarse_relax_data, 1);
   zzz_SMGRelaxSetZeroGuess(coarse_relax_data);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetPostRelaxParams
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetPostRelaxParams( void *smg_vdata,
                           int   zero_guess, 
                           void *post_relax_data )
{
   zzz_SMGData *smg_data = smg_vdata;
   int ierr = 0;

   zzz_SMGRelaxSetTol(post_relax_data, 0.0);
   zzz_SMGRelaxSetMaxIter(post_relax_data, 1);

   zzz_SMGRelaxSetNumSpaces(post_relax_data, 2);
   /* relax fine points */
   zzz_SMGRelaxSetSpace(post_relax_data, 0,
                        (smg_data -> fi), (smg_data -> fs));
   /* relax coarse points */
   zzz_SMGRelaxSetSpace(post_relax_data, 1,
                        (smg_data -> ci), (smg_data -> cs));

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetup( void             *smg_vdata,
              zzz_StructMatrix *A,
              zzz_StructVector *b,
              zzz_StructVector *x        )
{
   zzz_SMGData        *smg_data = smg_vdata;

   MPI_Comm           *comm = (smg_data -> comm);

   int                 max_iter;
   int                 max_levels;
                    
   int                 num_levels;

   int                 cdir;

   zzz_Index         **base_index_l;
   zzz_Index         **base_stride_l;
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
   zzz_StructVector  **r_l;
   zzz_StructVector  **e_l;

   void               *pre_relax_data_initial;
   void              **pre_relax_data_l;
   void               *coarse_relax_data;
   void              **post_relax_data_l;
   void              **residual_data_l;
   void              **restrict_data_l;
   void              **intadd_data_l;

   zzz_BoxArray       *all_boxes;
   zzz_SBoxArray      *coarse_points;
   int                *processes;

   zzz_SBox           *sbox;
   zzz_Box            *box;

   int                 idmin, idmax;
   int                 i, l;

   int                 b_num_ghost[] = {1, 1, 1, 1, 1, 1};
   int                 x_num_ghost[] = {1, 1, 1, 1, 1, 1};
   int                 r_num_ghost[] = {1, 1, 1, 1, 1, 1};

   int                 ierr;

   /*-----------------------------------------------------
    * Compute a new max_levels value based on the grid
    *-----------------------------------------------------*/

   cdir = zzz_StructStencilDim(zzz_StructMatrixStencil(A)) - 1;

   all_boxes = zzz_StructGridAllBoxes(zzz_StructMatrixGrid(A));
   idmin = zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, 0), cdir);
   idmax = zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, 0), cdir);
   zzz_ForBoxI(i, all_boxes)
   {
      idmin = min(idmin, zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, i), cdir));
      idmax = max(idmax, zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, i), cdir));
   }
   max_levels = zzz_Log2(idmax - idmin + 1) + 2;
   if ((smg_data -> max_levels) > 0)
      max_levels = min(max_levels, (smg_data -> max_levels));

   (smg_data -> max_levels) = max_levels;

   /*-----------------------------------------------------
    * Set up base index space and coarsening info:
    *
    * Dealing with the first grid is the trick, since only
    * some of the data there is being used.  All other
    * grids have uniform information, but it is more
    * convenient to refer to this info via an array.
    *-----------------------------------------------------*/

   /* insure at least 2 levels of info */
   i = max(max_levels, 2);
   base_index_l  = zzz_TAlloc(zzz_Index *, i);
   base_stride_l = zzz_TAlloc(zzz_Index *, i);
   cindex_l      = zzz_TAlloc(zzz_Index *, i);
   findex_l      = zzz_TAlloc(zzz_Index *, i);
   cstride_l     = zzz_TAlloc(zzz_Index *, i);
   fstride_l     = zzz_TAlloc(zzz_Index *, i);

   /* initialize info for the finest grid level */
   base_index_l[0]  = (smg_data -> base_index); 
   base_stride_l[0] = (smg_data -> base_stride);
   cindex_l[0]  = zzz_NewIndex();
   findex_l[0]  = zzz_NewIndex();
   cstride_l[0] = zzz_NewIndex();
   fstride_l[0] = zzz_NewIndex();
   zzz_CopyIndex(base_index_l[0], cindex_l[0]);
   zzz_CopyIndex(base_index_l[0], findex_l[0]);
   zzz_CopyIndex(base_stride_l[0], cstride_l[0]);
   zzz_CopyIndex(base_stride_l[0], fstride_l[0]);

   /* initialize info for the 1st coarse grid level */
   base_index_l[1]  = zzz_NewIndex();
   base_stride_l[1] = zzz_NewIndex();
   cindex_l[1]  = zzz_NewIndex();
   findex_l[1]  = zzz_NewIndex();
   cstride_l[1] = zzz_NewIndex();
   fstride_l[1] = zzz_NewIndex();
   zzz_SetIndex(base_index_l[1], 0, 0, 0);
   zzz_SetIndex(base_stride_l[1], 1, 1, 1);
   zzz_SetIndex(cindex_l[1], 0, 0, 0);
   zzz_SetIndex(findex_l[1], 0, 0, 0);
   zzz_SetIndex(cstride_l[1], 1, 1, 1);
   zzz_SetIndex(fstride_l[1], 1, 1, 1);

   /* adjust coarsening info for the 1st two grid levels */
   for (l = 0; l < 2; l++)
   {
      zzz_IndexD(cindex_l[l], cdir)  = (smg_data -> ci);
      zzz_IndexD(findex_l[l], cdir)  = (smg_data -> fi);
      zzz_IndexD(cstride_l[l], cdir) = (smg_data -> cs);
      zzz_IndexD(fstride_l[l], cdir) = (smg_data -> fs);
   }

   /* set coarsening info for the remaining grid levels */
   for (l = 2; l < max_levels; l++)
   {
      base_index_l[l]  = base_index_l[1];
      base_stride_l[l] = base_stride_l[1];
      cindex_l[l]      = cindex_l[1];
      findex_l[l]      = findex_l[1];
      cstride_l[l]     = cstride_l[1];
      fstride_l[l]     = fstride_l[1];
   }

   (smg_data -> cdir) = cdir;
   (smg_data -> base_index_l)  = base_index_l;
   (smg_data -> base_stride_l) = base_stride_l;
   (smg_data -> cindex_l)      = cindex_l;
   (smg_data -> findex_l)      = findex_l;
   (smg_data -> cstride_l)     = cstride_l;
   (smg_data -> fstride_l)     = fstride_l;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid_l = zzz_TAlloc(zzz_StructGrid *, max_levels);
   grid_l[0] = zzz_StructMatrixGrid(A);

   for (l = 0; ; l++)
   {
      /* check to see if we should coarsen */
      idmin = zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, 0), cdir);
      idmax = zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, 0), cdir);
      zzz_ForBoxI(i, all_boxes)
      {
         idmin = min(idmin, zzz_BoxIMinD(zzz_BoxArrayBox(all_boxes, i), cdir));
         idmax = max(idmax, zzz_BoxIMaxD(zzz_BoxArrayBox(all_boxes, i), cdir));
      }
      if ( (idmin == idmax) || (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen the grid */
      coarse_points = zzz_ProjectBoxArray(zzz_StructGridAllBoxes(grid_l[l]),
                                          cindex_l[l], cstride_l[l]);
      all_boxes = zzz_NewBoxArray();
      processes = zzz_TAlloc(int, zzz_SBoxArraySize(coarse_points));
      zzz_ForSBoxI(i, coarse_points)
      {
         sbox = zzz_SBoxArraySBox(coarse_points, i);
         box = zzz_DuplicateBox(zzz_SBoxBox(sbox));
         zzz_SMGMapFineToCoarse(zzz_BoxIMin(box), zzz_BoxIMin(box),
                                cindex_l[l], cstride_l[l]);
         zzz_SMGMapFineToCoarse(zzz_BoxIMax(box), zzz_BoxIMax(box),
                                cindex_l[l], cstride_l[l]);
         zzz_AppendBox(box, all_boxes);
         processes[i] = zzz_StructGridProcess(grid_l[l], i);
      }
      grid_l[l+1] =
         zzz_NewAssembledStructGrid(comm, zzz_StructGridDim(grid_l[l]),
                                    all_boxes, processes);
      zzz_FreeSBoxArray(coarse_points);
   }
   num_levels = l + 1;

   (smg_data -> num_levels) = num_levels;
   (smg_data -> grid_l)     = grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = zzz_TAlloc(zzz_StructMatrix *, num_levels);
   PT_l = zzz_TAlloc(zzz_StructMatrix *, num_levels - 1);
   R_l  = zzz_TAlloc(zzz_StructMatrix *, num_levels - 1);
   x_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   b_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   r_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   e_l  = zzz_TAlloc(zzz_StructVector *, num_levels);

   A_l[0] = A;
   x_l[0] = x;
   b_l[0] = b;

   r_l[0] = zzz_NewStructVector(comm, grid_l[0]);
   zzz_SetStructVectorNumGhost(r_l[0], r_num_ghost);
   zzz_InitializeStructVector(r_l[0]);
   zzz_AssembleStructVector(r_l[0]);
   e_l[0] = r_l[0];
   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = zzz_SMGNewInterpOp(A_l[l], grid_l[l+1], cdir);

      if (zzz_StructMatrixSymmetric(A))
         R_l[l] = PT_l[l];
      else
         R_l[l]   = zzz_SMGNewRestrictOp(A_l[l], grid_l[l+1], cdir);

      A_l[l+1] = zzz_SMGNewRAPOp(R_l[l], A_l[l], PT_l[l]);

      x_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      zzz_InitializeStructVector(x_l[l+1]);
      zzz_AssembleStructVector(x_l[l+1]);

      b_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(b_l[l+1], b_num_ghost);
      zzz_InitializeStructVector(b_l[l+1]);
      zzz_AssembleStructVector(b_l[l+1]);

      r_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(r_l[l+1], r_num_ghost);
      zzz_InitializeStructVectorShell(r_l[l+1]);
      zzz_InitializeStructVectorData(r_l[l+1], zzz_StructVectorData(r_l[0]));
      zzz_AssembleStructVector(r_l[l+1]);
      e_l[l+1] = r_l[l+1];
   }

   (smg_data -> A_l)  = A_l;
   (smg_data -> PT_l) = PT_l;
   (smg_data -> R_l)  = R_l;
   (smg_data -> x_l)  = x_l;
   (smg_data -> b_l)  = b_l;
   (smg_data -> r_l)  = r_l;
   (smg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up coarse grid operators and transfer operators
    *-----------------------------------------------------*/

   for (l = 0; l < (num_levels - 1); l++)
   {
      zzz_SMGSetupInterpOp(A_l[l], PT_l[l], r_l[l], cdir,
                           cindex_l[l], cstride_l[l],
                           findex_l[l], fstride_l[l]);

      if (!zzz_StructMatrixSymmetric(A))
         zzz_SMGSetupRestrictOp(A_l[l], R_l[l], r_l[l], cdir,
                                cindex_l[l], cstride_l[l]);

      zzz_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1],
                        cindex_l[l], cstride_l[l]);
   }

   /*-----------------------------------------------------
    * Call various setup routines
    *-----------------------------------------------------*/

   pre_relax_data_l  = zzz_TAlloc(void *, num_levels);
   post_relax_data_l = zzz_TAlloc(void *, num_levels);
   residual_data_l   = zzz_TAlloc(void *, num_levels);
   restrict_data_l   = zzz_TAlloc(void *, num_levels);
   intadd_data_l     = zzz_TAlloc(void *, num_levels);

   pre_relax_data_initial = zzz_SMGRelaxInitialize(comm);
   zzz_SMGRelaxSetBase(pre_relax_data_initial,
                       base_index_l[0], base_stride_l[0]);
   zzz_SMGSetPreRelaxParams(smg_data, (smg_data -> zero_guess),
                            pre_relax_data_initial);
   zzz_SMGRelaxSetup(pre_relax_data_initial, A_l[0], b_l[0], x_l[0], r_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      pre_relax_data_l[l] = zzz_SMGRelaxInitialize(comm);
      if (l > 0)
      {
         zzz_SMGSetPreRelaxParams(smg_data, 1, pre_relax_data_l[l]);
      }
      else
      {
         zzz_SMGRelaxSetBase(pre_relax_data_l[l],
                             base_index_l[l], base_stride_l[l]);
         zzz_SMGSetPreRelaxParams(smg_data, 0, pre_relax_data_l[l]);
      }
      zzz_SMGRelaxSetup(pre_relax_data_l[l], A_l[l], b_l[l], x_l[l], r_l[l]);

      residual_data_l[l] = zzz_SMGResidualInitialize();
      zzz_SMGResidualSetBase(residual_data_l[l],
                             base_index_l[l], base_stride_l[l]);
      zzz_SMGResidualSetup(residual_data_l[l], A_l[l], x_l[l], b_l[l], r_l[l]);

      restrict_data_l[l] = zzz_SMGRestrictInitialize();
      zzz_SMGRestrictSetup(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1],
                           cindex_l[l], cstride_l[l],
                           findex_l[l], fstride_l[l]);
   }

   coarse_relax_data = zzz_SMGRelaxInitialize(comm);
   zzz_SMGRelaxSetBase(coarse_relax_data,
                       base_index_l[l], base_stride_l[l]);
   zzz_SMGSetCoarseRelaxParams(smg_data, coarse_relax_data);
   zzz_SMGRelaxSetup(coarse_relax_data, A_l[l], b_l[l], x_l[l], r_l[l]);

   for (l = (num_levels - 2); l >= 0; l--)
   {
      intadd_data_l[l] = zzz_SMGIntAddInitialize();
      zzz_SMGIntAddSetup(intadd_data_l[l], PT_l[l], x_l[l+1], e_l[l], x_l[l],
                         cindex_l[l], cstride_l[l],
                         findex_l[l], fstride_l[l]);

      post_relax_data_l[l] = zzz_SMGRelaxInitialize(comm);
      zzz_SMGRelaxSetBase(post_relax_data_l[l],
                          base_index_l[l], base_stride_l[l]);
      zzz_SMGSetPostRelaxParams(smg_data, 0, post_relax_data_l[l]);
      zzz_SMGRelaxSetup(post_relax_data_l[l], A_l[l], b_l[l], x_l[l], r_l[l]);
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

#if 0
{
   char  filename[255];

   /* debugging stuff */
   for (l = 0; l < (num_levels - 1); l++)
   {
      sprintf(filename, "zout_A.%02d", l);
      zzz_PrintStructMatrix(filename, A_l[l], 0);
      sprintf(filename, "zout_PT.%02d", l);
      zzz_PrintStructMatrix(filename, PT_l[l], 0);
   }
   sprintf(filename, "zout_A.%02d", l);
   zzz_PrintStructMatrix(filename, A_l[l], 0);
}
#endif

   return ierr;
}

