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
 * hypre_SMGSetup
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetup( void             *smg_vdata,
              hypre_StructMatrix *A,
              hypre_StructVector *b,
              hypre_StructVector *x        )
{
   hypre_SMGData        *smg_data = smg_vdata;

   MPI_Comm           *comm = (smg_data -> comm);
   int                 ci   = (smg_data -> ci);
   int                 fi   = (smg_data -> fi);
   int                 cs   = (smg_data -> cs);
   int                 fs   = (smg_data -> fs);

   int                 n_pre   = (smg_data -> num_pre_relax);
   int                 n_post  = (smg_data -> num_post_relax);

   int                 max_iter;
   int                 max_levels;
                    
   int                 num_levels;

   int                 cdir;

   hypre_Index          *base_index_l;
   hypre_Index          *base_stride_l;
   hypre_Index          *cindex_l;
   hypre_Index          *findex_l;
   hypre_Index          *cstride_l;
   hypre_Index          *fstride_l;

   hypre_StructGrid    **grid_l;
                    
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
   double             *b_data;
   double             *x_data;

   void              **relax_data_l;
   void              **residual_data_l;
   void              **restrict_data_l;
   void              **intadd_data_l;

   hypre_BoxArray       *all_boxes;
   hypre_SBoxArray      *coarse_points;
   int                *processes;

   hypre_SBox           *sbox;
   hypre_Box            *box;

   int                 idmin, idmax;
   int                 i, l;

   int                 b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   int                 x_num_ghost[]  = {0, 0, 0, 0, 0, 0};

   int                 ierr;

   /*-----------------------------------------------------
    * Compute a new max_levels value based on the grid
    *-----------------------------------------------------*/

   cdir = hypre_StructStencilDim(hypre_StructMatrixStencil(A)) - 1;

   all_boxes = hypre_StructGridAllBoxes(hypre_StructMatrixGrid(A));
   idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), cdir);
   idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), cdir);
   hypre_ForBoxI(i, all_boxes)
   {
      idmin = min(idmin, hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), cdir));
      idmax = max(idmax, hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), cdir));
   }
   max_levels = hypre_Log2(idmax - idmin + 1) + 2;
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
   base_index_l  = hypre_TAlloc(hypre_Index, i);
   base_stride_l = hypre_TAlloc(hypre_Index, i);
   cindex_l      = hypre_TAlloc(hypre_Index, i);
   findex_l      = hypre_TAlloc(hypre_Index, i);
   cstride_l     = hypre_TAlloc(hypre_Index, i);
   fstride_l     = hypre_TAlloc(hypre_Index, i);

   /* initialize info for the finest grid level */
   hypre_CopyIndex((smg_data -> base_index), base_index_l[0]);
   hypre_CopyIndex((smg_data -> base_stride), base_stride_l[0]);
   hypre_CopyIndex((smg_data -> base_index), cindex_l[0]);
   hypre_CopyIndex((smg_data -> base_index), findex_l[0]);
   hypre_CopyIndex((smg_data -> base_stride), cstride_l[0]);
   hypre_CopyIndex((smg_data -> base_stride), fstride_l[0]);

   /* initialize info for the 1st coarse grid level */
   hypre_SetIndex(base_index_l[1], 0, 0, 0);
   hypre_SetIndex(base_stride_l[1], 1, 1, 1);
   hypre_SetIndex(cindex_l[1], 0, 0, 0);
   hypre_SetIndex(findex_l[1], 0, 0, 0);
   hypre_SetIndex(cstride_l[1], 1, 1, 1);
   hypre_SetIndex(fstride_l[1], 1, 1, 1);

   /* adjust coarsening info for the 1st two grid levels */
   for (l = 0; l < 2; l++)
   {
      hypre_IndexD(cindex_l[l], cdir)  = ci;
      hypre_IndexD(findex_l[l], cdir)  = fi;
      hypre_IndexD(cstride_l[l], cdir) = cs;
      hypre_IndexD(fstride_l[l], cdir) = fs;
   }

   /* set coarsening info for the remaining grid levels */
   for (l = 2; l < max_levels; l++)
   {
      hypre_CopyIndex(base_index_l[1], base_index_l[l]);
      hypre_CopyIndex(base_stride_l[1], base_stride_l[l]);
      hypre_CopyIndex(cindex_l[1], cindex_l[l]);
      hypre_CopyIndex(findex_l[1], findex_l[l]);
      hypre_CopyIndex(cstride_l[1], cstride_l[l]);
      hypre_CopyIndex(fstride_l[1], fstride_l[l]);
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

   grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
   grid_l[0] = hypre_StructMatrixGrid(A);

   for (l = 0; ; l++)
   {
      /* check to see if we should coarsen */
      idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), cdir);
      idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), cdir);
      hypre_ForBoxI(i, all_boxes)
      {
         idmin = min(idmin, hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), cdir));
         idmax = max(idmax, hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), cdir));
      }
      if ( (idmin == idmax) || (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen the grid */
      coarse_points = hypre_ProjectBoxArray(hypre_StructGridAllBoxes(grid_l[l]),
                                          cindex_l[l], cstride_l[l]);
      all_boxes = hypre_NewBoxArray();
      processes = hypre_TAlloc(int, hypre_SBoxArraySize(coarse_points));
      hypre_ForSBoxI(i, coarse_points)
      {
         sbox = hypre_SBoxArraySBox(coarse_points, i);
         box = hypre_DuplicateBox(hypre_SBoxBox(sbox));
         hypre_SMGMapFineToCoarse(hypre_BoxIMin(box), hypre_BoxIMin(box),
                                cindex_l[l], cstride_l[l]);
         hypre_SMGMapFineToCoarse(hypre_BoxIMax(box), hypre_BoxIMax(box),
                                cindex_l[l], cstride_l[l]);
         hypre_AppendBox(box, all_boxes);
         processes[i] = hypre_StructGridProcess(grid_l[l], i);
      }
      grid_l[l+1] =
         hypre_NewAssembledStructGrid(comm, hypre_StructGridDim(grid_l[l]),
                                    all_boxes, processes);
      hypre_FreeSBoxArray(coarse_points);
   }
   num_levels = l + 1;

   (smg_data -> num_levels) = num_levels;
   (smg_data -> grid_l)     = grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels);
   PT_l = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
   R_l  = hypre_TAlloc(hypre_StructMatrix *, num_levels - 1);
   b_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
   x_l  = hypre_TAlloc(hypre_StructVector *, num_levels);
   tb_l = hypre_TAlloc(hypre_StructVector *, num_levels);
   tx_l = hypre_TAlloc(hypre_StructVector *, num_levels);
   r_l  = tx_l;
   e_l  = tx_l;

   A_l[0] = A;
   b_l[0] = b;
   x_l[0] = x;

   for (i = 0; i <= cdir; i++)
   {
      x_num_ghost[2*i]     = 1;
      x_num_ghost[2*i + 1] = 1;
   }

   tb_l[0] = hypre_NewStructVector(comm, grid_l[0]);
   hypre_SetStructVectorNumGhost(tb_l[0], hypre_StructVectorNumGhost(b));
   hypre_InitializeStructVector(tb_l[0]);
   hypre_AssembleStructVector(tb_l[0]);

   tx_l[0] = hypre_NewStructVector(comm, grid_l[0]);
   hypre_SetStructVectorNumGhost(tx_l[0], hypre_StructVectorNumGhost(x));
   hypre_InitializeStructVector(tx_l[0]);
   hypre_AssembleStructVector(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = hypre_SMGNewInterpOp(A_l[l], grid_l[l+1], cdir);

      if (hypre_StructMatrixSymmetric(A))
         R_l[l] = PT_l[l];
      else
         R_l[l]   = hypre_SMGNewRestrictOp(A_l[l], grid_l[l+1], cdir);

      A_l[l+1] = hypre_SMGNewRAPOp(R_l[l], A_l[l], PT_l[l]);

      b_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(b_l[l+1], b_num_ghost);
      hypre_InitializeStructVector(b_l[l+1]);
      hypre_AssembleStructVector(b_l[l+1]);

      x_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      hypre_InitializeStructVector(x_l[l+1]);
      hypre_AssembleStructVector(x_l[l+1]);

      tb_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(tb_l[l+1], hypre_StructVectorNumGhost(b));
      hypre_InitializeStructVectorShell(tb_l[l+1]);
      hypre_InitializeStructVectorData(tb_l[l+1], hypre_StructVectorData(tb_l[0]));
      hypre_AssembleStructVector(tb_l[l+1]);

      tx_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(tx_l[l+1], hypre_StructVectorNumGhost(x));
      hypre_InitializeStructVectorShell(tx_l[l+1]);
      hypre_InitializeStructVectorData(tx_l[l+1], hypre_StructVectorData(tx_l[0]));
      hypre_AssembleStructVector(tx_l[l+1]);
   }

   (smg_data -> A_l)  = A_l;
   (smg_data -> PT_l) = PT_l;
   (smg_data -> R_l)  = R_l;
   (smg_data -> b_l)  = b_l;
   (smg_data -> x_l)  = x_l;
   (smg_data -> tb_l) = tb_l;
   (smg_data -> tx_l) = tx_l;
   (smg_data -> r_l)  = r_l;
   (smg_data -> e_l)  = e_l;

   /*-----------------------------------------------------
    * Set up multigrid operators and call setup routines
    *
    * Note: The routine that sets up interpolation uses
    * the same relaxation routines used in the solve
    * phase of the algorithm.  To do this, the data for
    * the fine-grid unknown and right-hand-side vectors
    * is temporarily changed to temporary data.
    *-----------------------------------------------------*/

   relax_data_l    = hypre_TAlloc(void *, num_levels);
   residual_data_l = hypre_TAlloc(void *, num_levels);
   restrict_data_l = hypre_TAlloc(void *, num_levels);
   intadd_data_l   = hypre_TAlloc(void *, num_levels);

   /* temporarily set the data for x_l[0] and b_l[0] to temp data */
   b_data = hypre_StructVectorData(b_l[0]);
   x_data = hypre_StructVectorData(x_l[0]);
   hypre_InitializeStructVectorData(b_l[0], hypre_StructVectorData(tb_l[0]));
   hypre_InitializeStructVectorData(x_l[0], hypre_StructVectorData(tx_l[0]));
   hypre_AssembleStructVector(b_l[0]);
   hypre_AssembleStructVector(x_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      /* set up relaxation */
      relax_data_l[l] = hypre_SMGRelaxInitialize(comm);
      hypre_SMGRelaxSetBase(relax_data_l[l], base_index_l[l], base_stride_l[l]);
      hypre_SMGRelaxSetMemoryUse(relax_data_l[l], (smg_data -> memory_use));
      hypre_SMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SMGRelaxSetNumSpaces(relax_data_l[l], 2);
      hypre_SMGRelaxSetSpace(relax_data_l[l], 0, ci, cs);
      hypre_SMGRelaxSetSpace(relax_data_l[l], 1, fi, fs);
      hypre_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
      hypre_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
      hypre_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
      hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      hypre_SMGSetupInterpOp(relax_data_l[l], A_l[l], b_l[l], x_l[l],
                           PT_l[l], cdir,
                           cindex_l[l], cstride_l[l],
                           findex_l[l], fstride_l[l]);

      /* (re)set relaxation parameters */
      hypre_SMGRelaxSetNumPreSpaces(relax_data_l[l], 0);
      hypre_SMGRelaxSetNumRegSpaces(relax_data_l[l], 2);
      hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      /* set up the residual routine */
      residual_data_l[l] = hypre_SMGResidualInitialize();
      hypre_SMGResidualSetBase(residual_data_l[l],
                             base_index_l[l], base_stride_l[l]);
      hypre_SMGResidualSetup(residual_data_l[l], A_l[l], x_l[l], b_l[l], r_l[l]);

      /* set up the interpolation routine */
      intadd_data_l[l] = hypre_SMGIntAddInitialize();
      hypre_SMGIntAddSetup(intadd_data_l[l], PT_l[l], x_l[l+1], e_l[l], x_l[l],
                         cindex_l[l], cstride_l[l],
                         findex_l[l], fstride_l[l]);

      /* set up the restriction operator */
      if (!hypre_StructMatrixSymmetric(A))
         hypre_SMGSetupRestrictOp(A_l[l], R_l[l], tx_l[l], cdir,
                                cindex_l[l], cstride_l[l]);

      /* set up the restriction routine */
      restrict_data_l[l] = hypre_SMGRestrictInitialize();
      hypre_SMGRestrictSetup(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1],
                           cindex_l[l], cstride_l[l],
                           findex_l[l], fstride_l[l]);

      /* set up the coarse grid operator */
      hypre_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1],
                        cindex_l[l], cstride_l[l]);
   }

   relax_data_l[l] = hypre_SMGRelaxInitialize(comm);
   hypre_SMGRelaxSetBase(relax_data_l[l], base_index_l[l], base_stride_l[l]);
   hypre_SMGRelaxSetTol(relax_data_l[l], 0.0);
   hypre_SMGRelaxSetMaxIter(relax_data_l[l], 1);
   hypre_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
   hypre_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
   hypre_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
   hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

   /* set the data for x_l[0] and b_l[0] the way they were */
   hypre_InitializeStructVectorData(b_l[0], b_data);
   hypre_InitializeStructVectorData(x_l[0], x_data);
   hypre_AssembleStructVector(b_l[0]);
   hypre_AssembleStructVector(x_l[0]);

   (smg_data -> relax_data_l)      = relax_data_l;
   (smg_data -> residual_data_l)   = residual_data_l;
   (smg_data -> restrict_data_l)   = restrict_data_l;
   (smg_data -> intadd_data_l)     = intadd_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((smg_data -> logging) > 0)
   {
      max_iter = (smg_data -> max_iter);
      (smg_data -> norms)     = hypre_TAlloc(double, max_iter);
      (smg_data -> rel_norms) = hypre_TAlloc(double, max_iter);
   }

#if 0
{
   if(hypre_StructGridDim(grid_l[0]) == 3)
   {
   char  filename[255];

   /* debugging stuff */
   for (l = 0; l < (num_levels - 1); l++)
   {
      sprintf(filename, "zout_A.%02d", l);
      hypre_PrintStructMatrix(filename, A_l[l], 0);
      sprintf(filename, "zout_PT.%02d", l);
      hypre_PrintStructMatrix(filename, PT_l[l], 0);
   }
   sprintf(filename, "zout_A.%02d", l);
   hypre_PrintStructMatrix(filename, A_l[l], 0);
   }
}
#endif

   return ierr;
}

