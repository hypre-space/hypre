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
zzz_SMGSetup( void             *smg_vdata,
              zzz_StructMatrix *A,
              zzz_StructVector *b,
              zzz_StructVector *x        )
{
   zzz_SMGData        *smg_data = smg_vdata;

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

   /* temp vectors */
   zzz_StructVector  **tb_l;
   zzz_StructVector  **tx_l;
   zzz_StructVector  **r_l;
   zzz_StructVector  **e_l;
   double             *b_data;
   double             *x_data;

   void              **relax_data_l;
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

   int                 b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   int                 x_num_ghost[]  = {0, 0, 0, 0, 0, 0};

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
      zzz_IndexD(cindex_l[l], cdir)  = ci;
      zzz_IndexD(findex_l[l], cdir)  = fi;
      zzz_IndexD(cstride_l[l], cdir) = cs;
      zzz_IndexD(fstride_l[l], cdir) = fs;
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
   b_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   x_l  = zzz_TAlloc(zzz_StructVector *, num_levels);
   tb_l = zzz_TAlloc(zzz_StructVector *, num_levels);
   tx_l = zzz_TAlloc(zzz_StructVector *, num_levels);
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

   tb_l[0] = zzz_NewStructVector(comm, grid_l[0]);
   zzz_SetStructVectorNumGhost(tb_l[0], zzz_StructVectorNumGhost(b));
   zzz_InitializeStructVector(tb_l[0]);
   zzz_AssembleStructVector(tb_l[0]);

   tx_l[0] = zzz_NewStructVector(comm, grid_l[0]);
   zzz_SetStructVectorNumGhost(tx_l[0], zzz_StructVectorNumGhost(x));
   zzz_InitializeStructVector(tx_l[0]);
   zzz_AssembleStructVector(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = zzz_SMGNewInterpOp(A_l[l], grid_l[l+1], cdir);

      if (zzz_StructMatrixSymmetric(A))
         R_l[l] = PT_l[l];
      else
         R_l[l]   = zzz_SMGNewRestrictOp(A_l[l], grid_l[l+1], cdir);

      A_l[l+1] = zzz_SMGNewRAPOp(R_l[l], A_l[l], PT_l[l]);

      b_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(b_l[l+1], b_num_ghost);
      zzz_InitializeStructVector(b_l[l+1]);
      zzz_AssembleStructVector(b_l[l+1]);

      x_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      zzz_InitializeStructVector(x_l[l+1]);
      zzz_AssembleStructVector(x_l[l+1]);

      tb_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(tb_l[l+1], zzz_StructVectorNumGhost(b));
      zzz_InitializeStructVectorShell(tb_l[l+1]);
      zzz_InitializeStructVectorData(tb_l[l+1], zzz_StructVectorData(tb_l[0]));
      zzz_AssembleStructVector(tb_l[l+1]);

      tx_l[l+1] = zzz_NewStructVector(comm, grid_l[l+1]);
      zzz_SetStructVectorNumGhost(tx_l[l+1], zzz_StructVectorNumGhost(x));
      zzz_InitializeStructVectorShell(tx_l[l+1]);
      zzz_InitializeStructVectorData(tx_l[l+1], zzz_StructVectorData(tx_l[0]));
      zzz_AssembleStructVector(tx_l[l+1]);
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

   relax_data_l    = zzz_TAlloc(void *, num_levels);
   residual_data_l = zzz_TAlloc(void *, num_levels);
   restrict_data_l = zzz_TAlloc(void *, num_levels);
   intadd_data_l   = zzz_TAlloc(void *, num_levels);

   /* temporarily set the data for x_l[0] and b_l[0] to temp data */
   b_data = zzz_StructVectorData(b_l[0]);
   x_data = zzz_StructVectorData(x_l[0]);
   zzz_InitializeStructVectorData(b_l[0], zzz_StructVectorData(tb_l[0]));
   zzz_InitializeStructVectorData(x_l[0], zzz_StructVectorData(tx_l[0]));
   zzz_AssembleStructVector(b_l[0]);
   zzz_AssembleStructVector(x_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      /* set up relaxation */
      relax_data_l[l] = zzz_SMGRelaxInitialize(comm);
      zzz_SMGRelaxSetBase(relax_data_l[l], base_index_l[l], base_stride_l[l]);
      zzz_SMGRelaxSetMemoryUse(relax_data_l[l], (smg_data -> memory_use));
      zzz_SMGRelaxSetTol(relax_data_l[l], 0.0);
      zzz_SMGRelaxSetNumSpaces(relax_data_l[l], 2);
      zzz_SMGRelaxSetSpace(relax_data_l[l], 0, ci, cs);
      zzz_SMGRelaxSetSpace(relax_data_l[l], 1, fi, fs);
      zzz_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
      zzz_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
      zzz_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
      zzz_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      zzz_SMGSetupInterpOp(relax_data_l[l], A_l[l], b_l[l], x_l[l],
                           PT_l[l], cdir,
                           cindex_l[l], cstride_l[l],
                           findex_l[l], fstride_l[l]);

      /* (re)set relaxation parameters */
      zzz_SMGRelaxSetNumPreSpaces(relax_data_l[l], 0);
      zzz_SMGRelaxSetNumRegSpaces(relax_data_l[l], 2);
      zzz_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      /* set up the residual routine */
      residual_data_l[l] = zzz_SMGResidualInitialize();
      zzz_SMGResidualSetBase(residual_data_l[l],
                             base_index_l[l], base_stride_l[l]);
      zzz_SMGResidualSetup(residual_data_l[l], A_l[l], x_l[l], b_l[l], r_l[l]);

      /* set up the interpolation routine */
      intadd_data_l[l] = zzz_SMGIntAddInitialize();
      zzz_SMGIntAddSetup(intadd_data_l[l], PT_l[l], x_l[l+1], e_l[l], x_l[l],
                         cindex_l[l], cstride_l[l],
                         findex_l[l], fstride_l[l]);

      /* set up the restriction operator */
      if (!zzz_StructMatrixSymmetric(A))
         zzz_SMGSetupRestrictOp(A_l[l], R_l[l], tx_l[l], cdir,
                                cindex_l[l], cstride_l[l]);

      /* set up the restriction routine */
      restrict_data_l[l] = zzz_SMGRestrictInitialize();
      zzz_SMGRestrictSetup(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1],
                           cindex_l[l], cstride_l[l],
                           findex_l[l], fstride_l[l]);

      /* set up the coarse grid operator */
      zzz_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1],
                        cindex_l[l], cstride_l[l]);
   }

   relax_data_l[l] = zzz_SMGRelaxInitialize(comm);
   zzz_SMGRelaxSetBase(relax_data_l[l], base_index_l[l], base_stride_l[l]);
   zzz_SMGRelaxSetTol(relax_data_l[l], 0.0);
   zzz_SMGRelaxSetMaxIter(relax_data_l[l], 1);
   zzz_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
   zzz_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
   zzz_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
   zzz_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

   /* set the data for x_l[0] and b_l[0] the way they were */
   zzz_InitializeStructVectorData(b_l[0], b_data);
   zzz_InitializeStructVectorData(x_l[0], x_data);
   zzz_AssembleStructVector(b_l[0]);
   zzz_AssembleStructVector(x_l[0]);

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
      (smg_data -> norms)     = zzz_TAlloc(double, max_iter);
      (smg_data -> rel_norms) = zzz_TAlloc(double, max_iter);
   }

#if 0
{
   if(zzz_StructGridDim(grid_l[0]) == 3)
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
}
#endif

   return ierr;
}

