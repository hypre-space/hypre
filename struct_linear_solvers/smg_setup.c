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
hypre_SMGSetup( void               *smg_vdata,
                hypre_StructMatrix *A,
                hypre_StructVector *b,
                hypre_StructVector *x        )
{
   hypre_SMGData        *smg_data = smg_vdata;

   MPI_Comm              comm = (smg_data -> comm);
   hypre_IndexRef        base_index  = (smg_data -> base_index);
   hypre_IndexRef        base_stride = (smg_data -> base_stride);
                     
   int                   n_pre   = (smg_data -> num_pre_relax);
   int                   n_post  = (smg_data -> num_post_relax);
                     
   int                   max_iter;
   int                   max_levels;
                      
   int                   num_levels;
                     
   int                   cdir;

   hypre_Index           bindex;
   hypre_Index           bstride;
   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   hypre_StructGrid    **grid_l;
                    
   double               *data;
   int                   data_size = 0;
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
   double               *b_data;
   double               *x_data;

   void                **relax_data_l;
   void                **residual_data_l;
   void                **restrict_data_l;
   void                **intadd_data_l;

   hypre_BoxArray       *boxes;
   hypre_BoxArray       *all_boxes;
   int                  *processes;
   hypre_SBoxArray      *coarse_points;
   int                  *box_ranks;
   int                   num_boxes;
   int                   num_all_boxes;

   hypre_SBox           *sbox;
   hypre_Box            *box;

   int                   idmin, idmax;
   int                   i, l;
                       
   int                   b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   int                   x_num_ghost[]  = {0, 0, 0, 0, 0, 0};
                       
   int                   ierr = 0;

   /*-----------------------------------------------------
    * Set up coarsening direction
    *-----------------------------------------------------*/

   cdir = hypre_StructStencilDim(hypre_StructMatrixStencil(A)) - 1;
   (smg_data -> cdir) = cdir;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_GatherAllBoxes(comm, boxes, &all_boxes, &processes, &box_ranks);
   num_boxes = hypre_BoxArraySize(boxes);
   num_all_boxes = hypre_BoxArraySize(all_boxes);

   /* Compute a new max_levels value based on the grid */
   idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), cdir);
   idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), cdir);
   for (i = 0; i < num_all_boxes; i++)
   {
      idmin =
         min(idmin, hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), cdir));
      idmax =
         max(idmax, hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), cdir));
   }
   max_levels = hypre_Log2(idmax - idmin + 1) + 2;
   if ((smg_data -> max_levels) > 0)
      max_levels = min(max_levels, (smg_data -> max_levels));
   (smg_data -> max_levels) = max_levels;

   grid_l = hypre_TAlloc(hypre_StructGrid *, max_levels);
   grid_l[0] = hypre_StructMatrixGrid(A);
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      hypre_SMGSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_SMGSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      idmin = hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, 0), cdir);
      idmax = hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, 0), cdir);
      for (i = 0; i < num_all_boxes; i++)
      {
         idmin =
            min(idmin, hypre_BoxIMinD(hypre_BoxArrayBox(all_boxes, i), cdir));
         idmax =
            max(idmax, hypre_BoxIMaxD(hypre_BoxArrayBox(all_boxes, i), cdir));
      }
      if ( (idmin == idmax) || (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen the grid by coarsening all_boxes (reduces communication) */
      coarse_points =
         hypre_ProjectBoxArray(all_boxes, cindex, stride);
      hypre_FreeBoxArray(all_boxes);
      all_boxes = hypre_NewBoxArray(num_all_boxes);
      for (i = 0; i < num_all_boxes; i++)
      {
         sbox = hypre_SBoxArraySBox(coarse_points, i);
         box = hypre_DuplicateBox(hypre_SBoxBox(sbox));
         hypre_SMGMapFineToCoarse(hypre_BoxIMin(box), hypre_BoxIMin(box),
                                  cindex, stride);
         hypre_SMGMapFineToCoarse(hypre_BoxIMax(box), hypre_BoxIMax(box),
                                  cindex, stride);
         hypre_AppendBox(box, all_boxes);
      }
      hypre_FreeSBoxArray(coarse_points);

      /* compute local boxes */
      boxes = hypre_NewBoxArray(num_boxes);
      for (i = 0; i < num_boxes; i++)
      {
         box = hypre_DuplicateBox(hypre_BoxArrayBox(all_boxes, box_ranks[i]));
         hypre_AppendBox(box, boxes);
      }

      grid_l[l+1] = hypre_NewStructGrid(comm, hypre_StructGridDim(grid_l[l]));
      hypre_SetStructGridBoxes(grid_l[l+1], boxes);
      hypre_AssembleStructGrid(grid_l[l+1], all_boxes, processes, box_ranks);
   }
   num_levels = l + 1;

   hypre_FreeBoxArray(all_boxes);
   hypre_TFree(processes);
   hypre_TFree(box_ranks);

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
   hypre_InitializeStructVectorShell(tb_l[0]);
   data_size += hypre_StructVectorDataSize(tb_l[0]);

   tx_l[0] = hypre_NewStructVector(comm, grid_l[0]);
   hypre_SetStructVectorNumGhost(tx_l[0], hypre_StructVectorNumGhost(x));
   hypre_InitializeStructVectorShell(tx_l[0]);
   data_size += hypre_StructVectorDataSize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = hypre_SMGNewInterpOp(A_l[l], grid_l[l+1], cdir);
      data_size += hypre_StructMatrixDataSize(PT_l[l]);

      if (hypre_StructMatrixSymmetric(A))
      {
         R_l[l] = PT_l[l];
      }
      else
      {
         R_l[l] = PT_l[l];
#if 0
         /* Allow R != PT for non symmetric case */
         R_l[l]   = hypre_SMGNewRestrictOp(A_l[l], grid_l[l+1], cdir);
         data_size += hypre_StructMatrixDataSize(R_l[l]);
#endif
      }

      A_l[l+1] = hypre_SMGNewRAPOp(R_l[l], A_l[l], PT_l[l]);
      data_size += hypre_StructMatrixDataSize(A_l[l+1]);

      b_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(b_l[l+1], b_num_ghost);
      hypre_InitializeStructVectorShell(b_l[l+1]);
      data_size += hypre_StructVectorDataSize(b_l[l+1]);

      x_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(x_l[l+1], x_num_ghost);
      hypre_InitializeStructVectorShell(x_l[l+1]);
      data_size += hypre_StructVectorDataSize(x_l[l+1]);

      tb_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(tb_l[l+1], hypre_StructVectorNumGhost(b));
      hypre_InitializeStructVectorShell(tb_l[l+1]);

      tx_l[l+1] = hypre_NewStructVector(comm, grid_l[l+1]);
      hypre_SetStructVectorNumGhost(tx_l[l+1], hypre_StructVectorNumGhost(x));
      hypre_InitializeStructVectorShell(tx_l[l+1]);
   }

   data = hypre_CTAlloc(double, data_size);
   (smg_data -> data) = data;

   hypre_InitializeStructVectorData(tb_l[0], data);
   hypre_AssembleStructVector(tb_l[0]);
   data += hypre_StructVectorDataSize(tb_l[0]);

   hypre_InitializeStructVectorData(tx_l[0], data);
   hypre_AssembleStructVector(tx_l[0]);
   data += hypre_StructVectorDataSize(tx_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_InitializeStructMatrixData(PT_l[l], data);
      data += hypre_StructMatrixDataSize(PT_l[l]);

#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_InitializeStructMatrixData(R_l[l], data);
         data += hypre_StructMatrixDataSize(R_l[l]);
      }
#endif

      hypre_InitializeStructMatrixData(A_l[l+1], data);
      data += hypre_StructMatrixDataSize(A_l[l+1]);

      hypre_InitializeStructVectorData(b_l[l+1], data);
      hypre_AssembleStructVector(b_l[l+1]);
      data += hypre_StructVectorDataSize(b_l[l+1]);

      hypre_InitializeStructVectorData(x_l[l+1], data);
      hypre_AssembleStructVector(x_l[l+1]);
      data += hypre_StructVectorDataSize(x_l[l+1]);

      hypre_InitializeStructVectorData(tb_l[l+1],
                                       hypre_StructVectorData(tb_l[0]));
      hypre_AssembleStructVector(tb_l[l+1]);

      hypre_InitializeStructVectorData(tx_l[l+1],
                                       hypre_StructVectorData(tx_l[0]));
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
      hypre_SMGSetBIndex(base_index, base_stride, l, bindex);
      hypre_SMGSetBStride(base_index, base_stride, l, bstride);
      hypre_SMGSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_SMGSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_SMGSetStride(base_index, base_stride, l, cdir, stride);

      /* set up relaxation */
      relax_data_l[l] = hypre_SMGRelaxInitialize(comm);
      hypre_SMGRelaxSetBase(relax_data_l[l], bindex, bstride);
      hypre_SMGRelaxSetMemoryUse(relax_data_l[l], (smg_data -> memory_use));
      hypre_SMGRelaxSetTol(relax_data_l[l], 0.0);
      hypre_SMGRelaxSetNumSpaces(relax_data_l[l], 2);
      hypre_SMGRelaxSetSpace(relax_data_l[l], 0,
                             hypre_IndexD(cindex, cdir),
                             hypre_IndexD(stride, cdir));
      hypre_SMGRelaxSetSpace(relax_data_l[l], 1,
                             hypre_IndexD(findex, cdir),
                             hypre_IndexD(stride, cdir));
      hypre_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
      hypre_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
      hypre_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
      hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      hypre_SMGSetupInterpOp(relax_data_l[l], A_l[l], b_l[l], x_l[l],
                             PT_l[l], cdir, cindex, stride, findex, stride);

      /* (re)set relaxation parameters */
      hypre_SMGRelaxSetNumPreSpaces(relax_data_l[l], 0);
      hypre_SMGRelaxSetNumRegSpaces(relax_data_l[l], 2);
      hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      /* set up the residual routine */
      residual_data_l[l] = hypre_SMGResidualInitialize();
      hypre_SMGResidualSetBase(residual_data_l[l], bindex, bstride);
      hypre_SMGResidualSetup(residual_data_l[l],
                             A_l[l], x_l[l], b_l[l], r_l[l]);

      /* set up the interpolation routine */
      intadd_data_l[l] = hypre_SMGIntAddInitialize();
      hypre_SMGIntAddSetup(intadd_data_l[l], PT_l[l], x_l[l+1], e_l[l], x_l[l],
                           cindex, stride, findex, stride);

      /* set up the restriction operator */
#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
         hypre_SMGSetupRestrictOp(A_l[l], R_l[l], tx_l[l], cdir,
                                  cindex, stride);
#endif

      /* set up the restriction routine */
      restrict_data_l[l] = hypre_SMGRestrictInitialize();
      hypre_SMGRestrictSetup(restrict_data_l[l], R_l[l], r_l[l], b_l[l+1],
                             cindex, stride, findex, stride);

      /* set up the coarse grid operator */
      hypre_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1],
                          cindex, stride);
   }

   hypre_SMGSetBIndex(base_index, base_stride, l, bindex);
   hypre_SMGSetBStride(base_index, base_stride, l, bstride);
   relax_data_l[l] = hypre_SMGRelaxInitialize(comm);
   hypre_SMGRelaxSetBase(relax_data_l[l], bindex, bstride);
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

