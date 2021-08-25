/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "smg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetup( void               *smg_vdata,
                hypre_StructMatrix *A,
                hypre_StructVector *b,
                hypre_StructVector *x )
{
   hypre_SMGData        *smg_data = (hypre_SMGData        *)smg_vdata;

   MPI_Comm              comm = (smg_data -> comm);
   hypre_IndexRef        base_index  = (smg_data -> base_index);
   hypre_IndexRef        base_stride = (smg_data -> base_stride);

   HYPRE_Int             n_pre   = (smg_data -> num_pre_relax);
   HYPRE_Int             n_post  = (smg_data -> num_post_relax);

   HYPRE_Int             max_iter;
   HYPRE_Int             max_levels;

   HYPRE_Int             num_levels;

   HYPRE_Int             cdir;

   hypre_Index           bindex;
   hypre_Index           bstride;
   hypre_Index           cindex;
   hypre_Index           findex;
   hypre_Index           stride;

   hypre_StructGrid    **grid_l;
   hypre_StructGrid    **PT_grid_l;

   HYPRE_Real           *data;
   HYPRE_Real           *data_const;
   HYPRE_Int             data_size = 0;
   HYPRE_Int             data_size_const = 0;

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
   HYPRE_Real           *b_data;
   HYPRE_Real           *x_data;
   HYPRE_Int             b_data_alloced;
   HYPRE_Int             x_data_alloced;

   void                **relax_data_l;
   void                **residual_data_l;
   void                **restrict_data_l;
   void                **interp_data_l;

   hypre_StructGrid     *grid;

   hypre_Box            *cbox;

   HYPRE_Int             i, l;

   HYPRE_Int             b_num_ghost[]  = {0, 0, 0, 0, 0, 0};
   HYPRE_Int             x_num_ghost[]  = {0, 0, 0, 0, 0, 0};

#if DEBUG
   char                  filename[255];
#endif

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_Int             num_level_GPU = 0;
   HYPRE_MemoryLocation  data_location;
   HYPRE_Int             max_box_size  = 0;
   HYPRE_Int             device_level  = (smg_data -> devicelevel);
#endif
   /*-----------------------------------------------------
    * Set up coarsening direction
    *-----------------------------------------------------*/

   HYPRE_ANNOTATE_FUNC_BEGIN;

   cdir = hypre_StructStencilNDim(hypre_StructMatrixStencil(A)) - 1;
   (smg_data -> cdir) = cdir;

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid = hypre_StructMatrixGrid(A);

   /* Compute a new max_levels value based on the grid */
   cbox = hypre_BoxDuplicate(hypre_StructGridBoundingBox(grid));
   max_levels = hypre_Log2(hypre_BoxSizeD(cbox, cdir)) + 2;
   if ((smg_data -> max_levels) > 0)
   {
      max_levels = hypre_min(max_levels, (smg_data -> max_levels));
   }
   (smg_data -> max_levels) = max_levels;

   grid_l = hypre_TAlloc(hypre_StructGrid *,  max_levels, HYPRE_MEMORY_HOST);
   PT_grid_l = hypre_TAlloc(hypre_StructGrid *,  max_levels, HYPRE_MEMORY_HOST);
   PT_grid_l[0] = NULL;
   hypre_StructGridRef(grid, &grid_l[0]);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   data_location = hypre_StructGridDataLocation(grid);
   if (data_location != HYPRE_MEMORY_HOST)
   {
      num_level_GPU = max_levels;
   }
   else
   {
      num_level_GPU = 0;
      device_level  = 0;
   }
   if (hypre_StructGridNDim(grid) != hypre_StructStencilNDim(hypre_StructMatrixStencil(A)))
   {
      device_level = num_level_GPU;
   }
#endif
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      hypre_SMGSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_SMGSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      if ( ( hypre_BoxIMinD(cbox, cdir) == hypre_BoxIMaxD(cbox, cdir) ) ||
           (l == (max_levels - 1)) )
      {
         /* stop coarsening */
         break;
      }

      /* coarsen cbox */
      hypre_ProjectBox(cbox, cindex, stride);
      hypre_StructMapFineToCoarse(hypre_BoxIMin(cbox), cindex, stride,
                                  hypre_BoxIMin(cbox));
      hypre_StructMapFineToCoarse(hypre_BoxIMax(cbox), cindex, stride,
                                  hypre_BoxIMax(cbox));

      /* build the interpolation grid */
      hypre_StructCoarsen(grid_l[l], cindex, stride, 0, &PT_grid_l[l+1]);

      /* build the coarse grid */
      hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l+1]);
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypre_StructGridDataLocation(PT_grid_l[l+1]) = data_location;
      if (device_level == -1 && num_level_GPU > 0)
      {
         max_box_size = hypre_StructGridGetMaxBoxSize(grid_l[l+1]);
         if (max_box_size < HYPRE_MIN_GPU_SIZE)
         {
            num_level_GPU = l+1;
            data_location = HYPRE_MEMORY_HOST;
            device_level  = num_level_GPU;
            //printf("num_level_GPU = %d,device_level = %d\n",num_level_GPU,device_level);
         }
      }
      else if (l+1 == device_level)
      {
         num_level_GPU = l+1;
         data_location = HYPRE_MEMORY_HOST;
      }

      hypre_StructGridDataLocation(grid_l[l+1]) = data_location;
#endif
   }
   num_levels = l + 1;

   /* free up some things */
   hypre_BoxDestroy(cbox);

   (smg_data -> num_levels) = num_levels;
   (smg_data -> grid_l)     = grid_l;
   (smg_data -> PT_grid_l)  = PT_grid_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = hypre_TAlloc(hypre_StructMatrix *,  num_levels, HYPRE_MEMORY_HOST);
   PT_l = hypre_TAlloc(hypre_StructMatrix *,  num_levels - 1, HYPRE_MEMORY_HOST);
   R_l  = hypre_TAlloc(hypre_StructMatrix *,  num_levels - 1, HYPRE_MEMORY_HOST);
   b_l  = hypre_TAlloc(hypre_StructVector *,  num_levels, HYPRE_MEMORY_HOST);
   x_l  = hypre_TAlloc(hypre_StructVector *,  num_levels, HYPRE_MEMORY_HOST);
   tb_l = hypre_TAlloc(hypre_StructVector *,  num_levels, HYPRE_MEMORY_HOST);
   tx_l = hypre_TAlloc(hypre_StructVector *,  num_levels, HYPRE_MEMORY_HOST);
   r_l  = tx_l;
   e_l  = tx_l;

   A_l[0] = hypre_StructMatrixRef(A);
   b_l[0] = hypre_StructVectorRef(b);
   x_l[0] = hypre_StructVectorRef(x);

   for (i = 0; i <= cdir; i++)
   {
      x_num_ghost[2*i]     = 1;
      x_num_ghost[2*i + 1] = 1;
   }

   tb_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
   hypre_StructVectorSetNumGhost(tb_l[0], hypre_StructVectorNumGhost(b));
   hypre_StructVectorInitializeShell(tb_l[0]);
   hypre_StructVectorSetDataSize(tb_l[0], &data_size, &data_size_const);

   tx_l[0] = hypre_StructVectorCreate(comm, grid_l[0]);
   hypre_StructVectorSetNumGhost(tx_l[0], hypre_StructVectorNumGhost(x));
   hypre_StructVectorInitializeShell(tx_l[0]);
   hypre_StructVectorSetDataSize(tx_l[0], &data_size, &data_size_const);

   for (l = 0; l < (num_levels - 1); l++)
   {
      PT_l[l]  = hypre_SMGCreateInterpOp(A_l[l], PT_grid_l[l+1], cdir);

      hypre_StructMatrixInitializeShell(PT_l[l]);
      data_size += hypre_StructMatrixDataSize(PT_l[l]);
      data_size_const += hypre_StructMatrixDataConstSize(PT_l[l]);

      if (hypre_StructMatrixSymmetric(A))
      {
         R_l[l] = PT_l[l];
      }
      else
      {
         R_l[l] = PT_l[l];
#if 0
         /* Allow R != PT for non symmetric case */
         /* NOTE: Need to create a non-pruned grid for this to work */
         R_l[l]   = hypre_SMGCreateRestrictOp(A_l[l], grid_l[l+1], cdir);
         hypre_StructMatrixInitializeShell(R_l[l]);
         data_size += hypre_StructMatrixDataSize(R_l[l]);
         data_size_const += hypre_StructMatrixDataConstSize(R_l[l]);
#endif
      }

      A_l[l+1] = hypre_SMGCreateRAPOp(R_l[l], A_l[l], PT_l[l], grid_l[l+1]);
      hypre_StructMatrixInitializeShell(A_l[l+1]);
      data_size += hypre_StructMatrixDataSize(A_l[l+1]);
      data_size_const += hypre_StructMatrixDataConstSize(A_l[l+1]);

      b_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(b_l[l+1], b_num_ghost);
      hypre_StructVectorInitializeShell(b_l[l+1]);
      hypre_StructVectorSetDataSize(b_l[l+1], &data_size, &data_size_const);

      x_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(x_l[l+1], x_num_ghost);
      hypre_StructVectorInitializeShell(x_l[l+1]);
      hypre_StructVectorSetDataSize(x_l[l+1], &data_size, &data_size_const);

      tb_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(tb_l[l+1], hypre_StructVectorNumGhost(b));
      hypre_StructVectorInitializeShell(tb_l[l+1]);

      tx_l[l+1] = hypre_StructVectorCreate(comm, grid_l[l+1]);
      hypre_StructVectorSetNumGhost(tx_l[l+1], hypre_StructVectorNumGhost(x));
      hypre_StructVectorInitializeShell(tx_l[l+1]);
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (l+1 == num_level_GPU)
      {
         hypre_StructVectorSetDataSize(tb_l[l+1], &data_size, &data_size_const);
         hypre_StructVectorSetDataSize(tx_l[l+1], &data_size, &data_size_const);
      }
#endif
   }

   data = hypre_CTAlloc(HYPRE_Real, data_size, HYPRE_MEMORY_DEVICE);
   data_const = hypre_CTAlloc(HYPRE_Real,data_size_const,HYPRE_MEMORY_HOST);
   //printf("data =%d,data_const=%d,data_location = %d\n",data_size,data_size_const,data_location);
   (smg_data -> data) = data;
   (smg_data -> data_const) = data_const;

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   //if (hypre_StructGridNDim(grid) == hypre_StructStencilNDim(hypre_StructMatrixStencil(A)))
   //   printf("num_level_GPU = %d,device_level = %d / %d\n",num_level_GPU,device_level,num_levels);
   data_location = hypre_StructGridDataLocation(grid_l[0]);
   if (data_location != HYPRE_MEMORY_HOST)
   {
      hypre_StructVectorInitializeData(tb_l[0], data);
      hypre_StructVectorAssemble(tb_l[0]);
      data += hypre_StructVectorDataSize(tb_l[0]);
      hypre_StructVectorInitializeData(tx_l[0], data);
      hypre_StructVectorAssemble(tx_l[0]);
      data += hypre_StructVectorDataSize(tx_l[0]);
      //printf("smg_setup: Alloc tx_l[0] on GPU\n");
   }
   else
   {
      hypre_StructVectorInitializeData(tb_l[0], data_const);
      hypre_StructVectorAssemble(tb_l[0]);
      data_const += hypre_StructVectorDataSize(tb_l[0]);
      hypre_StructVectorInitializeData(tx_l[0], data_const);
      hypre_StructVectorAssemble(tx_l[0]);
      data_const += hypre_StructVectorDataSize(tx_l[0]);
      //printf("smg_setup: Alloc tx_l[0] on CPU\n");
   }
#else
   hypre_StructVectorInitializeData(tb_l[0], data);
   hypre_StructVectorAssemble(tb_l[0]);
   data += hypre_StructVectorDataSize(tb_l[0]);

   hypre_StructVectorInitializeData(tx_l[0], data);
   hypre_StructVectorAssemble(tx_l[0]);
   data += hypre_StructVectorDataSize(tx_l[0]);
#endif
   for (l = 0; l < (num_levels - 1); l++)
   {
      hypre_StructMatrixInitializeData(PT_l[l], data, data_const);
      data += hypre_StructMatrixDataSize(PT_l[l]);
      data_const += hypre_StructMatrixDataConstSize(PT_l[l]);

#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
      {
         hypre_StructMatrixInitializeData(R_l[l], data, data_const);
         data += hypre_StructMatrixDataSize(R_l[l]);
         data_const += hypre_StructMatrixDataConstSize(R_l[l]);
      }
#endif

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (l+1 == num_level_GPU)
      {
         data_location = HYPRE_MEMORY_HOST;
      }
#endif

      hypre_StructMatrixInitializeData(A_l[l+1], data, data_const);
      data += hypre_StructMatrixDataSize(A_l[l+1]);
      data_const += hypre_StructMatrixDataConstSize(A_l[l+1]);

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (data_location != HYPRE_MEMORY_HOST)
      {
         hypre_StructVectorInitializeData(b_l[l+1], data);
         hypre_StructVectorAssemble(b_l[l+1]);
         data += hypre_StructVectorDataSize(b_l[l+1]);

         hypre_StructVectorInitializeData(x_l[l+1], data);
         hypre_StructVectorAssemble(x_l[l+1]);
         data += hypre_StructVectorDataSize(x_l[l+1]);
         hypre_StructVectorInitializeData(tb_l[l+1],
               hypre_StructVectorData(tb_l[0]));
         hypre_StructVectorAssemble(tb_l[l+1]);

         hypre_StructVectorInitializeData(tx_l[l+1],
               hypre_StructVectorData(tx_l[0]));
         hypre_StructVectorAssemble(tx_l[l+1]);
         //printf("\n Alloc x_l,b_l[%d] on GPU\n",l+1);
      }
      else
      {
         hypre_StructVectorInitializeData(b_l[l+1], data_const);
         hypre_StructVectorAssemble(b_l[l+1]);
         data_const += hypre_StructVectorDataSize(b_l[l+1]);

         hypre_StructVectorInitializeData(x_l[l+1], data_const);
         hypre_StructVectorAssemble(x_l[l+1]);
         data_const += hypre_StructVectorDataSize(x_l[l+1]);
         if (l+1 == num_level_GPU)
         {
            hypre_StructVectorInitializeData(tb_l[l+1], data_const);
            hypre_StructVectorAssemble(tb_l[l+1]);
            data_const += hypre_StructVectorDataSize(tb_l[l+1]);
            hypre_StructVectorInitializeData(tx_l[l+1], data_const);
            hypre_StructVectorAssemble(tx_l[l+1]);
            data_const += hypre_StructVectorDataSize(tx_l[l+1]);
         }
         else
         {
            hypre_StructVectorInitializeData(tb_l[l+1],
                  hypre_StructVectorData(tb_l[num_level_GPU]));
            hypre_StructVectorAssemble(tb_l[l+1]);

            hypre_StructVectorInitializeData(tx_l[l+1],
                  hypre_StructVectorData(tx_l[num_level_GPU]));
            hypre_StructVectorAssemble(tx_l[l+1]);
         }
         //printf("\n Alloc x_l,b_l[%d] on CPU\n",l+1);
      }
#else

      hypre_StructVectorInitializeData(b_l[l+1], data);
      hypre_StructVectorAssemble(b_l[l+1]);
      data += hypre_StructVectorDataSize(b_l[l+1]);

      hypre_StructVectorInitializeData(x_l[l+1], data);
      hypre_StructVectorAssemble(x_l[l+1]);
      data += hypre_StructVectorDataSize(x_l[l+1]);

      hypre_StructVectorInitializeData(tb_l[l+1],
                                       hypre_StructVectorData(tb_l[0]));
      hypre_StructVectorAssemble(tb_l[l+1]);

      hypre_StructVectorInitializeData(tx_l[l+1],
                                       hypre_StructVectorData(tx_l[0]));
      hypre_StructVectorAssemble(tx_l[l+1]);
#endif
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

   relax_data_l    = hypre_TAlloc(void *,  num_levels, HYPRE_MEMORY_HOST);
   residual_data_l = hypre_TAlloc(void *,  num_levels, HYPRE_MEMORY_HOST);
   restrict_data_l = hypre_TAlloc(void *,  num_levels, HYPRE_MEMORY_HOST);
   interp_data_l   = hypre_TAlloc(void *,  num_levels, HYPRE_MEMORY_HOST);

   /* temporarily set the data for x_l[0] and b_l[0] to temp data */
   b_data = hypre_StructVectorData(b_l[0]);
   b_data_alloced = hypre_StructVectorDataAlloced(b_l[0]);
   x_data = hypre_StructVectorData(x_l[0]);
   x_data_alloced = hypre_StructVectorDataAlloced(x_l[0]);
   hypre_StructVectorInitializeData(b_l[0], hypre_StructVectorData(tb_l[0]));
   hypre_StructVectorInitializeData(x_l[0], hypre_StructVectorData(tx_l[0]));
   hypre_StructVectorAssemble(b_l[0]);
   hypre_StructVectorAssemble(x_l[0]);

   for (l = 0; l < (num_levels - 1); l++)
   {
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (l == num_level_GPU)
      {
         hypre_SetDeviceOff();
      }
#endif

      hypre_SMGSetBIndex(base_index, base_stride, l, bindex);
      hypre_SMGSetBStride(base_index, base_stride, l, bstride);
      hypre_SMGSetCIndex(base_index, base_stride, l, cdir, cindex);
      hypre_SMGSetFIndex(base_index, base_stride, l, cdir, findex);
      hypre_SMGSetStride(base_index, base_stride, l, cdir, stride);

      /* set up relaxation */
      relax_data_l[l] = hypre_SMGRelaxCreate(comm);
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
      //hypre_SMGRelaxSetMaxLevel( relax_data_l[l], l+6);
      hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      hypre_SMGSetupInterpOp(relax_data_l[l], A_l[l], b_l[l], x_l[l],
                             PT_l[l], cdir, cindex, findex, stride);

      /* (re)set relaxation parameters */
      hypre_SMGRelaxSetNumPreSpaces(relax_data_l[l], 0);
      hypre_SMGRelaxSetNumRegSpaces(relax_data_l[l], 2);
      hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

      /* set up the residual routine */
      residual_data_l[l] = hypre_SMGResidualCreate();
      hypre_SMGResidualSetBase(residual_data_l[l], bindex, bstride);
      hypre_SMGResidualSetup(residual_data_l[l],
                             A_l[l], x_l[l], b_l[l], r_l[l]);

      /* set up the interpolation routine */
      interp_data_l[l] = hypre_SemiInterpCreate();
      hypre_SemiInterpSetup(interp_data_l[l], PT_l[l], 1, x_l[l+1], e_l[l],
                            cindex, findex, stride);

      /* set up the restriction operator */
#if 0
      /* Allow R != PT for non symmetric case */
      if (!hypre_StructMatrixSymmetric(A))
         hypre_SMGSetupRestrictOp(A_l[l], R_l[l], tx_l[l], cdir,
                                  cindex, stride);
#endif
      /* set up the restriction routine */
      restrict_data_l[l] = hypre_SemiRestrictCreate();
      hypre_SemiRestrictSetup(restrict_data_l[l], R_l[l], 0, r_l[l], b_l[l+1],
                              cindex, findex, stride);

      /* set up the coarse grid operator */
      hypre_SMGSetupRAPOp(R_l[l], A_l[l], PT_l[l], A_l[l+1],
                          cindex, stride);
   }

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (l == num_level_GPU)
   {
      hypre_SetDeviceOff();
   }
#endif

   hypre_SMGSetBIndex(base_index, base_stride, l, bindex);
   hypre_SMGSetBStride(base_index, base_stride, l, bstride);
   relax_data_l[l] = hypre_SMGRelaxCreate(comm);
   hypre_SMGRelaxSetBase(relax_data_l[l], bindex, bstride);
   hypre_SMGRelaxSetTol(relax_data_l[l], 0.0);
   hypre_SMGRelaxSetMaxIter(relax_data_l[l], 1);
   hypre_SMGRelaxSetTempVec(relax_data_l[l], tb_l[l]);
   hypre_SMGRelaxSetNumPreRelax( relax_data_l[l], n_pre);
   hypre_SMGRelaxSetNumPostRelax( relax_data_l[l], n_post);
   hypre_SMGRelaxSetup(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

   /* set up the residual routine in case of a single grid level */
   if( l == 0 )
   {
      residual_data_l[l] = hypre_SMGResidualCreate();
      hypre_SMGResidualSetBase(residual_data_l[l], bindex, bstride);
      hypre_SMGResidualSetup(residual_data_l[l],
                             A_l[l], x_l[l], b_l[l], r_l[l]);
   }

   /* set the data for x_l[0] and b_l[0] the way they were */
   hypre_StructVectorInitializeData(b_l[0], b_data);
   hypre_StructVectorDataAlloced(b_l[0]) = b_data_alloced;
   hypre_StructVectorInitializeData(x_l[0], x_data);
   hypre_StructVectorDataAlloced(x_l[0]) = x_data_alloced;
   hypre_StructVectorAssemble(b_l[0]);
   hypre_StructVectorAssemble(x_l[0]);

   (smg_data -> relax_data_l)      = relax_data_l;
   (smg_data -> residual_data_l)   = residual_data_l;
   (smg_data -> restrict_data_l)   = restrict_data_l;
   (smg_data -> interp_data_l)     = interp_data_l;

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((smg_data -> logging) > 0)
   {
      max_iter = (smg_data -> max_iter);
      (smg_data -> norms)     = hypre_TAlloc(HYPRE_Real,  max_iter, HYPRE_MEMORY_HOST);
      (smg_data -> rel_norms) = hypre_TAlloc(HYPRE_Real,  max_iter, HYPRE_MEMORY_HOST);
   }

#if DEBUG
   if(hypre_StructGridNDim(grid_l[0]) == 3)
   {
      for (l = 0; l < (num_levels - 1); l++)
      {
         hypre_sprintf(filename, "zout_A.%02d", l);
         hypre_StructMatrixPrint(filename, A_l[l], 0);
         hypre_sprintf(filename, "zout_PT.%02d", l);
         hypre_StructMatrixPrint(filename, PT_l[l], 0);
      }
      hypre_sprintf(filename, "zout_A.%02d", l);
      hypre_StructMatrixPrint(filename, A_l[l], 0);
   }
#endif
#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (hypre_StructGridDataLocation(grid) != HYPRE_MEMORY_HOST)
   {
      hypre_SetDeviceOn();
   }
#endif
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
