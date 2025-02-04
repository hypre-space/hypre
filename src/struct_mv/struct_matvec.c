/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

#define UNROLL_MAXDEPTH 4

/*--------------------------------------------------------------------------
 * Matvec data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_StructMatrix  *A;
   hypre_Index          xfstride;
   hypre_ComputePkg    *compute_pkg;
   hypre_BoxArray      *data_space;
   HYPRE_Int            transpose;

   HYPRE_Int            nentries;
   HYPRE_Int           *stentries;

} hypre_StructMatvecData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_StructMatvecCreate( void )
{
   hypre_StructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_StructMatvecData, 1, HYPRE_MEMORY_HOST);
   matvec_data -> stentries = NULL;

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecDestroy( void *matvec_vdata )
{
   hypre_StructMatvecData *matvec_data = (hypre_StructMatvecData *)matvec_vdata;

   if (matvec_data)
   {
      hypre_StructMatrixDestroy(matvec_data -> A);
      hypre_ComputePkgDestroy(matvec_data -> compute_pkg);
      hypre_BoxArrayDestroy(matvec_data -> data_space);
      hypre_TFree(matvec_data -> stentries, HYPRE_MEMORY_HOST);
      hypre_TFree(matvec_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecSetTranspose( void *matvec_vdata,
                                HYPRE_Int transpose )
{
   hypre_StructMatvecData *matvec_data = (hypre_StructMatvecData *)matvec_vdata;

   (matvec_data -> transpose) = transpose;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * If needed, resize x and set up the compute package.  Assume that the same
 * matrix is passed into setup and compute, but the vector x may change.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecResize( hypre_StructMatvecData  *matvec_data,
                          hypre_StructVector      *x )
{
   hypre_StructMatrix      *A        = (matvec_data -> A);
   hypre_IndexRef           xfstride = (matvec_data -> xfstride);
   HYPRE_Int                ndim = hypre_StructMatrixNDim(A);

   hypre_StructGrid        *grid, *xgrid;
   hypre_StructStencil     *stencil;
   hypre_ComputeInfo       *compute_info;
   hypre_ComputePkg        *compute_pkg;
   hypre_BoxArray          *data_space;
   HYPRE_Int               *num_ghost;
   hypre_IndexRef           dom_stride, xstride, fstride;
   HYPRE_Int                d, need_resize, need_computepkg;

   /* Ensure that the base grid for x is at least as fine as the one for A */
   grid = hypre_StructMatrixGrid(A);
   if (matvec_data -> transpose)
   {
      dom_stride = hypre_StructMatrixRanStride(A);
   }
   else
   {
      dom_stride = hypre_StructMatrixDomStride(A);
   }
   hypre_StructVectorRebase(x, grid, dom_stride);

   /* Rebase may not change the base grid, so get the grid directly from x */
   xgrid = hypre_StructVectorGrid(x);
   xstride = hypre_StructVectorStride(x);

   /* Matrix stencil offsets are on the index space of the finest grid */
   hypre_StructMatrixGetFStride(A, &fstride);

   /* Need to compute xfstride such that (xgrid, xfstride) has the same index
    * space as (grid, fstride) where the stencil is applied.  Can do this by
    * using the fact that (xgrid, xstride) and (grid, dom_stride) have the same
    * index spaces, and hence xfstride/fstride = xstride/dom_stride.  Note that
    * if A and x end up having the same base grids, then xstride = dom_stride
    * and xfstride = fstride. */
   for (d = 0; d < ndim; d++)
   {
      xfstride[d] = fstride[d] * xstride[d] / dom_stride[d];
   }

   /* Compute the minimal data_space needed to resize x */
   stencil = hypre_StructMatrixStencil(A);
   hypre_StructNumGhostFromStencil(stencil, &num_ghost);
   hypre_StructVectorComputeDataSpace(x, xfstride, num_ghost, &data_space);
   hypre_TFree(num_ghost, HYPRE_MEMORY_HOST);

   /* Determine if a resize is needed */
   need_resize = hypre_StructVectorNeedResize(x, data_space);

   /* Determine if a compute package needs to be created */
   need_computepkg = 0;
   if ((matvec_data -> compute_pkg) == NULL)
   {
      /* compute package hasn't been created yet */
      need_computepkg = 1;
   }
   else if (need_resize)
   {
      /* a resize was needed */
      need_computepkg = 1;
   }
   else if ( !hypre_BoxArraysEqual(data_space, (matvec_data -> data_space)) )
   {
      /* the data space has changed */
      need_computepkg = 1;
   }

   /* Resize if needed */
   if (need_resize)
   {
      hypre_StructVectorResize(x, data_space);
   }
   else
   {
      hypre_BoxArrayDestroy(data_space);
      hypre_StructVectorRestore(x);
   }

   /* Create a compute package if needed */
   if (need_computepkg)
   {
      /* Note: It's important to use the data_space in x in case there is no resize */
      data_space = hypre_StructVectorDataSpace(x);

      /* This computes the communication pattern for the new x data_space */
      hypre_CreateComputeInfo(xgrid, xfstride, stencil, &compute_info);
      /* First refine commm_info to put it on the index space of xgrid, then map */
      /* NOTE: Compute boxes will be appropriately projected in MatvecCompute */
      hypre_CommInfoRefine(hypre_ComputeInfoCommInfo(compute_info), NULL, xfstride);
      hypre_StructVectorMapCommInfo(x, hypre_ComputeInfoCommInfo(compute_info));
      hypre_ComputePkgCreate(compute_info, data_space, 1, grid, &compute_pkg);

      /* Save compute_pkg */
      if ((matvec_data -> compute_pkg) != NULL)
      {
         hypre_ComputePkgDestroy(matvec_data -> compute_pkg);
      }
      if ((matvec_data -> data_space) != NULL)
      {
         hypre_BoxArrayDestroy(matvec_data -> data_space);
      }
      (matvec_data -> compute_pkg) = compute_pkg;
      (matvec_data -> data_space) = hypre_BoxArrayClone(data_space);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecSetup( void               *matvec_vdata,
                         hypre_StructMatrix *A,
                         hypre_StructVector *x )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_StructStencil     *stencil;
   HYPRE_Int                stencil_diag;
   HYPRE_Int                stencil_size;
   HYPRE_Int                i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* This is needed in MatvecResize() */
   (matvec_data -> A) = hypre_StructMatrixRef(A);

   /* Make sure that the transpose coefficients are stored in A (note that the
    * resizing will provide no option to restore A to its previous state) */
   if (matvec_data -> transpose)
   {
      HYPRE_Int  resize;

      hypre_StructMatrixSetTranspose(A, 1, &resize);
      /* RDF: The following probably should go inside the MatrixSetTranspose
       * routine to ensure it always gets done */
      if ( (resize) && (hypre_StructMatrixDataSpace(A) != NULL) )
      {
         hypre_BoxArray  *data_space;

         hypre_StructMatrixComputeDataSpace(A, NULL, &data_space);
         hypre_StructMatrixResize(A, data_space);
         hypre_StructMatrixForget(A);  /* No restore allowed (saves memory) */
         hypre_StructMatrixAssemble(A);
      }
   }

   /* Set active stencil entries if it hasn't been done yet */
   stencil = hypre_StructMatrixStencil(A);
   stencil_size = hypre_StructStencilSize(stencil);
   stencil_diag = hypre_StructStencilDiagEntry(stencil);
   (matvec_data -> stentries) = hypre_TAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      (matvec_data -> stentries[i]) = i;
   }

   /* Move diagonal entry to first position */
   (matvec_data -> stentries[stencil_diag]) = (matvec_data -> stentries[0]);
   (matvec_data -> stentries[0]) = stencil_diag;

   /* Set number of stencil entries used in StructMatvecCompute */
   (matvec_data -> nentries) = stencil_size;

   /* If needed, resize and set up the compute package */
   hypre_StructMatvecResize(matvec_data, x);

   /* Restore the original grid and data layout (depending on VectorMemoryMode) */
   hypre_StructVectorRestore(x);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * The grids for A, x, and y must be compatible with respect to matrix-vector
 * multiply, but the initial grid boxes and strides may differ.  The routines
 * Rebase() and Resize() are called to convert the grid for the vector x to
 * match the domain grid of the matrix A.  As a result, both A and x have the
 * same list of underlying boxes and the domain stride for A is the same as the
 * stride for x.  The grid for y is assumed to match the range grid for A, but
 * with potentially different boxes and strides.  The number of boxes and the
 * corresponding box ids must match, however, so it is not necessary to search.
 * Here are some examples (after Rebase() and Resize() have been called for x):
 *
 *   RangeIsCoarse:                           DomainIsCoarse:
 *   Adstride = 1                             Adstride = 1
 *   xdstride = 3                             xdstride = 1
 *   ydstride = 1                             ydstride = 3
 *
 *   1     6               2 2                5     2     6 6    <-- domain/range strides
 *   | |   |               | | |              | |   |     | | |
 *   |y| = |       A       | | |              | |   |     | |x|
 *   | |   |               | |x|              |y| = |  A  | | |
 *                           | |              | |   |     |
 *                           | |              | |   |     |
 *                           | |              | |   |     |
 *
 * It is assumed here that the data space for y corresponds to a coarsening of
 * the base index space for A with range stride.  So, we are circumventing the
 * "MapData" routines in the matrix and vector classes to avoid making too many
 * function calls.  We could achieve the same goal by calling MapToCoarse(),
 * MapToFine(), and MapData() in sequence to move base index values first to the
 * range grid for A, then to the base index space and data space for y.
 *
 * RDF TODO: Consider modifications to the current DataMap interfaces to avoid
 * making assumptions like the above.  Look at the Matmult routine as well.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute( void               *matvec_vdata,
                           HYPRE_Complex       alpha,
                           hypre_StructMatrix *A,
                           hypre_StructVector *x,
                           HYPRE_Complex       beta,
                           hypre_StructVector *y )
{
   hypre_StructMatvecData  *matvec_data = (hypre_StructMatvecData  *)matvec_vdata;

   hypre_IndexRef           xfstride    = (matvec_data -> xfstride);
   HYPRE_Int                transpose   = (matvec_data -> transpose);
   HYPRE_Int                nentries    = (matvec_data -> nentries);
   HYPRE_Int               *stentries   = (matvec_data -> stentries);
   HYPRE_Int                ndim        = hypre_StructMatrixNDim(A);

   hypre_CommHandle        *comm_handle;
   hypre_ComputePkg        *compute_pkg;

   hypre_BoxArrayArray     *compute_box_aa;
   hypre_BoxArray          *compute_box_a;
   hypre_Box               *compute_box;

   hypre_BoxArray          *boxes;
   hypre_Index              loop_size, origin, stride;
   hypre_IndexRef           start;

   HYPRE_Complex            temp;
   HYPRE_Int                compute_i, i, j, k, si, ssi;
   HYPRE_Int                csi[UNROLL_MAXDEPTH], vsi[UNROLL_MAXDEPTH];

   hypre_Box               *A_data_box, *x_data_box, *y_data_box;
   HYPRE_Complex           *Ap, *xp, *yp;
   HYPRE_Int                Ab, xb, yb;
   hypre_Index              Adstride, xdstride, ydstride, ustride;

   hypre_StructStencil     *stencil;
   hypre_Index             *stencil_shape;
   HYPRE_Int                stencil_size;

   HYPRE_Int                ran_nboxes;
   HYPRE_Int               *ran_boxnums;
   hypre_IndexRef           ran_stride;
   HYPRE_Int                dom_is_coarse;
   HYPRE_Int                ran_is_coarse;
   HYPRE_Int                depth, cdepth, vdepth;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/
   HYPRE_ANNOTATE_FUNC_BEGIN;

#if 0
   /* RDF: Should not need this if the boundaries were cleared initially */
   constant_coefficient = hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) { hypre_StructVectorClearBoundGhostValues(x, 0); }
#endif

   /* Switch domain and range information if doing a transpose matvec */
   if (transpose)
   {
      ran_stride    = hypre_StructMatrixDomStride(A);
      ran_nboxes    = hypre_StructMatrixDomNBoxes(A);
      ran_boxnums   = hypre_StructMatrixDomBoxnums(A);
      dom_is_coarse = hypre_StructMatrixRangeIsCoarse(A);
      ran_is_coarse = hypre_StructMatrixDomainIsCoarse(A);
   }
   else
   {
      ran_stride    = hypre_StructMatrixRanStride(A);
      ran_nboxes    = hypre_StructMatrixRanNBoxes(A);
      ran_boxnums   = hypre_StructMatrixRanBoxnums(A);
      dom_is_coarse = hypre_StructMatrixDomainIsCoarse(A);
      ran_is_coarse = hypre_StructMatrixRangeIsCoarse(A);
   }

   compute_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
      hypre_ForBoxI(i, boxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
         hypre_StructVectorMapDataBox(y, compute_box);
         hypre_BoxGetSize(compute_box, loop_size);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         start = hypre_BoxIMin(compute_box);
         yp = hypre_StructVectorBoxData(y, i);

#define DEVICE_VAR is_device_ptr(yp)
         hypre_BoxLoop1Begin(ndim, loop_size,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] *= beta;
         }
         hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
      }
      hypre_BoxDestroy(compute_box);

      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }

   if (x == y)
   {
      x_tmp = hypre_StructVectorClone(y);
      x = x_tmp;
   }

   /*-----------------------------------------------------------------------
    * Do (alpha != 0.0) computation
    *-----------------------------------------------------------------------*/

   /* If needed, resize and set up the compute package */
   hypre_StructMatvecResize(matvec_data, x);
   compute_pkg = (matvec_data -> compute_pkg);

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * Initialize y = -(beta/alpha)*y and multiply everything by -alpha
             * at the end.  This optimizes the matvec for residual computations
             * where alpha=-1 and beta=1.
             *--------------------------------------------------------------*/

            temp = -(beta / alpha);
            if (temp != 1.0)
            {
               HYPRE_ANNOTATE_REGION_BEGIN("%s", "Computation-Init");
               boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
               hypre_ForBoxI(i, boxes)
               {
                  hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
                  hypre_StructVectorMapDataBox(y, compute_box);
                  hypre_BoxGetSize(compute_box, loop_size);
                  y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
                  start = hypre_BoxIMin(compute_box);
                  yp = hypre_StructVectorBoxData(y, i);

#define DEVICE_VAR is_device_ptr(yp)
                  if (temp == 0.0)
                  {
                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         y_data_box, start, ustride, yi);
                     {
                        yp[yi] = 0.0;
                     }
                     hypre_BoxLoop1End(yi);
                  }
                  else
                  {
                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         y_data_box, start, ustride, yi);
                     {
                        yp[yi] *= temp;
                     }
                     hypre_BoxLoop1End(yi);
                  }
#undef DEVICE_VAR
               }
               HYPRE_ANNOTATE_REGION_END("%s", "Computation-Init");
            }
         }
         break;

         case 1:
         {
            hypre_FinalizeIndtComputations(comm_handle);
            compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * y += A*x (or A^T*x)
       *--------------------------------------------------------------------*/

      hypre_StructMatrixGetStencilStride(A, stride);
      hypre_CopyToIndex(stride, ndim, Adstride);
      hypre_StructMatrixMapDataStride(A, Adstride);
      hypre_CopyToIndex(stride, ndim, xdstride);
      hypre_MapToFineIndex(xdstride, NULL, xfstride, ndim);
      hypre_StructVectorMapDataStride(x, xdstride);
      hypre_CopyToIndex(stride, ndim, ydstride);
      hypre_MapToCoarseIndex(ydstride, NULL, ran_stride, ndim);

      xb = 0;
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Computation-Ax");
      for (i = 0; i < ran_nboxes; i++)
      {
         HYPRE_Int   *Aids = hypre_StructMatrixBoxIDs(A);
         HYPRE_Int   *xids = hypre_StructVectorBoxIDs(x);
         hypre_Index  Adstart;
         hypre_Index  xdstart;
         hypre_Index  ydstart;

         /* The corresponding box IDs for the following boxnums should match */
         Ab = ran_boxnums[i];
         /* Rebase ensures that all box ids in A are also in x */
         while (xids[xb] != Aids[Ab])
         {
            xb++;
         }
         yb = hypre_StructVectorBoxnum(y, i);

         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, xb);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), Ab);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), xb);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), yb);

         xp = hypre_StructVectorBoxData(x, xb);
         yp = hypre_StructVectorBoxData(y, yb);

         if (transpose || dom_is_coarse || ran_is_coarse)
         {
            hypre_ForBoxI(j, compute_box_a)
            {
               /* TODO optimization: Unroll these loops in a separate kernel */
               for (si = 0; si < nentries; si++)
               {
                  /* If the domain grid is coarse, the compute box will change
                   * based on the stencil entry.  Otherwise, the next code block
                   * needs to be called only on the first stencil iteration.
                   *
                   * Note that the Adstart and xdstart values are set in different
                   * places depending on the value of transpose. */

                  ssi = stentries[si];
                  if ((si == 0) || dom_is_coarse)
                  {
                     hypre_CopyBox(hypre_BoxArrayBox(compute_box_a, j), compute_box);
                     hypre_StructMatrixGetStencilSpace(A, ssi, transpose, origin, stride);
                     hypre_ProjectBox(compute_box, origin, stride);
                     start = hypre_BoxIMin(compute_box);

                     if (!transpose) /* Set Adstart here and xdstart below */
                     {
                        hypre_CopyToIndex(start, ndim, Adstart);
                        hypre_StructMatrixMapDataIndex(A, Adstart);
                     }

                     hypre_CopyToIndex(start, ndim, ydstart);
                     hypre_MapToCoarseIndex(ydstart, NULL, ran_stride, ndim);

                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);
                  }

                  if (transpose)
                  {
                     hypre_SubtractIndexes(start, stencil_shape[ssi], ndim, Adstart);
                     hypre_StructMatrixMapDataIndex(A, Adstart);
                     hypre_SubtractIndexes(start, stencil_shape[ssi], ndim, xdstart);
                     hypre_MapToFineIndex(xdstart, NULL, xfstride, ndim);
                     hypre_StructVectorMapDataIndex(x, xdstart);
                  }
                  else /* Set Adstart above and xdstart here */
                  {
                     hypre_AddIndexes(start, stencil_shape[ssi], ndim, xdstart);
                     hypre_MapToFineIndex(xdstart, NULL, xfstride, ndim);
                     hypre_StructVectorMapDataIndex(x, xdstart);
                  }

                  Ap = hypre_StructMatrixBoxData(A, Ab, ssi);
#define DEVICE_VAR is_device_ptr(yp,xp,Ap)
                  if (hypre_StructMatrixConstEntry(A, ssi))
                  {
                     /* Constant coefficient case */
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         x_data_box, xdstart, xdstride, xi,
                                         y_data_box, ydstart, ydstride, yi);
                     {
                        yp[yi] -= Ap[0] * xp[xi];
                     }
                     hypre_BoxLoop2End(xi, yi);
                  }
                  else
                  {
                     /* Variable coefficient case */
                     hypre_BoxLoop3Begin(ndim, loop_size,
                                         A_data_box, Adstart, Adstride, Ai,
                                         x_data_box, xdstart, xdstride, xi,
                                         y_data_box, ydstart, ydstride, yi);
                     {
                        yp[yi] -= Ap[Ai] * xp[xi];
                     }
                     hypre_BoxLoop3End(Ai, xi, yi);
                  }
#undef DEVICE_VAR
               } /* loop on stencil vars */
            } /* hypre_ForBoxI */
         }
         else
         {
            hypre_ForBoxI(j, compute_box_a)
            {
               hypre_CopyBox(hypre_BoxArrayBox(compute_box_a, j), compute_box);

               /* unroll up to depth UNROLL_MAXDEPTH */
               for (si = 0; si < nentries; si += UNROLL_MAXDEPTH)
               {
                  depth = hypre_min(UNROLL_MAXDEPTH, (stencil_size - si));

                  cdepth = vdepth = 0;
                  for (k = 0; k < depth; k++)
                  {
                     if (hypre_StructMatrixConstEntry(A, stentries[si + k]))
                     {
                        csi[cdepth++] = stentries[si + k];
                     }
                     else
                     {
                        vsi[vdepth++] = stentries[si + k];
                     }
                  }

                  /* Operate on constant coefficients */
                  hypre_StructMatvecCompute_core_CC(A, x, y, Ab, xb, yb, cdepth, csi, compute_box,
                                                    x_data_box, y_data_box);

                  /* Operate on variable coefficients */
                  hypre_StructMatvecCompute_core_VC(A, x, y, Ab, xb, yb, vdepth, vsi, compute_box,
                                                    A_data_box, x_data_box, y_data_box);
               } /* loop on stencil entries */
            } /* hypre_ForBoxI */
         } /* rectangular/square matrix branch */
      } /* loop on ran_nboxes */
      HYPRE_ANNOTATE_REGION_END("%s", "Computation-Ax");
   }

   temp = -alpha;
   if (temp != 1.0)
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Computation-Scale");
      boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
      hypre_ForBoxI(i, boxes)
      {
         hypre_CopyBox(hypre_BoxArrayBox(boxes, i), compute_box);
         hypre_StructVectorMapDataBox(y, compute_box);
         hypre_BoxGetSize(compute_box, loop_size);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         start = hypre_BoxIMin(compute_box);
         yp = hypre_StructVectorBoxData(y, i);

#define DEVICE_VAR is_device_ptr(yp)
         hypre_BoxLoop1Begin(ndim, loop_size,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] *= temp;
         }
         hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
      }
      HYPRE_ANNOTATE_REGION_END("%s", "Computation-Scale");
   }

   if (x_tmp)
   {
      /* Reset x to be the same as y */
      hypre_StructVectorDestroy(x_tmp);
      x = y;
   }
   else
   {
      /* Restore the original grid and data layout (depending on VectorMemoryMode) */
      hypre_StructVectorRestore(x);
   }
   hypre_BoxDestroy(compute_box);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute_core_CC
 *
 * StructMatrix/Vector multiplication core function for constant coeficients.
 *
 * Note: This function computes -A*x.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute_core_CC( hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   hypre_StructVector *y,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_Box          *compute_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *xp,  *yp;
   HYPRE_Int             xoff0, xoff1, xoff2;
   HYPRE_Int             xoff3, xoff4, xoff5;
   HYPRE_Int             xoff6, xoff7, xoff8;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   start = hypre_BoxIMin(compute_box);
   hypre_BoxGetSize(compute_box, loop_size);
   hypre_SetIndex(ustride, 1);
   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);

#define DEVICE_VAR is_device_ptr(yp,xp)
   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, Ab, entries[8]);
         xoff8 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[8]]);

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, Ab, entries[7]);
         xoff7 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[7]]);

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, Ab, entries[6]);
         xoff6 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[6]]);

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, Ab, entries[5]);
         xoff5 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[5]]);

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, Ab, entries[4]);
         xoff4 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[4]]);

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, Ab, entries[3]);
         xoff3 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[3]]);

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, Ab, entries[2]);
         xoff2 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[2]]);

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, Ab, entries[1]);
         xoff1 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[1]]);

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, Ab, entries[0]);
         xoff0 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[0]]);

      case 0:
         break;
   }

   switch (nentries)
   {
      case 9:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2] +
                      Ap3[0] * xp[xi + xoff3] +
                      Ap4[0] * xp[xi + xoff4] +
                      Ap5[0] * xp[xi + xoff5] +
                      Ap6[0] * xp[xi + xoff6] +
                      Ap7[0] * xp[xi + xoff7] +
                      Ap8[0] * xp[xi + xoff8];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 8:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);

         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2] +
                      Ap3[0] * xp[xi + xoff3] +
                      Ap4[0] * xp[xi + xoff4] +
                      Ap5[0] * xp[xi + xoff5] +
                      Ap6[0] * xp[xi + xoff6] +
                      Ap7[0] * xp[xi + xoff7];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 7:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2] +
                      Ap3[0] * xp[xi + xoff3] +
                      Ap4[0] * xp[xi + xoff4] +
                      Ap5[0] * xp[xi + xoff5] +
                      Ap6[0] * xp[xi + xoff6];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 6:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2] +
                      Ap3[0] * xp[xi + xoff3] +
                      Ap4[0] * xp[xi + xoff4] +
                      Ap5[0] * xp[xi + xoff5];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 5:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2] +
                      Ap3[0] * xp[xi + xoff3] +
                      Ap4[0] * xp[xi + xoff4];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 4:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2] +
                      Ap3[0] * xp[xi + xoff3];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 3:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1] +
                      Ap2[0] * xp[xi + xoff2];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 2:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0] +
                      Ap1[0] * xp[xi + xoff1];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 1:
         hypre_BoxLoop2Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[0] * xp[xi + xoff0];
         }
         hypre_BoxLoop2End(xi, yi);
         break;

      case 0:
         break;
   } /* switch (nentries) */
#undef DEVICE_VAR

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute_core_VC
 *
 * StructMatrix/Vector multiplication core routine for variable coeficients.
 *
 * Note: This function computes -A*x.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VC( hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   hypre_StructVector *y,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_Box          *compute_box,
                                   hypre_Box          *A_data_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *xp,  *yp;
   HYPRE_Int             xoff0, xoff1, xoff2;
   HYPRE_Int             xoff3, xoff4, xoff5;
   HYPRE_Int             xoff6, xoff7, xoff8;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   start = hypre_BoxIMin(compute_box);
   hypre_BoxGetSize(compute_box, loop_size);
   hypre_SetIndex(ustride, 1);
   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);

#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8)
   switch (nentries)
   {
      case 9:
         Ap8 = hypre_StructMatrixBoxData(A, Ab, entries[8]);
         xoff8 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[8]]);

      case 8:
         Ap7 = hypre_StructMatrixBoxData(A, Ab, entries[7]);
         xoff7 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[7]]);

      case 7:
         Ap6 = hypre_StructMatrixBoxData(A, Ab, entries[6]);
         xoff6 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[6]]);

      case 6:
         Ap5 = hypre_StructMatrixBoxData(A, Ab, entries[5]);
         xoff5 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[5]]);

      case 5:
         Ap4 = hypre_StructMatrixBoxData(A, Ab, entries[4]);
         xoff4 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[4]]);

      case 4:
         Ap3 = hypre_StructMatrixBoxData(A, Ab, entries[3]);
         xoff3 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[3]]);

      case 3:
         Ap2 = hypre_StructMatrixBoxData(A, Ab, entries[2]);
         xoff2 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[2]]);

      case 2:
         Ap1 = hypre_StructMatrixBoxData(A, Ab, entries[1]);
         xoff1 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[1]]);

      case 1:
         Ap0 = hypre_StructMatrixBoxData(A, Ab, entries[0]);
         xoff0 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[0]]);

      case 0:
         break;
   }

   switch (nentries)
   {
      case 9:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2] +
                      Ap3[Ai] * xp[xi + xoff3] +
                      Ap4[Ai] * xp[xi + xoff4] +
                      Ap5[Ai] * xp[xi + xoff5] +
                      Ap6[Ai] * xp[xi + xoff6] +
                      Ap7[Ai] * xp[xi + xoff7] +
                      Ap8[Ai] * xp[xi + xoff8];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 8:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2] +
                      Ap3[Ai] * xp[xi + xoff3] +
                      Ap4[Ai] * xp[xi + xoff4] +
                      Ap5[Ai] * xp[xi + xoff5] +
                      Ap6[Ai] * xp[xi + xoff6] +
                      Ap7[Ai] * xp[xi + xoff7];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 7:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2] +
                      Ap3[Ai] * xp[xi + xoff3] +
                      Ap4[Ai] * xp[xi + xoff4] +
                      Ap5[Ai] * xp[xi + xoff5] +
                      Ap6[Ai] * xp[xi + xoff6];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 6:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2] +
                      Ap3[Ai] * xp[xi + xoff3] +
                      Ap4[Ai] * xp[xi + xoff4] +
                      Ap5[Ai] * xp[xi + xoff5];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 5:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2] +
                      Ap3[Ai] * xp[xi + xoff3] +
                      Ap4[Ai] * xp[xi + xoff4];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 4:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2] +
                      Ap3[Ai] * xp[xi + xoff3];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 3:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1] +
                      Ap2[Ai] * xp[xi + xoff2];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 2:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0] +
                      Ap1[Ai] * xp[xi + xoff1];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 1:
         hypre_BoxLoop3Begin(ndim, loop_size,
                             A_data_box, start, ustride, Ai,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi);
         {
            yp[yi] -= Ap0[Ai] * xp[xi + xoff0];
         }
         hypre_BoxLoop3End(Ai, xi, yi);
         break;

      case 0:
         break;
   } /* switch (nentries) */
#undef DEVICE_VAR

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvec( HYPRE_Complex       alpha,
                    hypre_StructMatrix *A,
                    hypre_StructVector *x,
                    HYPRE_Complex       beta,
                    hypre_StructVector *y )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetup(matvec_data, A, x);
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecT( HYPRE_Complex       alpha,
                     hypre_StructMatrix *A,
                     hypre_StructVector *x,
                     HYPRE_Complex       beta,
                     hypre_StructVector *y )
{
   void *matvec_data;

   matvec_data = hypre_StructMatvecCreate();
   hypre_StructMatvecSetTranspose(matvec_data, 1);
   hypre_StructMatvecSetup(matvec_data, A, x);
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}
