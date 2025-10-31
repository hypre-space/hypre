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
#include "struct_matvec_core.h"

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
   HYPRE_Int                ndim     = hypre_StructMatrixNDim(A);
   HYPRE_MemoryLocation     memloc   = hypre_StructVectorMemoryLocation(x);

   hypre_StructGrid        *grid, *xgrid;
   hypre_StructStencil     *stencil;
   hypre_ComputeInfo       *compute_info;
   hypre_ComputePkg        *compute_pkg;
   hypre_BoxArray          *data_space;
   HYPRE_Int               *num_ghost;
   hypre_IndexRef           dom_stride, xstride, fstride;
   HYPRE_Int                d, need_resize, need_computepkg;

   HYPRE_ANNOTATE_FUNC_BEGIN;

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
   else if (!hypre_BoxArraysEqual(data_space, (matvec_data -> data_space)))
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
      hypre_ComputePkgCreate(memloc, compute_info, data_space, 1, grid, &compute_pkg);

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
                           hypre_StructVector *y,
                           hypre_StructVector *z )
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

   hypre_Index              loop_size, origin, stride;
   hypre_IndexRef           start;

   HYPRE_Int                compute_i, i, j, si, se;
   HYPRE_Int                cnentries, vnentries;
   HYPRE_Int                centries[HYPRE_MAX_MMTERMS], ventries[HYPRE_MAX_MMTERMS];

   hypre_Box               *A_data_box, *x_data_box, *y_data_box, *z_data_box;
   HYPRE_Complex           *xp;
   HYPRE_Int                Ab, xb, yb, zb;
   hypre_Index              Adstride, xdstride, ydstride, zdstride, ustride;

   HYPRE_Int                ran_nboxes;
   HYPRE_Int               *ran_boxnums;
   hypre_IndexRef           ran_stride;

   hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      hypre_StructVectorAxpy(alpha, y, beta, y, z);
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("Matvec");

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
   }
   else
   {
      ran_stride    = hypre_StructMatrixRanStride(A);
      ran_nboxes    = hypre_StructMatrixRanNBoxes(A);
      ran_boxnums   = hypre_StructMatrixRanBoxnums(A);
   }

   compute_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

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

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
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
       * z = beta * y + alpha * A*x (or A^T*x)
       *--------------------------------------------------------------------*/

      hypre_StructMatrixGetStencilStride(A, stride);
      hypre_CopyToIndex(stride, ndim, Adstride);
      hypre_StructMatrixMapDataStride(A, Adstride);

      hypre_CopyToIndex(stride, ndim, xdstride);
      hypre_MapToFineIndex(xdstride, NULL, xfstride, ndim);
      hypre_StructVectorMapDataStride(x, xdstride);

      hypre_CopyToIndex(stride, ndim, ydstride);
      hypre_MapToCoarseIndex(ydstride, NULL, ran_stride, ndim);

      hypre_CopyToIndex(stride, ndim, zdstride);
      hypre_MapToCoarseIndex(zdstride, NULL, ran_stride, ndim);

      xb = 0;
      for (i = 0; i < ran_nboxes; i++)
      {
         HYPRE_Int   *Aids = hypre_StructMatrixBoxIDs(A);
         HYPRE_Int   *xids = hypre_StructVectorBoxIDs(x);
         HYPRE_Int    num_ss;
         HYPRE_Int   *se_sspaces;
         hypre_Index *ss_origins;
         HYPRE_Int    s;

         /* The corresponding box IDs for the following boxnums should match */
         Ab = ran_boxnums[i];

         /* Rebase ensures that all box ids in A are also in x */
         while (xids[xb] != Aids[Ab])
         {
            xb++;
         }
         yb = hypre_StructVectorBoxnum(y, i);
         zb = hypre_StructVectorBoxnum(z, i);

         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, xb);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), Ab);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), xb);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), yb);
         z_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(z), zb);

         /* Get stencil spaces - then do unrolling for entries in each */
         hypre_StructMatrixGetStSpaces(A, transpose, &num_ss, &se_sspaces, &ss_origins, stride);

         for (s = 0; s < num_ss; s++)
         {
            /* Only compute with entries in stencil space s */
            cnentries = vnentries = 0;
            for (si = 0; si < nentries; si++)
            {
               se = stentries[si];
               if (se_sspaces[se] == s)
               {
                  if (hypre_StructMatrixConstEntry(A, se))
                  {
                     centries[cnentries++] = se;
                  }
                  else
                  {
                     ventries[vnentries++] = se;
                  }
               }
            }

            hypre_ForBoxI(j, compute_box_a)
            {
               hypre_CopyBox(hypre_BoxArrayBox(compute_box_a, j), compute_box);
               hypre_CopyToIndex(ss_origins[s], ndim, origin);
               hypre_ProjectBox(compute_box, origin, stride);
               start = hypre_BoxIMin(compute_box);
               hypre_BoxGetStrideSize(compute_box, stride, loop_size);

               if (cnentries == 1 && vnentries > 0)
               {
                  /* One constant and an arbitrary number of variable coefficients */
                  hypre_StructMatvecCompute_core_VCC(alpha, A, x, beta, y, z,
                                                     Ab, xb, yb, zb, transpose,
                                                     centries[0], vnentries, ventries,
                                                     start, stride, loop_size, xfstride, ran_stride,
                                                     Adstride, xdstride, ydstride, zdstride,
                                                     A_data_box, x_data_box, y_data_box, z_data_box);
               }
               else
               {
                  /* Operate on constant coefficients */
                  hypre_StructMatvecCompute_core_CC(alpha, A, x, beta, y, z,
                                                    Ab, xb, yb, zb, transpose,
                                                    cnentries, centries,
                                                    start, stride, loop_size, xfstride, ran_stride,
                                                    xdstride, ydstride, zdstride,
                                                    x_data_box, y_data_box, z_data_box);

                  /* Operate on variable coefficients */
                  hypre_StructMatvecCompute_core_VC(alpha, A, x, beta, y, z,
                                                    Ab, xb, yb, zb, transpose,
                                                    (cnentries > 0), vnentries, ventries,
                                                    start, stride, loop_size, xfstride, ran_stride,
                                                    Adstride, xdstride, ydstride, zdstride,
                                                    A_data_box, x_data_box, y_data_box, z_data_box);
               }
            } /* hypre_ForBoxI */
         } /* for stencil space s */

         hypre_TFree(se_sspaces, HYPRE_MEMORY_HOST);
         hypre_TFree(ss_origins, HYPRE_MEMORY_HOST);
      }
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

   hypre_GpuProfilingPopRange();
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
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
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
   hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y, y);
   hypre_StructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}
