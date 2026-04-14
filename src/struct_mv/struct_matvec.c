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
   hypre_StructGrid        *grid     = hypre_StructMatrixGrid(A);
   HYPRE_MemoryLocation     memloc   = hypre_StructVectorMemoryLocation(x);

   hypre_StructStencil     *stencil;
   hypre_ComputeInfo       *compute_info;
   hypre_ComputePkg        *compute_pkg;
   hypre_BoxArray          *data_space;
   HYPRE_Int               *num_ghost;
   hypre_IndexRef           dom_stride;
   HYPRE_Int                need_resize, need_computepkg;

   hypre_Index              ustride;
   hypre_SetIndex(ustride, 1);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Set dom_stride based on transpose flag */
   if (matvec_data -> transpose)
   {
      dom_stride = hypre_StructMatrixRanStride(A);
   }
   else
   {
      dom_stride = hypre_StructMatrixDomStride(A);
   }

   /* Compute the minimal data_space needed to resize x */
   stencil = hypre_StructMatrixStencil(A);
   hypre_StructNumGhostFromStencil(stencil, &num_ghost);
   hypre_StructVectorComputeDataSpace(x, NULL, num_ghost, &data_space);  // RDF: too big for now
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

      /* This computes the communication pattern for the x data_space.
         Pass in the matrix grid and ustride - map (coarsen) to x data space */
      hypre_CreateComputeInfo(grid, ustride, stencil, &compute_info);

      /* Map (coarsen) the comm info - compute boxes will be mapped in MatvecCompute */
      hypre_CommInfoCoarsen(hypre_ComputeInfoCommInfo(compute_info), NULL, dom_stride);
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
 *
 * RDF BASE - check the above comments for correctness
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
   hypre_Index              Adstride, xdstride, ydstride, zdstride, ustride;

   HYPRE_Int                ran_nboxes;
   HYPRE_Int               *ran_boxnums;
   hypre_IndexRef           ran_stride, dom_stride;

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
      ran_stride  = hypre_StructMatrixDomStride(A);
      ran_nboxes  = hypre_StructMatrixDomNBoxes(A);
      ran_boxnums = hypre_StructMatrixDomBoxnums(A);
      dom_stride  = hypre_StructMatrixRanStride(A);
   }
   else
   {
      ran_stride  = hypre_StructMatrixRanStride(A);
      ran_nboxes  = hypre_StructMatrixRanNBoxes(A);
      ran_boxnums = hypre_StructMatrixRanBoxnums(A);
      dom_stride  = hypre_StructMatrixDomStride(A);
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
      hypre_MapToCoarseIndex(xdstride, NULL, dom_stride, ndim);

      hypre_CopyToIndex(stride, ndim, ydstride);
      hypre_MapToCoarseIndex(ydstride, NULL, ran_stride, ndim);

      hypre_CopyToIndex(stride, ndim, zdstride);
      hypre_MapToCoarseIndex(zdstride, NULL, ran_stride, ndim);

      for (i = 0; i < ran_nboxes; i++)
      {
         HYPRE_Int    num_ss;
         HYPRE_Int   *se_sspaces;
         hypre_Index *ss_origins;
         HYPRE_Int    bb, s;

         compute_box_a = hypre_BoxArrayArrayBoxArray(compute_box_aa, ran_boxnums[i]);
         if (hypre_BoxArraySize(compute_box_a) == 0)
         {
            /* Nothing to compute so go to the next range box */
            continue;
         }

         bb = hypre_StructMatrixBaseBoxnum(A, ran_boxnums[i]);

         A_data_box = hypre_StructMatrixBaseDataBox(A, bb);
         x_data_box = hypre_StructVectorBaseDataBox(x, bb);
         y_data_box = hypre_StructVectorBaseDataBox(y, bb);
         z_data_box = hypre_StructVectorBaseDataBox(z, bb);

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
                  hypre_StructMatvecCompute_core_VCC(
                     alpha, A, x, beta, y, z, bb, transpose, centries[0], vnentries, ventries,
                     start, stride, loop_size, ran_stride, dom_stride,
                     Adstride, xdstride, ydstride, zdstride,
                     A_data_box, x_data_box, y_data_box, z_data_box);
               }
               else
               {
                  /* Operate on constant coefficients */
                  hypre_StructMatvecCompute_core_CC(
                     alpha, A, x, beta, y, z, bb, transpose, cnentries, centries,
                     start, stride, loop_size, ran_stride, dom_stride,
                     xdstride, ydstride, zdstride,
                     x_data_box, y_data_box, z_data_box);

                  /* Operate on variable coefficients */
                  hypre_StructMatvecCompute_core_VC(
                     alpha, A, x, beta, y, z, bb, transpose, (cnentries > 0), vnentries, ventries,
                     start, stride, loop_size, ran_stride, dom_stride,
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
