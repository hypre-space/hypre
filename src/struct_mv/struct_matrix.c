/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetFStride( hypre_StructMatrix *matrix,
                              hypre_IndexRef     *fstride )
{
   *fstride = hypre_StructMatrixRanStride(matrix);
   if (hypre_StructMatrixRangeIsCoarse(matrix))
   {
      *fstride = hypre_StructMatrixDomStride(matrix);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetCStride( hypre_StructMatrix *matrix,
                              hypre_IndexRef     *cstride )
{
   *cstride = hypre_StructMatrixRanStride(matrix);
   if (hypre_StructMatrixDomainIsCoarse(matrix))
   {
      *cstride = hypre_StructMatrixDomStride(matrix);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Matrix data is currently stored relative to the coarse matrix stride
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetDataMapStride( hypre_StructMatrix *matrix,
                                    hypre_IndexRef     *stride )
{
   hypre_StructMatrixGetCStride(matrix, stride);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine assumes that 'dindex' is in the range index space.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMapDataIndex( hypre_StructMatrix *matrix,
                                hypre_Index         dindex )
{
   HYPRE_Int      ndim = hypre_StructMatrixNDim(matrix);
   hypre_IndexRef stride;

   hypre_StructMatrixGetDataMapStride(matrix, &stride);
   if (hypre_StructMatrixDomainIsCoarse(matrix))
   {
      hypre_SnapIndexNeg(dindex, NULL, stride, ndim);
   }
   hypre_MapToCoarseIndex(dindex, NULL, stride, ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine first ensures that the lower and upper indexes of 'dbox' are in
 * the range index space.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMapDataBox( hypre_StructMatrix *matrix,
                              hypre_Box          *dbox )
{
   if (hypre_StructMatrixRangeIsCoarse(matrix))
   {
      hypre_ProjectBox(dbox, NULL, hypre_StructMatrixRanStride(matrix));
   }
   hypre_StructMatrixMapDataIndex(matrix, hypre_BoxIMin(dbox));
   hypre_StructMatrixMapDataIndex(matrix, hypre_BoxIMax(dbox));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMapDataStride( hypre_StructMatrix *matrix,
                                 hypre_Index         dstride )
{
   HYPRE_Int      ndim = hypre_StructMatrixNDim(matrix);
   hypre_IndexRef stride;

   hypre_StructMatrixGetDataMapStride(matrix, &stride);
   hypre_MapToCoarseIndex(dstride, NULL, stride, ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixUnMapDataIndex( hypre_StructMatrix *matrix,
                                  hypre_Index         dindex )
{
   HYPRE_Int      ndim = hypre_StructMatrixNDim(matrix);
   hypre_IndexRef stride;

   hypre_StructMatrixGetDataMapStride(matrix, &stride);
   hypre_MapToFineIndex(dindex, NULL, stride, ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixUnMapDataBox( hypre_StructMatrix *matrix,
                                hypre_Box          *dbox )
{
   hypre_StructMatrixUnMapDataIndex(matrix, hypre_BoxIMin(dbox));
   hypre_StructMatrixUnMapDataIndex(matrix, hypre_BoxIMax(dbox));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixUnMapDataStride( hypre_StructMatrix *matrix,
                                   hypre_Index         dstride )
{
   HYPRE_Int      ndim = hypre_StructMatrixNDim(matrix);
   hypre_IndexRef stride;

   hypre_StructMatrixGetDataMapStride(matrix, &stride);
   hypre_MapToFineIndex(dstride, NULL, stride, ndim);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Places the center of the stencil correctly on the base index space given a
 * stencil entry and a data index.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPlaceStencil( hypre_StructMatrix *matrix,
                                HYPRE_Int           entry,
                                hypre_Index         dindex,
                                hypre_Index         index )
{
   HYPRE_Int  ndim = hypre_StructMatrixNDim(matrix);

   /* Map the data index to the base index space */
   hypre_CopyToIndex(dindex, ndim, index);
   hypre_StructMatrixUnMapDataIndex(matrix, index);

   if (hypre_StructMatrixDomainIsCoarse(matrix))
   {
      hypre_IndexRef        stride  = hypre_StructMatrixDomStride(matrix);
      hypre_StructStencil  *stencil = hypre_StructMatrixStencil(matrix);
      hypre_IndexRef        offset  = hypre_StructStencilOffset(stencil, entry);
      hypre_Index           snapoffset;

      /* Shift to the right based on offset: index += (SnapIndexPos(offset) - offset) */
      hypre_CopyToIndex(offset, ndim, snapoffset);
      hypre_SnapIndexPos(snapoffset, NULL, stride, ndim);
      hypre_AddIndexes(index, snapoffset, ndim, index);  /* + SnapIndexPos(offset) */
      hypre_SubtractIndexes(index, offset, ndim, index); /* - offset */
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Same as hypre_StructMatrixGetStencilSpace(), but just returns the stride.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetStencilStride( hypre_StructMatrix *matrix,
                                    hypre_Index         stride )
{
   HYPRE_Int  ndim = hypre_StructMatrixNDim(matrix);

   if (hypre_StructMatrixDomainIsCoarse(matrix))
   {
      hypre_CopyToIndex(hypre_StructMatrixDomStride(matrix), ndim, stride);
   }
   else
   {
      hypre_CopyToIndex(hypre_StructMatrixRanStride(matrix), ndim, stride);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the (origin,stride) index space for a given stencil entry in a
 * canonical representation with origin in the interval [0, stride).
 * If transpose is set, the transpose of the matrix is used instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetStencilSpace( hypre_StructMatrix *matrix,
                                   HYPRE_Int           entry,
                                   HYPRE_Int           transpose,
                                   hypre_Index         origin,
                                   hypre_Index         stride )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(matrix);
   hypre_IndexRef        offset  = hypre_StructStencilOffset(stencil, entry);

   /* Initialize origin */
   hypre_SetIndex(origin, 0);

   /* Note that if stride = 1, then ConvertToCanonicalIndex() below would always
    * set origin = 0, so the 'if' checks for Domain/RangeIsCoarse only eliminate
    * unecessary computations; they are not strictly needed. */
   hypre_StructMatrixGetStencilStride(matrix, stride);
   if ( !transpose && hypre_StructMatrixDomainIsCoarse(matrix) )
   {
      hypre_SubtractIndexes(origin, offset, ndim, origin); /* origin -= offset */
      /* Convert origin to a canonical representation in [0, stride) */
      hypre_ConvertToCanonicalIndex(origin, stride, ndim);
   }
   else if ( transpose && hypre_StructMatrixRangeIsCoarse(matrix) )
   {
      hypre_AddIndexes(origin, offset, ndim, origin); /* origin += offset */
      /* Convert origin to a canonical representation in [0, stride) */
      hypre_ConvertToCanonicalIndex(origin, stride, ndim);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute the unique set of "stencil spaces", the stencil entries that belong
 * to each space, and the corresponding origins and stride (only one).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetStSpaces( hypre_StructMatrix *matrix,
                               HYPRE_Int           transpose,
                               HYPRE_Int          *num_sspaces_ptr,
                               HYPRE_Int         **sentry_sspaces_ptr,
                               hypre_Index       **sspace_origins_ptr,
                               hypre_Index         stride )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(matrix);
   HYPRE_Int             stencil_size = hypre_StructStencilSize(stencil);

   HYPRE_Int             num_sspaces;
   HYPRE_Int            *sentry_sspaces;
   hypre_Index          *sspace_origins;

   hypre_Index           origin;
   HYPRE_Int             e, s;

   sentry_sspaces = hypre_TAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
   sspace_origins = hypre_TAlloc(hypre_Index, stencil_size, HYPRE_MEMORY_HOST);
   num_sspaces = 0;
   for (e = 0; e < stencil_size; e++)
   {
      hypre_StructMatrixGetStencilSpace(matrix, e, transpose, origin, stride);

      for (s = 0; s < num_sspaces; s++)
      {
         /* Only check origin (assume that stride is always the same) */
         if ( hypre_IndexesEqual(origin, sspace_origins[s], ndim) )
         {
            break;
         }
      }
      sentry_sspaces[e] = s;

      if (s == num_sspaces)
      {
         /* This is a new space */
         hypre_CopyToIndex(origin, ndim, sspace_origins[s]);
         num_sspaces++;
      }
   }

   /* Set return values */
   *num_sspaces_ptr    = num_sspaces;
   *sentry_sspaces_ptr = sentry_sspaces;
   *sspace_origins_ptr = sspace_origins;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMapCommInfo( hypre_StructMatrix *matrix,
                               hypre_IndexRef      origin,
                               hypre_Index         stride,
                               hypre_CommInfo     *comm_info )
{
   HYPRE_Int       ndim   = hypre_StructMatrixNDim(matrix);
   hypre_IndexRef  dmstride;

   /* Map the comm_info boxes only for non-unit stride */
   hypre_StructMatrixGetDataMapStride(matrix, &dmstride);
   if ( !hypre_IndexEqual(stride, 1, ndim) ||
        !hypre_IndexEqual(dmstride, 1, ndim) )
   {
      hypre_BoxArrayArray  *boxaa;
      hypre_BoxArray       *boxa;
      hypre_Box            *box;
      HYPRE_Int             i, j;

      boxaa = hypre_CommInfoSendBoxes(comm_info);
      hypre_ForBoxArrayI(i, boxaa)
      {
         boxa = hypre_BoxArrayArrayBoxArray(boxaa, i);
         hypre_ForBoxI(j, boxa)
         {
            box = hypre_BoxArrayBox(boxa, j);
            hypre_ProjectBox(box, origin, stride);
            hypre_StructMatrixMapDataBox(matrix, box);
         }
      }

      boxaa = hypre_CommInfoSendRBoxes(comm_info);
      hypre_ForBoxArrayI(i, boxaa)
      {
         boxa = hypre_BoxArrayArrayBoxArray(boxaa, i);
         hypre_ForBoxI(j, boxa)
         {
            box = hypre_BoxArrayBox(boxa, j);
            hypre_ProjectBox(box, origin, stride);
            hypre_StructMatrixMapDataBox(matrix, box);
         }
      }

      boxaa = hypre_CommInfoRecvBoxes(comm_info);
      hypre_ForBoxArrayI(i, boxaa)
      {
         boxa = hypre_BoxArrayArrayBoxArray(boxaa, i);
         hypre_ForBoxI(j, boxa)
         {
            box = hypre_BoxArrayBox(boxa, j);
            hypre_ProjectBox(box, origin, stride);
            hypre_StructMatrixMapDataBox(matrix, box);
         }
      }

      boxaa = hypre_CommInfoRecvRBoxes(comm_info);
      hypre_ForBoxArrayI(i, boxaa)
      {
         boxa = hypre_BoxArrayArrayBoxArray(boxaa, i);
         hypre_ForBoxI(j, boxa)
         {
            box = hypre_BoxArrayBox(boxa, j);
            hypre_ProjectBox(box, origin, stride);
            hypre_StructMatrixMapDataBox(matrix, box);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Create a matrix communication package from comm_info and destroy comm_info.
 *
 * When the domain is coarse, multiple intermediate communication packages are
 * first created, then agglomerated into one.  This is because only a subset of
 * the stencil entries is valid for any given range-space index (for example,
 * linear interpolation in 1D uses the east and west coefficients at F-points
 * and the center coefficient at C-points).  Hence, for a given send or receive
 * box in 'comm_info', the corresponding data box may be different for different
 * stencil entries.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixCreateCommPkg( hypre_StructMatrix *matrix,
                                 hypre_CommInfo     *comm_info,
                                 hypre_CommPkg     **comm_pkg_ptr,
                                 HYPRE_Complex    ***comm_data_ptr )
{
   hypre_CommPkg        *comm_pkg;
   HYPRE_Complex       **comm_data;

   HYPRE_Int             num_values = hypre_StructMatrixNumValues(matrix);

   if (hypre_StructMatrixDomainIsCoarse(matrix))
   {
      hypre_CommInfo       *comm_info_clone;
      hypre_CommPkg       **comm_pkgs;
      hypre_StructStencil  *stencil      = hypre_StructMatrixStencil(matrix);
      HYPRE_Int             stencil_size = hypre_StructStencilSize(stencil);
      HYPRE_Int            *constant     = hypre_StructMatrixConstant(matrix);
      HYPRE_Int            *symm         = hypre_StructMatrixSymmEntries(matrix);
      hypre_Index          *origins, stride;
      HYPRE_Int            *v_to_s, *s_to_v, *order;
      HYPRE_Int            *stencil_spaces, num_spaces;
      HYPRE_Int             i, e, s;

      /* Compute mappings between "values" (v) and "stencil entries" (s).
       * Consider pre-computing this and storing in the matrix itself. */
      v_to_s = hypre_TAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
      s_to_v = hypre_TAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
      for (e = 0, i = 0; e < stencil_size; e++)
      {
         s_to_v[e] = -1;
         if ((symm[e] < 0) && (!constant[e]))  /* this is a stored variable coefficient */
         {
            v_to_s[i] = e;
            s_to_v[e] = i;
            i++;
         }
      }

      /* Get the stencil spaces (each space may induce different data boxes for communication). */
      hypre_StructMatrixGetStSpaces(matrix, 0, &num_spaces, &stencil_spaces, &origins, stride);

      /* Compute communication packages for each stencil space */
      comm_pkgs = hypre_TAlloc(hypre_CommPkg *, num_spaces, HYPRE_MEMORY_HOST);
      comm_data = hypre_TAlloc(HYPRE_Complex *, num_spaces, HYPRE_MEMORY_HOST);
      order = hypre_TAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
      for (s = 0; s < num_spaces; s++)
      {
         /* Set order[i] = -1 to skip values not in this stencil space */
         for (i = 0; i < num_values; i++)
         {
            order[i] = i;
            if (stencil_spaces[v_to_s[i]] != s)
            {
               order[i] = -1;
            }
         }
         hypre_CommInfoClone(comm_info, &comm_info_clone);
         hypre_StructMatrixMapCommInfo(matrix, origins[s], stride, comm_info_clone);
         hypre_CommPkgCreate(comm_info_clone,
                             hypre_StructMatrixDataSpace(matrix),
                             hypre_StructMatrixDataSpace(matrix), num_values, &order, 0,
                             hypre_StructMatrixComm(matrix),
                             hypre_StructMatrixMemoryLocation(matrix),
                             &comm_pkgs[s]);
         comm_data[s] = hypre_StructMatrixVData(matrix);
         hypre_CommInfoDestroy(comm_info_clone);
      }
      hypre_TFree(order, HYPRE_MEMORY_HOST);
      hypre_TFree(stencil_spaces, HYPRE_MEMORY_HOST);
      hypre_TFree(origins, HYPRE_MEMORY_HOST);
      hypre_TFree(v_to_s, HYPRE_MEMORY_HOST);
      hypre_TFree(s_to_v, HYPRE_MEMORY_HOST);

      /* Agglomerate comm_pkgs into one comm_pkg */
      comm_pkg = comm_pkgs[0];
      if (num_spaces > 1)
      {
         hypre_CommPkgAgglomerate(num_spaces, comm_pkgs, &comm_pkg);
         for (s = 0; s < num_spaces; s++)
         {
            hypre_CommPkgDestroy(comm_pkgs[s]);
         }
      }
      hypre_TFree(comm_pkgs, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_IndexRef  stride;

      comm_data = hypre_TAlloc(HYPRE_Complex *, 1, HYPRE_MEMORY_HOST);
      hypre_StructMatrixGetDataMapStride(matrix, &stride);
      hypre_StructMatrixMapCommInfo(matrix, NULL, stride, comm_info);
      hypre_CommPkgCreate(comm_info,
                          hypre_StructMatrixDataSpace(matrix),
                          hypre_StructMatrixDataSpace(matrix), num_values, NULL, 0,
                          hypre_StructMatrixComm(matrix),
                          hypre_StructMatrixMemoryLocation(matrix),
                          &comm_pkg);
      comm_data[0] = hypre_StructMatrixVData(matrix);
   }

   hypre_CommInfoDestroy(comm_info);

   *comm_pkg_ptr  = comm_pkg;
   *comm_data_ptr = comm_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns a pointer to data in `matrix' coresponding to the stencil offset
 * specified by `index'. If index does not exist in the matrix stencil, the NULL
 * pointer is returned.
 *--------------------------------------------------------------------------*/

HYPRE_Complex *
hypre_StructMatrixExtractPointerByIndex( hypre_StructMatrix *matrix,
                                         HYPRE_Int           b,
                                         hypre_Index         index  )
{
   hypre_StructStencil   *stencil;
   HYPRE_Int              entry;

   stencil = hypre_StructMatrixStencil(matrix);
   entry = hypre_StructStencilOffsetEntry( stencil, index );

   if ( entry >= 0 )
   {
      return hypre_StructMatrixBoxData(matrix, b, entry);
   }
   else
   {
      return NULL;   /* error - invalid index */
   }
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixCreate( MPI_Comm             comm,
                          hypre_StructGrid    *grid,
                          hypre_StructStencil *user_stencil )
{
   HYPRE_Int            ndim = hypre_StructGridNDim(grid);
   hypre_StructMatrix  *matrix;
   HYPRE_Int            i;

   matrix = hypre_CTAlloc(hypre_StructMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_StructMatrixComm(matrix)        = comm;
   hypre_StructGridRef(grid, &hypre_StructMatrixGrid(matrix));
   hypre_StructMatrixSetRangeStride(matrix, NULL);                 /* Set default stride */
   hypre_StructMatrixSetDomainStride(matrix, NULL);                /* Set default stride */
   hypre_StructMatrixUserStencil(matrix) =
      hypre_StructStencilRef(user_stencil);
   hypre_StructMatrixConstant(matrix) =
      hypre_CTAlloc(HYPRE_Int, hypre_StructStencilSize(user_stencil), HYPRE_MEMORY_HOST);
   hypre_StructMatrixDataAlloced(matrix) = 1;
   hypre_StructMatrixRefCount(matrix)    = 1;

   /* set defaults */
   hypre_StructMatrixRangeIsCoarse(matrix) = 0;
   hypre_StructMatrixDomainIsCoarse(matrix) = 0;
   hypre_StructMatrixSymmetric(matrix) = 0;
   hypre_StructMatrixConstantCoefficient(matrix) = 0;

   /* RDF TODO: Change to work with default num ghost of zero */
   for (i = 0; i < 2 * ndim; i++)
   {
      hypre_StructMatrixNumGhost(matrix)[i] = hypre_StructGridNumGhost(grid)[i];
      hypre_StructMatrixSymGhost(matrix)[i] = 0;
      hypre_StructMatrixTrnGhost(matrix)[i] = 0;
   }

   hypre_StructMatrixMemoryLocation(matrix) = hypre_HandleMemoryLocation(hypre_handle());

#if defined(HYPRE_MIXED_PRECISION)
   hypre_StructMatrixPrecision(matrix) = HYPRE_OBJECT_PRECISION;
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRef( hypre_StructMatrix *matrix )
{
   hypre_StructMatrixRefCount(matrix) ++;

   return matrix;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixDestroy( hypre_StructMatrix *matrix )
{
   HYPRE_Int  i;

   if (matrix)
   {
      hypre_StructMatrixRefCount(matrix) --;
      if (hypre_StructMatrixRefCount(matrix) == 0)
      {
         if (hypre_StructMatrixDataAlloced(matrix))
         {
            hypre_TFree(hypre_StructMatrixData(matrix), hypre_StructMatrixMemoryLocation(matrix));
         }

         hypre_ForBoxI(i, hypre_StructMatrixDataSpace(matrix))
         {
            hypre_TFree(hypre_StructMatrixDataIndices(matrix)[i], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(hypre_StructMatrixDataIndices(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_StructMatrixConstIndices(matrix), HYPRE_MEMORY_HOST);

         hypre_BoxArrayDestroy(hypre_StructMatrixDataSpace(matrix));
         hypre_BoxArrayDestroy(hypre_StructMatrixDataBoxes(matrix));

         hypre_TFree(hypre_StructMatrixSymmEntries(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_StructMatrixConstant(matrix), HYPRE_MEMORY_HOST);
         hypre_StructStencilDestroy(hypre_StructMatrixUserStencil(matrix));
         hypre_StructStencilDestroy(hypre_StructMatrixStencil(matrix));
         hypre_TFree(hypre_StructMatrixDomBoxnums(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_StructMatrixRanBoxnums(matrix), HYPRE_MEMORY_HOST);
         hypre_StructGridDestroy(hypre_StructMatrixGrid(matrix));
         hypre_StructMatrixForget(matrix);

         hypre_TFree(matrix, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetMemoryLocation( hypre_StructMatrix   *matrix,
                                     HYPRE_MemoryLocation  memory_location )
{
   hypre_StructMatrixMemoryLocation(matrix) = memory_location;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set matrix ran_stride, ran_nboxes, and ran_boxnums.
 * If range_stride == NULL, set default values.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetRangeStride( hypre_StructMatrix *matrix,
                                  hypre_IndexRef      range_stride )
{
   HYPRE_Int  ran_nboxes, *ran_boxnums, ndim = hypre_StructMatrixNDim(matrix);

   if (range_stride != NULL)
   {
      hypre_CopyToIndex(range_stride, ndim, hypre_StructMatrixRanStride(matrix));
   }
   else
   {
      /* set default stride of 1 */
      hypre_SetIndex(hypre_StructMatrixRanStride(matrix), 1);
   }
   hypre_StructGridComputeBoxnums(hypre_StructMatrixGrid(matrix), 0, NULL,
                                  hypre_StructMatrixRanStride(matrix), &ran_nboxes, &ran_boxnums);
   hypre_TFree(hypre_StructMatrixRanBoxnums(matrix), HYPRE_MEMORY_HOST);
   hypre_StructMatrixRanNBoxes(matrix)  = ran_nboxes;
   hypre_StructMatrixRanBoxnums(matrix) = ran_boxnums;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set matrix dom_stride, dom_nboxes, and dom_boxnums.
 * If domain_stride == NULL, set default values.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetDomainStride( hypre_StructMatrix *matrix,
                                   hypre_IndexRef      domain_stride )
{
   HYPRE_Int  dom_nboxes, *dom_boxnums, ndim = hypre_StructMatrixNDim(matrix);

   if (domain_stride != NULL)
   {
      hypre_CopyToIndex(domain_stride, ndim, hypre_StructMatrixDomStride(matrix));
   }
   else
   {
      /* set default stride of 1 */
      hypre_SetIndex(hypre_StructMatrixDomStride(matrix), 1);
   }
   hypre_StructGridComputeBoxnums(hypre_StructMatrixGrid(matrix), 0, NULL,
                                  hypre_StructMatrixDomStride(matrix), &dom_nboxes, &dom_boxnums);
   hypre_TFree(hypre_StructMatrixDomBoxnums(matrix), HYPRE_MEMORY_HOST);
   hypre_StructMatrixDomNBoxes(matrix)  = dom_nboxes;
   hypre_StructMatrixDomBoxnums(matrix) = dom_boxnums;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This computes a matrix data space from a num_ghost array.  If the num_ghost
 * argument is NULL, the matrix num_ghost is used instead.  The routine takes
 * into account additional ghost values needed for symmetric matrices.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixComputeDataSpace( hypre_StructMatrix *matrix,
                                    HYPRE_Int          *num_ghost,
                                    hypre_BoxArray    **data_space_ptr )
{
   HYPRE_Int          ndim      = hypre_StructMatrixNDim(matrix);
   hypre_StructGrid  *grid      = hypre_StructMatrixGrid(matrix);
   HYPRE_Int         *sym_ghost = hypre_StructMatrixSymGhost(matrix);
   HYPRE_Int         *trn_ghost = hypre_StructMatrixTrnGhost(matrix);
   hypre_BoxArray    *data_space;
   hypre_Box         *data_box;
   HYPRE_Int          i, d, d2;

   if (num_ghost == NULL)
   {
      /* Use the matrix num_ghost */
      num_ghost = hypre_StructMatrixNumGhost(matrix);
   }

   /* Add ghost layers and map the data space */
   data_space = hypre_BoxArrayClone(hypre_StructGridBoxes(grid));
   hypre_ForBoxI(i, data_space)
   {
      data_box = hypre_BoxArrayBox(data_space, i);
      for (d = 0; d < ndim; d++)
      {
         d2 = d * 2;
         hypre_BoxIMinD(data_box, d) -= hypre_max(num_ghost[d2] + sym_ghost[d2], trn_ghost[d2]);
         d2 = d * 2 + 1;
         hypre_BoxIMaxD(data_box, d) += hypre_max(num_ghost[d2] + sym_ghost[d2], trn_ghost[d2]);
      }
      hypre_StructMatrixMapDataBox(matrix, data_box);
   }

   *data_space_ptr = data_space;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This grows the current matrix data space (if needed) to support the minimum
 * num_ghost array specified in the argument list.  It is assumed that the
 * current data space already accounts for symmetric or transpose matrices.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGrowDataSpace( hypre_StructMatrix *matrix,
                                 HYPRE_Int          *num_ghost,
                                 hypre_BoxArray    **data_space_ptr )
{
   HYPRE_Int          ndim        = hypre_StructMatrixNDim(matrix);
   hypre_StructGrid  *grid        = hypre_StructMatrixGrid(matrix);
   hypre_BoxArray    *cdata_space = hypre_StructMatrixDataSpace(matrix);
   hypre_BoxArray    *data_space;
   hypre_Box         *data_box;
   HYPRE_Int          i, d, d2;

   /* Compute data space based only on num_ghost */
   data_space = hypre_BoxArrayClone(hypre_StructGridBoxes(grid));
   hypre_ForBoxI(i, data_space)
   {
      data_box = hypre_BoxArrayBox(data_space, i);
      for (d = 0; d < ndim; d++)
      {
         d2 = d * 2;
         hypre_BoxIMinD(data_box, d) -= num_ghost[d2];
         d2 = d * 2 + 1;
         hypre_BoxIMaxD(data_box, d) += num_ghost[d2];
      }
      hypre_StructMatrixMapDataBox(matrix, data_box);
      hypre_BoxGrowByBox(data_box, hypre_BoxArrayBox(cdata_space, i));
   }

   *data_space_ptr = data_space;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Check to see if a resize is needed
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixNeedResize( hypre_StructMatrix *matrix,
                              hypre_BoxArray     *data_space )
{
   hypre_BoxArray  *cdata_space = hypre_StructMatrixDataSpace(matrix);
   HYPRE_Int        need_resize = 0;

   if (cdata_space == NULL)
   {
      /* resize if no current data space (cdata_space) */
      need_resize = 1;
   }
   else if (!hypre_BoxArraysEqual(data_space, cdata_space))
   {
      /* resize if data_space is no the same as cdata_space (note that we check
       * equality here, which we need for StructMatmult()) */
      need_resize = 1;
   }

   return need_resize;
}

/*--------------------------------------------------------------------------
 * This routine takes new data space information and recomputes entries in the
 * matrix that depend on it (e.g., data_indices and data_size).  The routine
 * will also re-allocate the matrix data if their was data to begin with.
 *
 * The boxes in the data_space argument may be larger (but not smaller) than
 * those computed by the routine MatrixComputeDataSpace().
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixResize( hypre_StructMatrix *matrix,
                          hypre_BoxArray     *data_space )
{
   HYPRE_Complex        *old_data         = hypre_StructMatrixData(matrix);
   hypre_BoxArray       *old_data_space   = hypre_StructMatrixDataSpace(matrix);
   hypre_BoxArray       *old_data_boxes   = hypre_StructMatrixDataBoxes(matrix);
   HYPRE_Int             old_data_size    = hypre_StructMatrixDataSize(matrix);
   HYPRE_Int           **old_data_indices = hypre_StructMatrixDataIndices(matrix);

   HYPRE_MemoryLocation  memory_location  = hypre_StructMatrixMemoryLocation(matrix);
   HYPRE_Int             ndim             = hypre_StructMatrixNDim(matrix);
   hypre_StructStencil  *stencil          = hypre_StructMatrixStencil(matrix);
   hypre_Index          *stencil_shape    = hypre_StructStencilShape(stencil);
   HYPRE_Int             stencil_size     = hypre_StructStencilSize(stencil);
   HYPRE_Int            *constant         = hypre_StructMatrixConstant(matrix);
   HYPRE_Int            *symm_entries     = hypre_StructMatrixSymmEntries(matrix);

   HYPRE_Complex        *data;
   hypre_BoxArray       *data_boxes;
   HYPRE_Int             data_size;
   HYPRE_Int           **data_indices;
   HYPRE_Int            *const_indices;

   hypre_Box            *data_box;
   HYPRE_Int             data_box_volume;
   HYPRE_Int             i, j;

   if (hypre_StructMatrixSaveDataSpace(matrix) != NULL)
   {
      /* Call Restore or Forget first */
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Resize has already been called");
      return hypre_error_flag;
   }

   /* Set up data_boxes */
   data_boxes = hypre_BoxArrayClone(data_space);
   hypre_ForBoxI(i, data_boxes)
   {
      data_box = hypre_BoxArrayBox(data_boxes, i);
      hypre_StructMatrixUnMapDataBox(matrix, data_box);
   }

   /* Set up const_indices (constant values at the beginning of data array) */
   const_indices = hypre_StructMatrixConstIndices(matrix);
   if (const_indices == NULL)
   {
      const_indices = hypre_CTAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
      for (j = 0; j < stencil_size; j++)
      {
         if (constant[j])
         {
            if (symm_entries[j] < 0)
            {
               /* set pointers for "stored" coefficients */
               const_indices[j] = j;
            }
            else
            {
               /* set pointers for "symmetric" coefficients */
               const_indices[j] = symm_entries[j];
            }
         }
      }
   }

   /* Set up data_indices and data_size (constant values at the beginning of data array) */
   data_indices = hypre_CTAlloc(HYPRE_Int *, hypre_BoxArraySize(data_space), HYPRE_MEMORY_HOST);
   data_size = stencil_size;
   hypre_ForBoxI(i, data_space)
   {
      data_box = hypre_BoxArrayBox(data_space, i);
      data_box_volume  = hypre_BoxVolume(data_box);

      data_indices[i] = hypre_CTAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);

      for (j = 0; j < stencil_size; j++)
      {
         /* set to const_indices initially */
         data_indices[i][j] = const_indices[j];

         if ( !constant[j] && (symm_entries[j] < 0) )
         {
            /* set pointers for "stored" coefficients */
            data_indices[i][j] = data_size;
            data_size += data_box_volume;
         }
      }
      for (j = 0; j < stencil_size; j++)
      {
         if ( !constant[j] && (symm_entries[j] >= 0) )
         {
            /* set pointers for "symmetric" coefficients */
            data_indices[i][j] = data_indices[i][symm_entries[j]] +
                                 hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
         }
      }
   }

   /* Copy or move old_data to data */
   data = NULL;
   if (old_data != NULL)
   {
      HYPRE_Int   nval = hypre_StructMatrixNumValues(matrix);

      /* TODO (VPM): We should be able to just do hypre_TAlloc */
      data = hypre_CTAlloc(HYPRE_Complex, data_size, memory_location);

      /* Copy constant data values */
      hypre_TMemcpy(data, old_data, HYPRE_Complex, stencil_size,
                    memory_location, memory_location);

      /* Copy the data */
      hypre_StructDataCopy(old_data + stencil_size, old_data_space,
                           data + stencil_size, data_space, ndim, nval);
      if (hypre_StructMatrixDataAlloced(matrix))
      {
         hypre_TFree(old_data, memory_location);
      }
   }

   hypre_StructMatrixData(matrix)         = data;
   hypre_StructMatrixDataSpace(matrix)    = data_space;
   hypre_StructMatrixDataBoxes(matrix)    = data_boxes;
   hypre_StructMatrixDataSize(matrix)     = data_size;
   hypre_StructMatrixDataIndices(matrix)  = data_indices;
   hypre_StructMatrixConstIndices(matrix) = const_indices;
   hypre_StructMatrixVDataOffset(matrix)  = stencil_size;

   /* Clean up and save data */
   if (old_data_space != NULL)
   {
      hypre_BoxArrayDestroy(old_data_boxes);
      hypre_ForBoxI(i, old_data_space)
      {
         hypre_TFree(old_data_indices[i], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(old_data_indices, HYPRE_MEMORY_HOST);

      hypre_StructMatrixSaveData(matrix)      = old_data;
      hypre_StructMatrixSaveDataSpace(matrix) = old_data_space;
      hypre_StructMatrixSaveDataSize(matrix)  = old_data_size;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine restores the old data and data space.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixRestore( hypre_StructMatrix *matrix )
{
   HYPRE_Complex        *old_data        = hypre_StructMatrixData(matrix);
   hypre_BoxArray       *old_data_space  = hypre_StructMatrixDataSpace(matrix);
   HYPRE_Complex        *data            = hypre_StructMatrixSaveData(matrix);
   hypre_BoxArray       *data_space      = hypre_StructMatrixSaveDataSpace(matrix);
   HYPRE_Int             data_size       = hypre_StructMatrixSaveDataSize(matrix);
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(matrix);
   hypre_StructStencil  *stencil         = hypre_StructMatrixStencil(matrix);
   HYPRE_Int             stencil_size    = hypre_StructStencilSize(stencil);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (data_space != NULL)
   {
      HYPRE_Int   ndim = hypre_StructMatrixNDim(matrix);
      HYPRE_Int   nval = hypre_StructMatrixNumValues(matrix);
      HYPRE_Int   i;

      /* Move the data */
      if (hypre_StructMatrixDataAlloced(matrix))
      {
         data = hypre_TAlloc(HYPRE_Complex, data_size, memory_location);
      }

      /* Copy constant data values */
      for (i = 0; i < stencil_size; i++)
      {
         data[i] = old_data[i];
      }
      hypre_StructDataCopy(old_data + stencil_size, old_data_space,
                           data + stencil_size, data_space, ndim, nval);
      hypre_TFree(old_data, memory_location);

      /* Reset certain fields to enable the Resize call below */
      hypre_StructMatrixSaveData(matrix)      = NULL;
      hypre_StructMatrixSaveDataSpace(matrix) = NULL;
      hypre_StructMatrixSaveDataSize(matrix)  = 0;
      hypre_StructMatrixData(matrix)          = NULL;

      /* Set the data space and recompute data_indices, etc. */
      hypre_StructMatrixResize(matrix, data_space);
      hypre_StructMatrixForget(matrix);

      /* Set the data pointer */
      hypre_StructMatrixData(matrix) = data;
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine will clear data needed to do a Restore
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixForget( hypre_StructMatrix *matrix )
{
   hypre_BoxArray  *save_data_space = hypre_StructMatrixSaveDataSpace(matrix);

   if (save_data_space != NULL)
   {
      hypre_BoxArrayDestroy(save_data_space);
      hypre_StructMatrixSaveData(matrix)      = NULL;
      hypre_StructMatrixSaveDataSpace(matrix) = NULL;
      hypre_StructMatrixSaveDataSize(matrix)  = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: This currently assumes that either ran_stride or dom_stride is 1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixInitializeShell( hypre_StructMatrix *matrix )
{
   HYPRE_Int             ndim       = hypre_StructMatrixNDim(matrix);
   hypre_StructGrid     *grid       = hypre_StructMatrixGrid(matrix);
   HYPRE_Int            *constant   = hypre_StructMatrixConstant(matrix);
   hypre_IndexRef        ran_stride = hypre_StructMatrixRanStride(matrix);
   hypre_IndexRef        dom_stride = hypre_StructMatrixDomStride(matrix);
   hypre_IndexRef        periodic   = hypre_StructGridPeriodic(grid);

   hypre_StructStencil  *user_stencil;
   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   HYPRE_Int             stencil_size;
   HYPRE_Int             num_values, num_cvalues;
   HYPRE_Int            *symm_entries;
   HYPRE_Int             domain_is_coarse;
   hypre_BoxArray       *data_space;
   HYPRE_Int            *sym_ghost;
   HYPRE_Int             i, j, d, resize;

   /*-----------------------------------------------------------------------
    * First, check consistency of ran_stride, dom_stride, and symmetric.
    *-----------------------------------------------------------------------*/

   /* domain_is_coarse={0,1,-1,-2} -> coarse grid={range,domain,neither,error} */
   domain_is_coarse = -1;
   for (d = 0; d < ndim; d++)
   {
      if (ran_stride[d] > dom_stride[d])
      {
         if ((domain_is_coarse == 1) || (ran_stride[d] % dom_stride[d] != 0))
         {
            domain_is_coarse = -2;
            break;
         }
         else
         {
            /* Range index space is coarsest */
            domain_is_coarse = 0;
         }
      }
      else if (dom_stride[d] > ran_stride[d])
      {
         if ((domain_is_coarse == 0) || (dom_stride[d] % ran_stride[d] != 0))
         {
            domain_is_coarse = -2;
            break;
         }
         else
         {
            /* Domain index space is coarsest */
            domain_is_coarse = 1;
         }
      }
   }
   if (domain_is_coarse > -1)
   {
      if (domain_is_coarse)
      {
         /* Domain is coarse */
         hypre_StructMatrixDomainIsCoarse(matrix) = 1;
      }
      else
      {
         /* Range is coarse */
         hypre_StructMatrixRangeIsCoarse(matrix) = 1;
      }
      /* Can't have a symmetric stencil for a rectangular matrix */
      hypre_StructMatrixSymmetric(matrix) = 0;
   }
   if (domain_is_coarse == -2)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid matrix domain and range strides");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------------
    * Now check consistency of ran_stride, dom_stride, and periodic.
    *-----------------------------------------------------------------------*/

   for (d = 0; d < ndim; d++)
   {
      if ( (periodic[d] % ran_stride[d]) || (periodic[d] % dom_stride[d]) )
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                           "Periodicity must be an integral multiple of the matrix domain and range strides");
         return hypre_error_flag;
      }
   }

   /*-----------------------------------------------------------------------
    * Set up stencil, num_values, and num_cvalues:
    *
    * If the matrix is symmetric, then the stencil is a "symmetrized"
    * version of the user's stencil.  If the matrix is not symmetric,
    * then the stencil is the same as the user's stencil.
    *
    * The `symm_entries' array is used to determine what data is explicitely
    * stored (symm_entries[i] < 0) and what data is not explicitely stored
    * (symm_entries[i] >= 0), but is instead stored as the transpose
    * coefficient at a neighboring grid point.
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixStencil(matrix) == NULL)
   {
      user_stencil = hypre_StructMatrixUserStencil(matrix);
      symm_entries = NULL;

      if (hypre_StructMatrixSymmetric(matrix))
      {
         /* store only symmetric stencil entry data */
         hypre_StructStencilSymmetrize(user_stencil, &stencil, &symm_entries);
         stencil_size = hypre_StructStencilSize(stencil);
         constant = hypre_TReAlloc(constant, HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
         for (i = 0; i < stencil_size; i++)
         {
            if (symm_entries[i] >= 0)
            {
               constant[i] = constant[symm_entries[i]];
            }
         }
      }
      else
      {
         /* store all stencil entry data */
         stencil = hypre_StructStencilRef(user_stencil);
         stencil_size = hypre_StructStencilSize(stencil);
         symm_entries = hypre_TAlloc(HYPRE_Int, stencil_size, HYPRE_MEMORY_HOST);
         for (i = 0; i < stencil_size; i++)
         {
            symm_entries[i] = -1;
         }
      }

      /* Compute number of stored constant and variables coeffs */
      num_values = 0;
      num_cvalues = 0;
      for (j = 0; j < stencil_size; j++)
      {
         if (symm_entries[j] < 0)
         {
            if (constant[j])
            {
               num_cvalues++;
            }
            else
            {
               num_values++;
            }
         }
      }

      hypre_StructMatrixStencil(matrix)     = stencil;
      hypre_StructMatrixConstant(matrix)    = constant;
      hypre_StructMatrixSymmEntries(matrix) = symm_entries;
      hypre_StructMatrixNumValues(matrix)   = num_values;
      hypre_StructMatrixNumCValues(matrix)  = num_cvalues;
   }

   /*-----------------------------------------------------------------------
    * Compute the needed additional ghost-layer size for symmetric storage
    * (square matrices only).  All stencil coeffs are to be available at each
    * point in the grid, including the user-specified ghost layer.
    *-----------------------------------------------------------------------*/

   sym_ghost     = hypre_StructMatrixSymGhost(matrix);
   stencil       = hypre_StructMatrixStencil(matrix);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);
   symm_entries  = hypre_StructMatrixSymmEntries(matrix);

   /* Initialize additional ghost size */
   for (d = 0; d < 2 * ndim; d++)
   {
      sym_ghost[d] = 0;
   }

   /* Add ghost layers for symmetric storage */
   if (hypre_StructMatrixSymmetric(matrix))
   {
      for (i = 0; i < stencil_size; i++)
      {
         if (symm_entries[i] >= 0)
         {
            for (d = 0; d < ndim; d++)
            {
               j = hypre_IndexD(stencil_shape[i], d);
               sym_ghost[2 * d]     = hypre_max(sym_ghost[2 * d],    -j);
               sym_ghost[2 * d + 1] = hypre_max(sym_ghost[2 * d + 1], j);
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Compute the ghost layers needed for storing the transpose
    *-----------------------------------------------------------------------*/

   hypre_StructMatrixSetTranspose(matrix, hypre_StructMatrixTranspose(matrix), &resize);

   /*-----------------------------------------------------------------------
    * Compute global size
    *-----------------------------------------------------------------------*/

   hypre_StructMatrixGlobalSize(matrix) = hypre_StructGridGlobalSize(grid) * stencil_size;

   /*-----------------------------------------------------------------------
    * Set up information related to the data space and data storage
    *-----------------------------------------------------------------------*/

   if (hypre_StructMatrixDataSpace(matrix) == NULL)
   {
      hypre_StructMatrixComputeDataSpace(matrix, NULL, &data_space);
      hypre_StructMatrixResize(matrix, data_space);
      hypre_StructMatrixForget(matrix);
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixInitializeData( hypre_StructMatrix *matrix,
                                  HYPRE_Int           zero_init,
                                  HYPRE_Complex      *data   )
{
   if (data == NULL)
   {
      if (zero_init)
      {
         data = hypre_CTAlloc(HYPRE_Complex,
                              hypre_StructMatrixDataSize(matrix),
                              hypre_StructMatrixMemoryLocation(matrix));
      }
      else
      {
         data = hypre_TAlloc(HYPRE_Complex,
                             hypre_StructMatrixDataSize(matrix),
                             hypre_StructMatrixMemoryLocation(matrix));
      }
      hypre_StructMatrixDataAlloced(matrix) = 1;
   }
   else
   {
      hypre_StructMatrixDataAlloced(matrix) = 0;
   }
   hypre_StructMatrixData(matrix) = data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixInitialize( hypre_StructMatrix *matrix )
{
   hypre_StructMatrixInitializeShell(matrix);
   hypre_StructMatrixInitializeData(matrix, 1, NULL);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *
 * Should not be called to set a constant-coefficient part of the matrix.
 * Call hypre_StructMatrixSetConstantValues instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetValues( hypre_StructMatrix *matrix,
                             hypre_Index         grid_index,
                             HYPRE_Int           num_stencil_indices,
                             HYPRE_Int          *stencil_indices,
                             HYPRE_Complex      *values,
                             HYPRE_Int           action,
                             HYPRE_Int           boxnum,
                             HYPRE_Int           outside )
{
   HYPRE_Int             *constant        = hypre_StructMatrixConstant(matrix);
   HYPRE_Int             *symm_entries    = hypre_StructMatrixSymmEntries(matrix);

   hypre_BoxArray        *grid_boxes;
   hypre_Box             *grid_box;
   hypre_BoxArray        *data_boxes;
   hypre_Box             *data_box;
   HYPRE_Complex         *matp;

   HYPRE_Int              i, j, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataBoxes(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }
   data_boxes = hypre_StructMatrixDataBoxes(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_boxes, i);

      if (hypre_IndexInBox(grid_index, grid_box))
      {
         hypre_StructMatrixMapDataIndex(matrix, grid_index);

         for (s = 0; s < num_stencil_indices; s++)
         {
            j = stencil_indices[s];

            /* call SetConstantValues instead */
            if (constant[j])
            {
               hypre_error(HYPRE_ERROR_GENERIC);
            }

            /* only set stored stencil values */
            if (symm_entries[j] < 0)
            {
               matp = hypre_StructMatrixBoxDataValue(matrix, i, j, data_box, grid_index);

#if defined(HYPRE_USING_GPU)
               HYPRE_MemoryLocation memory_location = hypre_StructMatrixMemoryLocation(matrix);

               if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
               {
                  if (action > 0)
                  {
#define DEVICE_VAR is_device_ptr(matp, values)
                     hypre_LoopBegin(1, k)
                     {
                        *matp += values[s];
                     }
                     hypre_LoopEnd()
#undef DEVICE_VAR
                  }
                  else if (action > -1)
                  {
                     hypre_TMemcpy(matp, values + s, HYPRE_Complex, 1,
                                   memory_location, memory_location);
                  }
                  else /* action < 0 */
                  {
                     hypre_TMemcpy(values + s, matp, HYPRE_Complex, 1,
                                   memory_location, memory_location);
                     if (action == -2)
                     {
#define DEVICE_VAR is_device_ptr(matp, values)
                        hypre_LoopBegin(1, k)
                        {
                           *matp = 0.0;
                        }
                        hypre_LoopEnd()
#undef DEVICE_VAR
                     }
                  }
               }
               else
#endif
               {
                  if (action > 0)
                  {
                     *matp += values[s];
                  }
                  else if (action > -1)
                  {
                     *matp = values[s];
                  }
                  else /* action < 0 */
                  {
                     values[s] = *matp;
                     if (action == -2)
                     {
                        *matp = 0;
                     }
                  }
               } /* if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE) */
            } /* if (symm_entries[j] < 0) */
         } /* for (s = 0; s < num_stencil_indices; s++) */
      } /* if (hypre_IndexInBox(grid_index, grid_box)) */
   } /* for (i = istart; i < istop; i++) */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *
 * Should not be called to set a constant-coefficient part of the matrix.
 * Call hypre_StructMatrixSetConstantValues instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetBoxValues( hypre_StructMatrix *matrix,
                                hypre_Box          *set_box,
                                hypre_Box          *value_box,
                                HYPRE_Int           num_stencil_indices,
                                HYPRE_Int          *stencil_indices,
                                HYPRE_Complex      *values,
                                HYPRE_Int           action,
                                HYPRE_Int           boxnum,
                                HYPRE_Int           outside )
{
   HYPRE_Int            ndim         = hypre_StructMatrixNDim(matrix);
   HYPRE_Int           *constant     = hypre_StructMatrixConstant(matrix);
   HYPRE_Int           *symm_entries = hypre_StructMatrixSymmEntries(matrix);
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_Box           *int_box;

   hypre_BoxArray      *data_space;
   hypre_Box           *data_box;
   hypre_IndexRef       data_start;
   hypre_Index          data_stride;
   HYPRE_Complex       *datap;

   hypre_Box           *dval_box;
   hypre_Index          dval_start;
   hypre_Index          dval_stride;
   HYPRE_Int            dvali;

   hypre_Index          loop_size;
   HYPRE_Int            i, j, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataBoxes(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }
   data_space = hypre_StructMatrixDataSpace(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(data_stride, 1);

   int_box = hypre_BoxCreate(ndim);
   dval_box = hypre_BoxClone(value_box);
   hypre_StructMatrixMapDataBox(matrix, dval_box);
   hypre_BoxIMinD(dval_box, 0) *= num_stencil_indices;
   hypre_BoxIMaxD(dval_box, 0) *= num_stencil_indices;
   hypre_BoxIMaxD(dval_box, 0) += num_stencil_indices - 1;
   hypre_SetIndex(dval_stride, 1);
   hypre_IndexD(dval_stride, 0) = num_stencil_indices;

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_IntersectBoxes(set_box, grid_box, int_box);

      /* if there was an intersection */
      if (hypre_BoxVolume(int_box))
      {
         hypre_StructMatrixMapDataBox(matrix, int_box);

         data_start = hypre_BoxIMin(int_box);
         hypre_CopyIndex(data_start, dval_start);
         hypre_IndexD(dval_start, 0) *= num_stencil_indices;

         for (s = 0; s < num_stencil_indices; s++)
         {
            j = stencil_indices[s];

            /* only set stored stencil values */
            if (symm_entries[j] < 0)
            {
               datap = hypre_StructMatrixBoxData(matrix, i, j);

               /* TODO: this is not the most efficient way of acessing constant entries.
                        Should use SetConstantValues instead. */
               if (constant[j])
               {
                  dvali = hypre_BoxIndexRank(dval_box, dval_start);

                  if (action > 0)
                  {
                     *datap += values[dvali];
                  }
                  else if (action > -1)
                  {
                     *datap = values[dvali];
                  }
                  else
                  {
                     hypre_BoxGetSize(int_box, loop_size);
                     hypre_BoxLoop1Begin(ndim, loop_size,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        values[dvali] = *datap;
                     }
                     hypre_BoxLoop1End(dvali);

                     if (action == -2)
                     {
                        *datap = 0;
                     }
                  }
               }
               else
               {
                  hypre_BoxGetSize(int_box, loop_size);

                  if (action > 0)
                  {
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        datap[datai] += values[dvali];
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
                  else if (action > -1)
                  {
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        datap[datai] = values[dvali];
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
                  else
                  {
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        values[dvali] = datap[datai];
                        if (action == -2)
                        {
                           datap[datai] = 0;
                        }
                     }
                     hypre_BoxLoop2End(datai, dvali);
                  }
               } /* end if (constant) */
            } /* end if (symm_entries) */

            hypre_IndexD(dval_start, 0) ++;
         }
      }
   }

   hypre_BoxDestroy(int_box);
   hypre_BoxDestroy(dval_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *
 * Should be called to set a constant-coefficient part of the matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetConstantValues( hypre_StructMatrix *matrix,
                                     HYPRE_Int           num_stencil_indices,
                                     HYPRE_Int          *stencil_indices,
                                     HYPRE_Complex      *values,
                                     HYPRE_Int           action )
{
   HYPRE_Int           *constant     = hypre_StructMatrixConstant(matrix);
   HYPRE_Complex       *matp;
   HYPRE_Int            j, s;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   for (s = 0; s < num_stencil_indices; s++)
   {
      j = stencil_indices[s];

      matp = hypre_StructMatrixConstData(matrix, j);
      if (!constant[j])
      {
         hypre_error(HYPRE_ERROR_GENERIC);
      }

      if (action > 0)
      {
         *matp += values[s];
      }
      else if (action > -1)
      {
         *matp = values[s];
      }
      else /* action < 0 */
      {
         values[s] = *matp;
         if (action == -2)
         {
            *matp = 0;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearValues( hypre_StructMatrix *matrix,
                               hypre_Index         grid_index,
                               HYPRE_Int           num_stencil_indices,
                               HYPRE_Int          *stencil_indices,
                               HYPRE_Int           boxnum,
                               HYPRE_Int           outside )
{
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_BoxArray      *data_boxes;
   hypre_Box           *data_box;

   HYPRE_Complex       *matp;

   HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataBoxes(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }
   data_boxes = hypre_StructMatrixDataBoxes(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Clear the matrix coefficients
    *-----------------------------------------------------------------------*/

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_boxes, i);

      if (hypre_IndexInBox(grid_index, grid_box))
      {
         hypre_StructMatrixMapDataIndex(matrix, grid_index);

         for (s = 0; s < num_stencil_indices; s++)
         {
            matp = hypre_StructMatrixBoxData(matrix, i, stencil_indices[s]) +
                   hypre_BoxIndexRank(data_box, grid_index);
            *matp = 0.0;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearBoxValues( hypre_StructMatrix *matrix,
                                  hypre_Box          *clear_box,
                                  HYPRE_Int           num_stencil_indices,
                                  HYPRE_Int          *stencil_indices,
                                  HYPRE_Int           boxnum,
                                  HYPRE_Int           outside )
{
   HYPRE_Int            ndim = hypre_StructMatrixNDim(matrix);
   hypre_BoxArray      *grid_boxes;
   hypre_Box           *grid_box;
   hypre_Box           *int_box;

   HYPRE_Int           *symm_entries;
   hypre_BoxArray      *data_space;
   hypre_Box           *data_box;
   hypre_IndexRef       data_start;
   hypre_Index          data_stride;
   HYPRE_Complex       *datap;

   hypre_Index          loop_size;

   HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = hypre_StructMatrixDataBoxes(matrix);
   }
   else
   {
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   }
   data_space = hypre_StructMatrixDataSpace(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Clear the matrix coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(data_stride, 1);

   symm_entries = hypre_StructMatrixSymmEntries(matrix);

   int_box = hypre_BoxCreate(ndim);

   for (i = istart; i < istop; i++)
   {
      grid_box = hypre_BoxArrayBox(grid_boxes, i);
      data_box = hypre_BoxArrayBox(data_space, i);

      hypre_IntersectBoxes(clear_box, grid_box, int_box);

      /* if there was an intersection */
      if (hypre_BoxVolume(int_box))
      {
         hypre_StructMatrixMapDataBox(matrix, int_box);

         data_start = hypre_BoxIMin(int_box);

         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only clear stencil entries that are explicitly stored */
            if (symm_entries[stencil_indices[s]] < 0)
            {
               datap = hypre_StructMatrixBoxData(matrix, i, stencil_indices[s]);

               hypre_BoxGetSize(int_box, loop_size);

               hypre_BoxLoop1Begin(ndim, loop_size,
                                   data_box, data_start, data_stride, datai);
               {
                  datap[datai] = 0.0;
               }
               hypre_BoxLoop1End(datai);
            }
         }
      }
   }

   hypre_BoxDestroy(int_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixAssemble( hypre_StructMatrix *matrix )
{
   HYPRE_Int num_values = hypre_StructMatrixNumValues(matrix);

   /*-----------------------------------------------------------------------
    * Update the ghost data
    * This takes care of the communication needs of all known functions
    * referencing the matrix.
    *-----------------------------------------------------------------------*/

   if (num_values > 0)
   {
      HYPRE_Int          ndim      = hypre_StructMatrixNDim(matrix);
      HYPRE_Int         *num_ghost = hypre_StructMatrixNumGhost(matrix);
      HYPRE_Int         *sym_ghost = hypre_StructMatrixSymGhost(matrix);
      HYPRE_Int         *trn_ghost = hypre_StructMatrixTrnGhost(matrix);
      hypre_StructGrid  *grid      = hypre_StructMatrixGrid(matrix);

      hypre_CommInfo    *comm_info;
      hypre_CommPkg     *comm_pkg;
      hypre_CommHandle  *comm_handle;
      HYPRE_Complex    **comm_data;
      hypre_IndexRef     fstride;
      HYPRE_Int          i, tot_num_ghost[2 * HYPRE_MAXDIM];

      /* RDF TODO: Use CommStencil to do communication; don't forget to account
       * for symmetric stencil entries */
      for (i = 0; i < 2 * ndim; i++)
      {
         tot_num_ghost[i] = hypre_max(num_ghost[i] + sym_ghost[i], trn_ghost[i]);
      }

      hypre_StructMatrixGetFStride(matrix, &fstride);
      hypre_CreateCommInfoFromNumGhost(grid, fstride, tot_num_ghost, &comm_info);
      hypre_StructMatrixCreateCommPkg(matrix, comm_info, &comm_pkg, &comm_data);

      hypre_StructCommunicationInitialize(comm_pkg, comm_data, comm_data, 0, 0, &comm_handle);
      hypre_StructCommunicationFinalize(comm_handle);
      hypre_CommPkgDestroy(comm_pkg);
      hypre_TFree(comm_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nentries - number of stencil entries
 * entries  - array of stencil entries
 *
 * The following four possibilites are supported for backward compatibility:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * - anything else                       (constant_coefficient==3)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetConstantEntries( hypre_StructMatrix *matrix,
                                      HYPRE_Int           nentries,
                                      HYPRE_Int          *entries )
{
   hypre_StructStencil *stencil            = hypre_StructMatrixUserStencil(matrix);
   HYPRE_Int           *constant           = hypre_StructMatrixConstant(matrix);
   HYPRE_Int            stencil_size       = hypre_StructStencilSize(stencil);
   HYPRE_Int            stencil_diag_entry = hypre_StructStencilDiagEntry(stencil);

   HYPRE_Int            i, j;
   HYPRE_Int            nconst;
   HYPRE_Int            constant_coefficient;

   /* By counting the nonzeros in constant, and by checking whether its diagonal
      entry is nonzero, we can distinguish between the three legal values of
      constant_coefficient, and detect input errors.  We do not need to treat
      duplicates in 'entries' as an error condition. */

   for (i = 0; i < nentries; i++)
   {
      constant[entries[i]] = 1;
   }
   nconst = 0;
   for (j = 0; j < stencil_size; j++)
   {
      nconst += constant[j];
   }

   if (nconst == 0)
   {
      constant_coefficient = 0;
   }
   else if (nconst >= stencil_size)
   {
      constant_coefficient = 1;
   }
   else if ( (constant[stencil_diag_entry] == 0) && (nconst == (stencil_size - 1)) )
   {
      constant_coefficient = 2;
   }
   else
   {
      constant_coefficient = 3;
      /* TODO: Fix this for rectangular matrices */
      //hypre_error(HYPRE_ERROR_GENERIC);
   }

   hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine may be called at any time.  The boolean 'resize' is returned to
 * indicate whether a MatrixResize() is needed.  Because the matrix stencil is
 * required to compute the correct ghost layer size for the transpose != 0 case,
 * this routine is also called in the MatrixInitialize() routine.
 *
 * Use this routine exclusively to set hypre_StructMatrixTranspose().
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetTranspose( hypre_StructMatrix *matrix,
                                HYPRE_Int           transpose,
                                HYPRE_Int          *resize )
{
   HYPRE_Int   ndim = hypre_StructMatrixNDim(matrix);
   HYPRE_Int  *trn_ghost = hypre_StructMatrixTrnGhost(matrix);
   HYPRE_Int   lghost, rghost, d, e;

   *resize = 0;
   if (hypre_StructMatrixTranspose(matrix) != transpose)
   {
      for (d = 0; d < ndim; d++)
      {
         lghost = 0;
         rghost = 0;
         if ((transpose) && (hypre_StructMatrixStencil(matrix) != NULL))
         {
            hypre_StructStencil  *stencil       = hypre_StructMatrixStencil(matrix);
            hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);
            HYPRE_Int             stencil_size  = hypre_StructStencilSize(stencil);

            for (e = 0; e < stencil_size; e++)
            {
               lghost = hypre_max(lghost, -hypre_IndexD(stencil_shape[e], d));
               rghost = hypre_max(rghost,  hypre_IndexD(stencil_shape[e], d));
            }
         }
         if (trn_ghost[2 * d] != lghost)
         {
            trn_ghost[2 * d] = lghost;
            *resize = 1;
         }
         if (trn_ghost[2 * d + 1] != rghost)
         {
            trn_ghost[2 * d + 1] = rghost;
            *resize = 1;
         }
      }

      hypre_StructMatrixTranspose(matrix) = transpose;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine may be called at any time.  The boolean 'resize' is returned to
 * indicate whether a MatrixResize() is needed.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetNumGhost( hypre_StructMatrix *matrix,
                               HYPRE_Int          *num_ghost,
                               HYPRE_Int          *resize )
{
   HYPRE_Int  i, ndim = hypre_StructMatrixNDim(matrix);

   *resize = 0;
   for (i = 0; i < 2 * ndim; i++)
   {
      if (hypre_StructMatrixNumGhost(matrix)[i] != num_ghost[i])
      {
         hypre_StructMatrixNumGhost(matrix)[i] = num_ghost[i];
         *resize = 1;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixSetGhost( hypre_StructMatrix *matrix,
                            HYPRE_Int           ghost,
                            HYPRE_Int          *resize )
{
   HYPRE_Int  i, ndim = hypre_StructMatrixNDim(matrix);
   HYPRE_Int  num_ghost[2 * HYPRE_MAXDIM];

   for (i = 0; i < 2 * ndim; i++)
   {
      num_ghost[i] = ghost;
   }
   hypre_StructMatrixSetNumGhost(matrix, num_ghost, resize);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearGhostValues( hypre_StructMatrix *matrix )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(matrix);

   HYPRE_Complex        *mp;

   hypre_StructStencil  *stencil;
   HYPRE_Int            *constant;
   HYPRE_Int            *symm_entries;
   hypre_BoxArray       *grid_boxes;
   hypre_BoxArray       *data_space;
   hypre_Box            *data_box;
   hypre_Box            *grid_data_box;
   hypre_BoxArray       *diff_boxes;
   hypre_Box            *diff_box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           ustride;

   HYPRE_Int             i, j, s;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   hypre_SetIndex(ustride, 1);

   grid_data_box = hypre_BoxCreate(ndim);

   stencil = hypre_StructMatrixStencil(matrix);
   constant = hypre_StructMatrixConstant(matrix);
   symm_entries = hypre_StructMatrixSymmEntries(matrix);
   grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(matrix));
   //data_space = hypre_StructMatrixDataBoxes(matrix);
   data_space = hypre_StructMatrixDataSpace(matrix);
   diff_boxes = hypre_BoxArrayCreate(0, ndim);
   hypre_ForBoxI(i, grid_boxes)
   {
      hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, i), grid_data_box);
      data_box = hypre_BoxArrayBox(data_space, i);
      hypre_BoxArraySetSize(diff_boxes, 0);
      hypre_StructMatrixMapDataBox(matrix, grid_data_box);
      hypre_SubtractBoxes(data_box, grid_data_box, diff_boxes);

      for (s = 0; s < hypre_StructStencilSize(stencil); s++)
      {
         /* only clear stencil entries that are explicitly stored
            and those regarding variable coefficients */
         if ((symm_entries[s] < 0) && !(constant[s]))
         {
            mp = hypre_StructMatrixBoxData(matrix, i, s);
            hypre_ForBoxI(j, diff_boxes)
            {
               diff_box = hypre_BoxArrayBox(diff_boxes, j);
               start = hypre_BoxIMin(diff_box);

               hypre_BoxGetSize(diff_box, loop_size);

               hypre_BoxLoop1Begin(ndim, loop_size,
                                   data_box, start, ustride, mi);
               {
                  mp[mi] = 0.0;
               }
               hypre_BoxLoop1End(mi);
            }
         }
      }
   }
   hypre_BoxArrayDestroy(diff_boxes);
   hypre_BoxDestroy(grid_data_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructMatrixPrintData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPrintData( FILE               *file,
                             hypre_StructMatrix *matrix,
                             HYPRE_Int           all )
{
   HYPRE_Int             ndim            = hypre_StructMatrixNDim(matrix);
   HYPRE_Int             num_values      = hypre_StructMatrixNumValues(matrix);
   HYPRE_Int             num_cvalues     = hypre_StructMatrixNumCValues(matrix);
   hypre_StructGrid     *grid            = hypre_StructMatrixGrid(matrix);
   hypre_StructStencil  *stencil         = hypre_StructMatrixStencil(matrix);
   HYPRE_Int             stencil_size    = hypre_StructStencilSize(stencil);
   HYPRE_Int            *symm_entries    = hypre_StructMatrixSymmEntries(matrix);
   hypre_BoxArray       *data_space      = hypre_StructMatrixDataSpace(matrix);
   HYPRE_Int             data_size       = hypre_StructMatrixDataSize(matrix);
   HYPRE_Complex        *data            = hypre_StructMatrixData(matrix);
   HYPRE_Int            *const_indices   = hypre_StructMatrixConstIndices(matrix);
   HYPRE_MemoryLocation  memory_location = hypre_StructMatrixMemoryLocation(matrix);

   HYPRE_Int             i, ci, vi;
   HYPRE_Int            *value_ids;
   HYPRE_Int            *cvalue_ids;

   hypre_BoxArray       *boxes;
   hypre_Box            *box;
   hypre_Index           data_origin;
   HYPRE_Complex        *h_data;
   HYPRE_Complex        *h_vdata;
   HYPRE_Complex         value;

   /* Allocate/Point to data on the host memory */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      h_data = hypre_TAlloc(HYPRE_Complex, data_size, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_data, data, HYPRE_Complex, data_size,
                    HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_data = data;
   }

   /* This assumes that zero maps to/from zero in the data space */
   hypre_SetIndex(data_origin, 0);

   /* Build constant and variable coefficient stencil indices */
   value_ids  = hypre_TAlloc(HYPRE_Int, num_values, HYPRE_MEMORY_HOST);
   cvalue_ids = hypre_TAlloc(HYPRE_Int, num_cvalues, HYPRE_MEMORY_HOST);
   vi = ci = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (symm_entries[i] < 0)
      {
         if (hypre_StructMatrixConstEntry(matrix, i))
         {
            cvalue_ids[ci++] = i;
         }
         else
         {
            value_ids[vi++] = i;
         }
      }
   }

   /* Print ghost data (all) also or only real data? */
   if (all)
   {
      boxes = data_space;
   }
   else
   {
      boxes = hypre_BoxArrayClone(hypre_StructGridBoxes(grid));
      hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);
         hypre_StructMatrixMapDataBox(matrix, box);
      }
   }

   /* Print constant data to file */
   hypre_fprintf(file, "\nConstant Data:\n");
   /* RDF: It looks like the following 'if' handles matrices with no data */
   if (hypre_StructMatrixDataSize(matrix) > 0)
   {
      hypre_fprintf(file, "%d\n", num_cvalues);
      for (ci = 0; ci < num_cvalues; ci++)
      {
         i = cvalue_ids[ci];
         value = h_data[const_indices[i]];
#ifdef HYPRE_COMPLEX
         hypre_fprintf(file, "*: (*; %d) %.14e, %.14e\n",
                       i, hypre_creal(value), hypre_cimag(value));
#else
         hypre_fprintf(file, "*: (*; %d) %.14e\n", i, value);
#endif
      }
   }

   /* Print variable data to file */
   hypre_fprintf(file, "\nVariable Data:\n");
   h_vdata = h_data + hypre_StructMatrixVDataOffset(matrix);
   /* RDF: It looks like the following 'if' handles matrices with no data */
   if (hypre_StructMatrixDataSize(matrix) > 0)
   {
      hypre_PrintBoxArrayData(file, ndim, boxes, data_space, num_values, value_ids, h_vdata);
   }

   /*----------------------------------------
    * Clean up
    *----------------------------------------*/

   hypre_TFree(value_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(cvalue_ids, HYPRE_MEMORY_HOST);
   if (!all)
   {
      hypre_BoxArrayDestroy(boxes);
   }
   if (h_data != data)
   {
      hypre_TFree(h_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixReadData( FILE               *file,
                            hypre_StructMatrix *matrix )
{
   HYPRE_Int             ndim  = hypre_StructMatrixNDim(matrix);
   hypre_StructGrid     *grid  = hypre_StructMatrixGrid(matrix);
   hypre_BoxArray       *boxes = hypre_StructGridBoxes(grid);

   hypre_Box            *box;
   HYPRE_Int             num_values, num_cvalues;
   HYPRE_Complex        *values, *cvalues, value;
#ifdef HYPRE_COMPLEX
   HYPRE_Complex         rvalue, ivalue;
#endif
   HYPRE_Int            *value_ids, *cvalue_ids;
   HYPRE_Int             ci, i, vi;


   /* Read constant data from file */
   hypre_fscanf(file, "\nConstant Data:\n");
   hypre_fscanf(file, "%d\n", &num_cvalues);
   cvalue_ids = hypre_TAlloc(HYPRE_Int, num_cvalues, HYPRE_MEMORY_HOST);
   cvalues    = hypre_TAlloc(HYPRE_Complex, num_cvalues, HYPRE_MEMORY_HOST);
   for (ci = 0; ci < num_cvalues; ci++)
   {
#ifdef HYPRE_COMPLEX
      hypre_fscanf(file, "*: (*; %d) %le, %le\n", &i, &rvalue, &ivalue);
      value = rvalue + I * ivalue;
#else
      hypre_fscanf(file, "*: (*; %d) %le\n", &i, &value);
#endif
      cvalue_ids[ci] = i;
      cvalues[ci] = value;
   }
   HYPRE_StructMatrixSetConstantEntries(matrix, num_cvalues, cvalue_ids);

   /* Read variable data from file */
   hypre_fscanf(file, "\nVariable Data:\n");
   hypre_ReadBoxArrayData(file, ndim, boxes, &num_values, &value_ids, &values);

   /* Set matrix values */
   HYPRE_StructMatrixInitialize(matrix);
   HYPRE_StructMatrixSetConstantValues(matrix, num_cvalues, cvalue_ids, cvalues);
   vi = 0;
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      HYPRE_StructMatrixSetBoxValues(matrix, hypre_BoxIMin(box), hypre_BoxIMax(box),
                                     num_values, value_ids, &values[vi]);
      vi += num_values * hypre_BoxVolume(box);
   }

   /* Clean up */
   hypre_TFree(cvalue_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(cvalues, HYPRE_MEMORY_HOST);
   hypre_TFree(value_ids, HYPRE_MEMORY_HOST);
   hypre_TFree(values, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixPrint( const char         *filename,
                         hypre_StructMatrix *matrix,
                         HYPRE_Int           all )
{
   HYPRE_Int             ndim          = hypre_StructMatrixNDim(matrix);
   hypre_StructStencil  *stencil       = hypre_StructMatrixStencil(matrix);

   FILE                 *file;
   char                  new_filename[255];
   HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(hypre_StructMatrixComm(matrix), &myid);

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   hypre_fprintf(file, "StructMatrix\n");
   hypre_fprintf(file, "\nSymmetric: %d\n", hypre_StructMatrixSymmetric(matrix));
   hypre_fprintf(file, "\nConstantCoefficient: %d\n",
                 hypre_StructMatrixConstantCoefficient(matrix));

   /* print grid info */
   hypre_fprintf(file, "\nGrid:\n");
   hypre_StructGridPrint(file, hypre_StructMatrixGrid(matrix));

   /* print ran_stride and dom_stride info */
   hypre_fprintf(file, "\nRange Stride: ");
   hypre_IndexPrint(file, ndim, hypre_StructMatrixRanStride(matrix));

   hypre_fprintf(file, "\nDomain Stride: ");
   hypre_IndexPrint(file, ndim, hypre_StructMatrixDomStride(matrix));

   /* print stencil info */
   hypre_fprintf(file, "\n\nStencil:\n");
   hypre_StructStencilPrint(file, stencil);

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   hypre_StructMatrixPrintData(file, matrix, all);

   /*----------------------------------------
    * Clean up
    *----------------------------------------*/

   fflush(file);
   fclose(file);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: This will not read files with ghost values in them.
 * RDF TODO: Fix this to use new matrix structure
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_StructMatrixRead( MPI_Comm    comm,
                        const char *filename,
                        HYPRE_Int  *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];

   hypre_StructMatrix   *matrix;

   hypre_StructGrid     *grid;
   HYPRE_Int             ndim;

   hypre_StructStencil  *stencil;
   HYPRE_Int             symmetric;
   HYPRE_Int             constant_coefficient;
   hypre_Index           ran_stride;
   hypre_Index           dom_stride;

   HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   hypre_MPI_Comm_rank(comm, &myid );

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   hypre_fscanf(file, "StructMatrix\n");
   hypre_fscanf(file, "\nSymmetric: %d\n", &symmetric);
   hypre_fscanf(file, "\nConstantCoefficient: %d\n", &constant_coefficient);

   /* read grid info */
   hypre_fscanf(file, "\nGrid:\n");
   hypre_StructGridRead(comm, file, &grid);
   ndim = hypre_StructGridNDim(grid);

   /* read ran_stride and dom_stride info */
   hypre_fscanf(file, "\nRange Stride: ");
   hypre_IndexRead(file, ndim, ran_stride);

   hypre_fscanf(file, "\nDomain Stride: ");
   hypre_IndexRead(file, ndim, dom_stride);

   /* read stencil info */
   hypre_fscanf(file, "\n\nStencil:\n");
   hypre_StructStencilRead(file, ndim, &stencil);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   HYPRE_StructMatrixCreate(comm, grid, stencil, &matrix);
   HYPRE_StructMatrixSetNumGhost(matrix, num_ghost);  // RDF Is this needed?
   HYPRE_StructMatrixSetSymmetric(matrix, symmetric);
   HYPRE_StructMatrixSetRangeStride(matrix, ran_stride);
   HYPRE_StructMatrixSetDomainStride(matrix, dom_stride);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   hypre_StructMatrixReadData(file, matrix);

   /*----------------------------------------
    * Assemble the matrix
    *----------------------------------------*/

   hypre_StructMatrixAssemble(matrix);

   /*----------------------------------------
    * Clean up and close file
    *----------------------------------------*/

   HYPRE_StructGridDestroy(grid);
   HYPRE_StructStencilDestroy(stencil);

   fclose(file);

   return matrix;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixMigrate( hypre_StructMatrix *from_matrix,
                           hypre_StructMatrix *to_matrix   )
{
   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;
   hypre_CommHandle      *comm_handle;

   HYPRE_Int              constant_coefficient, comm_num_values;
   HYPRE_Int              stencil_size, mat_num_values;
   hypre_StructStencil   *stencil;
   HYPRE_Int              data_initial_offset = 0;
   HYPRE_Complex         *matrix_data_from = hypre_StructMatrixData(from_matrix);
   HYPRE_Complex         *matrix_data_to = hypre_StructMatrixData(to_matrix);
   HYPRE_Complex         *matrix_data_comm_from = matrix_data_from;
   HYPRE_Complex         *matrix_data_comm_to = matrix_data_to;

   /*------------------------------------------------------
    * Set up hypre_CommPkg
    *------------------------------------------------------*/

   constant_coefficient = hypre_StructMatrixConstantCoefficient( from_matrix );
   hypre_assert( constant_coefficient == hypre_StructMatrixConstantCoefficient( to_matrix ) );

   mat_num_values = hypre_StructMatrixNumValues(from_matrix);
   hypre_assert((mat_num_values == hypre_StructMatrixNumValues(to_matrix)));

   if ( constant_coefficient == 0 )
   {
      comm_num_values = mat_num_values;
   }
   else if ( constant_coefficient == 1 )
   {
      comm_num_values = 0;
   }
   else /* constant_coefficient==2 */
   {
      comm_num_values = 1;
      stencil = hypre_StructMatrixStencil(from_matrix);
      stencil_size = hypre_StructStencilSize(stencil);
      hypre_assert(stencil_size ==
                   hypre_StructStencilSize( hypre_StructMatrixStencil(to_matrix) ) );
      data_initial_offset = stencil_size;
      matrix_data_comm_from = &( matrix_data_from[data_initial_offset] );
      matrix_data_comm_to = &( matrix_data_to[data_initial_offset] );
   }

   hypre_CreateCommInfoFromGrids(hypre_StructMatrixGrid(from_matrix),
                                 hypre_StructMatrixGrid(to_matrix),
                                 &comm_info);
   hypre_CommPkgCreate(comm_info,
                       hypre_StructMatrixDataSpace(from_matrix),
                       hypre_StructMatrixDataSpace(to_matrix),
                       comm_num_values, NULL, 0,
                       hypre_StructMatrixComm(from_matrix),
                       hypre_StructMatrixMemoryLocation(from_matrix),
                       &comm_pkg);
   hypre_CommInfoDestroy(comm_info);
   /* is this correct for periodic? */

   /*-----------------------------------------------------------------------
    * Migrate the matrix data
    *-----------------------------------------------------------------------*/

   if (constant_coefficient != 1)
   {
      hypre_StructCommunicationInitialize(comm_pkg,
                                          &matrix_data_comm_from,
                                          &matrix_data_comm_to, 0, 0,
                                          &comm_handle);
      hypre_StructCommunicationFinalize(comm_handle);
   }
   hypre_CommPkgDestroy(comm_pkg);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * clears matrix stencil coefficients reaching outside of the physical boundaries
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixClearBoundary( hypre_StructMatrix *matrix)
{
   HYPRE_Int            ndim = hypre_StructMatrixNDim(matrix);
   HYPRE_Complex       *data;
   hypre_BoxArray      *grid_boxes, *data_space, *boundary;
   hypre_Box           *box, *dbox, *tbox;
   hypre_Index         *shape;
   hypre_Index          stencil_offset;
   hypre_Index          loop_size;
   hypre_IndexRef       dstart;
   hypre_Index          origin, stride, dstride;
   hypre_StructGrid    *grid;
   hypre_StructStencil *stencil;

   HYPRE_Int           i, j, e;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   box = hypre_BoxCreate(ndim);

   grid = hypre_StructMatrixGrid(matrix);
   stencil = hypre_StructMatrixStencil(matrix);
   grid_boxes = hypre_StructGridBoxes(grid);
   ndim = hypre_StructStencilNDim(stencil);
   data_space = hypre_StructMatrixDataSpace(matrix);
   hypre_SetIndex(dstride, 1);
   shape = hypre_StructStencilShape(stencil);

   for (e = 0; e < hypre_StructStencilSize(stencil); e++)
   {
      if (!hypre_StructMatrixConstEntry(matrix, e))
      {
         hypre_CopyIndex(shape[e], stencil_offset);
         if (!hypre_IndexEqual(stencil_offset, 0, ndim))
         {
            hypre_ForBoxI(i, grid_boxes)
            {
               hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, i), box);
               hypre_StructMatrixGetStencilSpace(matrix, e, 0, origin, stride);
               /*hypre_ProjectBox(box, origin, stride);*/
               dbox = hypre_BoxArrayBox(data_space, i);
               boundary = hypre_BoxArrayCreate(0, ndim);
               hypre_GeneralBoxBoundaryIntersect(box, grid, stencil_offset, boundary);
               data = hypre_StructMatrixBoxData(matrix, i, e);
               hypre_ForBoxI(j, boundary)
               {
                  tbox = hypre_BoxArrayBox(boundary, j);
                  hypre_ProjectBox(tbox, origin, stride);
                  if (hypre_BoxVolume(tbox))
                  {
                     hypre_StructMatrixMapDataBox(matrix, tbox);
                     hypre_BoxGetSize(tbox, loop_size);
                     dstart = hypre_BoxIMin(tbox);
                     hypre_BoxLoop1Begin(ndim, loop_size, dbox, dstart, dstride, di);
                     {
                        data[di] = 0.0;
                     }
                     hypre_BoxLoop1End(di);
                  }
               }
               hypre_BoxArrayDestroy(boundary);
            }
         }
      }
   }

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the diagonal of a square StructMatrix as a StructVector
 *
 * TODO: Create StructVector if diag is NULL
 *
 * RDF TODO: This routine should assume that the base grid for matrix and diag
 * are the same.  Extend it to work when the matrix has non-unitary stride.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatrixGetDiagonal( hypre_StructMatrix  *matrix,
                               hypre_StructVector  *diag )
{
   HYPRE_Int               ndim    = hypre_StructMatrixNDim(matrix);
   hypre_StructGrid       *grid    = hypre_StructMatrixGrid(matrix);
   hypre_BoxArray         *boxes   = hypre_StructGridBoxes(grid);
   hypre_StructStencil    *stencil = hypre_StructMatrixStencil(matrix);
   hypre_BoxArray         *A_data_space = hypre_StructMatrixDataSpace(matrix);
   hypre_BoxArray         *x_data_space = hypre_StructVectorDataSpace(diag);

   hypre_IndexRef          start;
   hypre_Index             ustride;
   hypre_Index             loop_size;
   hypre_Box              *box;
   hypre_Box              *A_data_box;
   hypre_Box              *d_data_box;
   HYPRE_Int               stencil_diag;
   HYPRE_Int               i;
   HYPRE_Real             *Ap, *dp;

   hypre_SetIndex(ustride, 1);
   hypre_ForBoxArrayI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);
      stencil_diag = hypre_StructStencilDiagEntry(stencil);

      A_data_box = hypre_BoxArrayBox(A_data_space, i);
      d_data_box = hypre_BoxArrayBox(x_data_space, i);

      Ap = hypre_StructMatrixBoxData(matrix, i, stencil_diag);
      dp = hypre_StructVectorBoxData(diag, i);

      start = hypre_BoxIMin(box);
      hypre_BoxGetStrideSize(box, ustride, loop_size);

      hypre_BoxLoop2Begin(ndim, loop_size,
                          A_data_box, start, ustride, Ai,
                          d_data_box, start, ustride, di)
      {
         dp[di] = Ap[Ai];
      }
      hypre_BoxLoop2End(Ai, di)
   }

   return hypre_error_flag;
}
