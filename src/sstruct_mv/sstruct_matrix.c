/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_SStructPMatrix class.
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "_hypre_struct_mv.hpp"
#include "_hypre_onedpl.hpp"

/* #define DEBUG_MATCONV */
/* #define DEBUG_U2S */
/* #define DEBUG_SETBOX */

/*==========================================================================
 * SStructPMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixRef( hypre_SStructPMatrix  *matrix,
                         hypre_SStructPMatrix **matrix_ref )
{
   hypre_SStructPMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO: If we had a data structure for an integer array that keeps track
 *       of its size, we could combine num_centries and centries into a
 *       single variable of that type. The same could be done with
 *       sentries_size and sentries.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixCreate( MPI_Comm               comm,
                            hypre_SStructPGrid    *pgrid,
                            hypre_SStructStencil **stencils,
                            hypre_SStructPMatrix **pmatrix_ptr )
{
   HYPRE_Int              ndim  = hypre_SStructPGridNDim(pgrid);
   HYPRE_Int              nvars = hypre_SStructPGridNVars(pgrid);

   hypre_SStructPMatrix  *pmatrix;
   HYPRE_Int            **smaps;
   hypre_StructStencil ***sstencils;
   hypre_StructMatrix  ***smatrices;
   HYPRE_Int            **symmetric;
   HYPRE_Int            **num_centries;
   HYPRE_Int           ***centries;

   hypre_StructStencil   *sstencil;
   HYPRE_Int             *vars;
   hypre_Index           *sstencil_shape;
   HYPRE_Int              sstencil_size;
   HYPRE_Int             *new_sizes;
   hypre_Index          **new_shapes;
   HYPRE_Int              size;
   hypre_StructGrid      *sgrid;

   HYPRE_Int              vi, vj;
   HYPRE_Int              i, j, k;

   pmatrix = hypre_TAlloc(hypre_SStructPMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_SStructPMatrixComm(pmatrix)     = comm;
   hypre_SStructPGridRef(pgrid, &hypre_SStructPMatrixPGrid(pmatrix));
   hypre_SStructPMatrixStencils(pmatrix) = stencils;
   hypre_SStructPMatrixNVars(pmatrix)    = nvars;

   /* create sstencils */
   smaps      = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
   sstencils  = hypre_TAlloc(hypre_StructStencil **, nvars, HYPRE_MEMORY_HOST);
   new_sizes  = hypre_TAlloc(HYPRE_Int, nvars, HYPRE_MEMORY_HOST);
   new_shapes = hypre_TAlloc(hypre_Index *, nvars, HYPRE_MEMORY_HOST);
   size = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      sstencils[vi] = hypre_TAlloc(hypre_StructStencil *, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sstencils[vi][vj] = NULL;
         new_sizes[vj] = 0;
      }

      sstencil       = hypre_SStructStencilSStencil(stencils[vi]);
      vars           = hypre_SStructStencilVars(stencils[vi]);
      sstencil_shape = hypre_StructStencilShape(sstencil);
      sstencil_size  = hypre_StructStencilSize(sstencil);

      smaps[vi] = hypre_TAlloc(HYPRE_Int, sstencil_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         new_sizes[j]++;
      }
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            new_shapes[vj] = hypre_TAlloc(hypre_Index, new_sizes[vj], HYPRE_MEMORY_HOST);
            new_sizes[vj] = 0;
         }
      }
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         k = new_sizes[j];
         hypre_CopyIndex(sstencil_shape[i], new_shapes[j][k]);
         smaps[vi][i] = k;
         new_sizes[j]++;
      }
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            sstencils[vi][vj] =
               hypre_StructStencilCreate(ndim, new_sizes[vj], new_shapes[vj]);
         }
         size = hypre_max(size, new_sizes[vj]);
      }
   }
   hypre_SStructPMatrixSMaps(pmatrix)     = smaps;
   hypre_SStructPMatrixSStencils(pmatrix) = sstencils;
   hypre_TFree(new_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(new_shapes, HYPRE_MEMORY_HOST);

   /* create smatrices */
   smatrices = hypre_TAlloc(hypre_StructMatrix **, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatrices[vi] = hypre_TAlloc(hypre_StructMatrix *, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         smatrices[vi][vj] = NULL;
         if (sstencils[vi][vj] != NULL)
         {
            sgrid = hypre_SStructPGridSGrid(pgrid, vi);
            smatrices[vi][vj] =
               hypre_StructMatrixCreate(comm, sgrid, sstencils[vi][vj]);
         }
      }
   }
   hypre_SStructPMatrixSMatrices(pmatrix) = smatrices;

   /* create domain and range grid strides */
   hypre_SetIndex(hypre_SStructPMatrixDomainStride(pmatrix), 1);
   hypre_SetIndex(hypre_SStructPMatrixRangeStride(pmatrix), 1);

   /* create arrays */
   symmetric     = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
   num_centries  = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
   centries      = hypre_TAlloc(HYPRE_Int **, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      symmetric[vi]    = hypre_TAlloc(HYPRE_Int, nvars, HYPRE_MEMORY_HOST);
      num_centries[vi] = hypre_TAlloc(HYPRE_Int, nvars, HYPRE_MEMORY_HOST);
      centries[vi]     = hypre_TAlloc(HYPRE_Int *, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         centries[vi][vj]     = hypre_TAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
         symmetric[vi][vj]    = 0;
         num_centries[vi][vj] = 0;
      }
   }

   hypre_SStructPMatrixSymmetric(pmatrix) = symmetric;
   hypre_SStructPMatrixNumCEntries(pmatrix) = num_centries;
   hypre_SStructPMatrixCEntries(pmatrix) = centries;
   hypre_SStructPMatrixSEntriesSize(pmatrix) = size;
   hypre_SStructPMatrixSEntries(pmatrix) = hypre_TAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
   hypre_SStructPMatrixAccumulated(pmatrix) = 0;
   hypre_SStructPMatrixRefCount(pmatrix) = 1;

   *pmatrix_ptr = pmatrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixDestroy( hypre_SStructPMatrix *pmatrix )
{
   hypre_SStructStencil  **stencils;
   HYPRE_Int               nvars;
   HYPRE_Int             **smaps;
   hypre_StructStencil  ***sstencils;
   hypre_StructMatrix   ***smatrices;
   HYPRE_Int             **symmetric;
   HYPRE_Int             **num_centries;
   HYPRE_Int            ***centries;
   HYPRE_Int              *sentries;

   HYPRE_Int               vi, vj;

   if (pmatrix)
   {
      hypre_SStructPMatrixRefCount(pmatrix) --;
      if (hypre_SStructPMatrixRefCount(pmatrix) == 0)
      {
         stencils     = hypre_SStructPMatrixStencils(pmatrix);
         nvars        = hypre_SStructPMatrixNVars(pmatrix);
         smaps        = hypre_SStructPMatrixSMaps(pmatrix);
         sstencils    = hypre_SStructPMatrixSStencils(pmatrix);
         smatrices    = hypre_SStructPMatrixSMatrices(pmatrix);
         symmetric    = hypre_SStructPMatrixSymmetric(pmatrix);
         num_centries = hypre_SStructPMatrixNumCEntries(pmatrix);
         centries     = hypre_SStructPMatrixCEntries(pmatrix);
         sentries     = hypre_SStructPMatrixSEntries(pmatrix);

         for (vi = 0; vi < nvars; vi++)
         {
            HYPRE_SStructStencilDestroy(stencils[vi]);
            hypre_TFree(smaps[vi], HYPRE_MEMORY_HOST);
            for (vj = 0; vj < nvars; vj++)
            {
               hypre_StructStencilDestroy(sstencils[vi][vj]);
               hypre_StructMatrixDestroy(smatrices[vi][vj]);
               hypre_TFree(centries[vi][vj], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(sstencils[vi], HYPRE_MEMORY_HOST);
            hypre_TFree(smatrices[vi], HYPRE_MEMORY_HOST);
            hypre_TFree(symmetric[vi], HYPRE_MEMORY_HOST);
            hypre_TFree(num_centries[vi], HYPRE_MEMORY_HOST);
            hypre_TFree(centries[vi], HYPRE_MEMORY_HOST);
         }
         hypre_SStructPGridDestroy(hypre_SStructPMatrixPGrid(pmatrix));
         hypre_TFree(stencils, HYPRE_MEMORY_HOST);
         hypre_TFree(smaps, HYPRE_MEMORY_HOST);
         hypre_TFree(sstencils, HYPRE_MEMORY_HOST);
         hypre_TFree(smatrices, HYPRE_MEMORY_HOST);
         hypre_TFree(symmetric, HYPRE_MEMORY_HOST);
         hypre_TFree(num_centries, HYPRE_MEMORY_HOST);
         hypre_TFree(centries, HYPRE_MEMORY_HOST);
         hypre_TFree(sentries, HYPRE_MEMORY_HOST);
         hypre_TFree(pmatrix, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixInitialize( hypre_SStructPMatrix *pmatrix )
{
   HYPRE_Int             nvars        = hypre_SStructPMatrixNVars(pmatrix);
   HYPRE_Int           **symmetric    = hypre_SStructPMatrixSymmetric(pmatrix);
   HYPRE_Int           **num_centries = hypre_SStructPMatrixNumCEntries(pmatrix);
   HYPRE_Int          ***centries     = hypre_SStructPMatrixCEntries(pmatrix);
   hypre_IndexRef        dom_stride   = hypre_SStructPMatrixDomainStride(pmatrix);
   hypre_IndexRef        ran_stride   = hypre_SStructPMatrixRangeStride(pmatrix);
   //   HYPRE_Int             num_ghost[2*HYPRE_MAXDIM];
   hypre_StructMatrix   *smatrix;
   HYPRE_Int             vi, vj;
   //   HYPRE_Int             d, ndim;

#if 0
   ndim = hypre_SStructPMatrixNDim(pmatrix);
   /* RDF: Why are the ghosts being reset to one? Maybe it needs to be at least
    * one to set shared coefficients correctly, but not exactly one? */
   for (d = 0; d < ndim; d++)
   {
      num_ghost[2 * d] = num_ghost[2 * d + 1] = 1;
   }
   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      num_ghost[2 * d] = num_ghost[2 * d + 1] = 0;
   }
#endif
   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            HYPRE_StructMatrixSetDomainStride(smatrix, dom_stride);
            HYPRE_StructMatrixSetRangeStride(smatrix, ran_stride);
            HYPRE_StructMatrixSetConstantEntries(smatrix,
                                                 num_centries[vi][vj],
                                                 centries[vi][vj]);
            HYPRE_StructMatrixSetSymmetric(smatrix, symmetric[vi][vj]);
            //            HYPRE_StructMatrixSetNumGhost(smatrix, num_ghost);
            hypre_StructMatrixInitialize(smatrix);
            /* needed to get AddTo accumulation correct between processors */
            hypre_StructMatrixClearGhostValues(smatrix);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixSetValues( hypre_SStructPMatrix *pmatrix,
                               hypre_Index           index,
                               HYPRE_Int             var,
                               HYPRE_Int             nentries,
                               HYPRE_Int            *entries,
                               HYPRE_Complex        *values,
                               HYPRE_Int             action )
{
   hypre_SStructStencil *stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   HYPRE_Int            *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   HYPRE_Int            *vars    = hypre_SStructStencilVars(stencil);
   hypre_StructMatrix   *smatrix;
   hypre_BoxArray       *grid_boxes;
   hypre_Box            *box, *grow_box;
   HYPRE_Int            *sentries;
   HYPRE_Int             i;

   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   sentries = hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   /* set values inside the grid */
   hypre_StructMatrixSetValues(smatrix, index, nentries, sentries, values,
                               action, -1, 0);

   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      hypre_SStructPGrid  *pgrid = hypre_SStructPMatrixPGrid(pmatrix);
      hypre_Index          varoffset;
      HYPRE_Int            done = 0;

      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(smatrix));

      hypre_ForBoxI(i, grid_boxes)
      {
         box = hypre_BoxArrayBox(grid_boxes, i);
         if (hypre_IndexInBox(index, box))
         {
            done = 1;
            break;
         }
      }

      if (!done)
      {
         grow_box = hypre_BoxCreate(hypre_BoxArrayNDim(grid_boxes));
         hypre_SStructVariableGetOffset(hypre_SStructPGridVarType(pgrid, var),
                                        hypre_SStructPGridNDim(pgrid), varoffset);
         hypre_ForBoxI(i, grid_boxes)
         {
            box = hypre_BoxArrayBox(grid_boxes, i);
            hypre_CopyBox(box, grow_box);
            hypre_BoxGrowByIndex(grow_box, varoffset);
            if (hypre_IndexInBox(index, grow_box))
            {
               hypre_StructMatrixSetValues(smatrix, index, nentries, sentries,
                                           values, action, i, 1);
               break;
            }
         }
         hypre_BoxDestroy(grow_box);
      }
   }
   else
   {
      /* Set */
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(smatrix));

      hypre_ForBoxI(i, grid_boxes)
      {
         box = hypre_BoxArrayBox(grid_boxes, i);
         if (!hypre_IndexInBox(index, box))
         {
            hypre_StructMatrixClearValues(smatrix, index, nentries, sentries, i, 1);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixSetBoxValues( hypre_SStructPMatrix *pmatrix,
                                  hypre_Box            *set_box,
                                  HYPRE_Int             var,
                                  HYPRE_Int             nentries,
                                  HYPRE_Int            *entries,
                                  hypre_Box            *value_box,
                                  HYPRE_Complex        *values,
                                  HYPRE_Int             action )
{
   HYPRE_Int             ndim    = hypre_SStructPMatrixNDim(pmatrix);
   hypre_SStructStencil *stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   HYPRE_Int            *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   HYPRE_Int            *vars    = hypre_SStructStencilVars(stencil);
   hypre_StructMatrix   *smatrix;
   hypre_BoxArray       *grid_boxes;
   HYPRE_Int            *sentries;
   HYPRE_Int             i, j;

   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   sentries = hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   /* set values inside the grid */
   hypre_StructMatrixSetBoxValues(smatrix, set_box, value_box, nentries, sentries,
                                  values, action, -1, 0);

   /* TODO: Why need DeviceSync? */
#if defined(HYPRE_USING_GPU)
   hypre_SyncDevice();
#endif

   /* set (AddTo/Get) or clear (Set) values outside the grid in ghost zones */
   if (action != 0)
   {
      /* AddTo/Get */
      hypre_SStructPGrid  *pgrid = hypre_SStructPMatrixPGrid(pmatrix);
      hypre_Index          varoffset;
      hypre_BoxArray      *left_boxes, *done_boxes, *temp_boxes;
      hypre_Box           *left_box, *done_box, *int_box;

      hypre_SStructVariableGetOffset(hypre_SStructPGridVarType(pgrid, var),
                                     hypre_SStructPGridNDim(pgrid), varoffset);
      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(smatrix));

      left_boxes = hypre_BoxArrayCreate(1, ndim);
      done_boxes = hypre_BoxArrayCreate(2, ndim);
      temp_boxes = hypre_BoxArrayCreate(0, ndim);

      /* done_box always points to the first box in done_boxes */
      done_box = hypre_BoxArrayBox(done_boxes, 0);
      /* int_box always points to the second box in done_boxes */
      int_box = hypre_BoxArrayBox(done_boxes, 1);

      hypre_CopyBox(set_box, hypre_BoxArrayBox(left_boxes, 0));
      hypre_BoxArraySetSize(left_boxes, 1);
      hypre_SubtractBoxArrays(left_boxes, grid_boxes, temp_boxes);

      hypre_BoxArraySetSize(done_boxes, 0);
      hypre_ForBoxI(i, grid_boxes)
      {
         hypre_SubtractBoxArrays(left_boxes, done_boxes, temp_boxes);
         hypre_BoxArraySetSize(done_boxes, 1);
         hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, i), done_box);
         hypre_BoxGrowByIndex(done_box, varoffset);
         hypre_ForBoxI(j, left_boxes)
         {
            left_box = hypre_BoxArrayBox(left_boxes, j);
            hypre_IntersectBoxes(left_box, done_box, int_box);
            hypre_StructMatrixSetBoxValues(smatrix, int_box, value_box,
                                           nentries, sentries,
                                           values, action, i, 1);
         }
      }

      hypre_BoxArrayDestroy(left_boxes);
      hypre_BoxArrayDestroy(done_boxes);
      hypre_BoxArrayDestroy(temp_boxes);
   }
   else
   {
      /* Set */
      hypre_BoxArray  *diff_boxes;
      hypre_Box       *grid_box, *diff_box;

      grid_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(smatrix));
      diff_boxes = hypre_BoxArrayCreate(0, ndim);

      hypre_ForBoxI(i, grid_boxes)
      {
         grid_box = hypre_BoxArrayBox(grid_boxes, i);
         hypre_BoxArraySetSize(diff_boxes, 0);
         hypre_SubtractBoxes(set_box, grid_box, diff_boxes);

         hypre_ForBoxI(j, diff_boxes)
         {
            diff_box = hypre_BoxArrayBox(diff_boxes, j);
            hypre_StructMatrixClearBoxValues(smatrix, diff_box, nentries, sentries,
                                             i, 1);
         }
      }
      hypre_BoxArrayDestroy(diff_boxes);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixAccumulate( hypre_SStructPMatrix *pmatrix )
{
   hypre_SStructPGrid    *pgrid    = hypre_SStructPMatrixPGrid(pmatrix);
   HYPRE_Int              nvars    = hypre_SStructPMatrixNVars(pmatrix);
   HYPRE_Int              ndim     = hypre_SStructPGridNDim(pgrid);
   HYPRE_SStructVariable *vartypes = hypre_SStructPGridVarTypes(pgrid);

   hypre_StructMatrix    *smatrix;
   hypre_Index            varoffset;
   HYPRE_Int              num_ghost[2 * HYPRE_MAXDIM];
   hypre_StructGrid      *sgrid;
   HYPRE_Int              vi, vj, d;

   hypre_CommInfo        *comm_info;
   hypre_CommPkg         *comm_pkg;
   hypre_CommHandle      *comm_handle;
   HYPRE_Complex         *data;
   hypre_Index            ustride;

   /* if values already accumulated, just return */
   if (hypre_SStructPMatrixAccumulated(pmatrix))
   {
      return hypre_error_flag;
   }

   hypre_SetIndex(ustride, 1);

   for (d = ndim; d < HYPRE_MAXDIM; d++)
   {
      num_ghost[2 * d] = num_ghost[2 * d + 1] = 0;
   }
   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            sgrid = hypre_StructMatrixGrid(smatrix);
            /* assumes vi and vj vartypes are the same */
            hypre_SStructVariableGetOffset(vartypes[vi], ndim, varoffset);
            for (d = 0; d < ndim; d++)
            {
               num_ghost[2 * d] = num_ghost[2 * d + 1] = hypre_IndexD(varoffset, d);
            }

            /* accumulate values from AddTo */
            hypre_CreateCommInfoFromNumGhost(sgrid, ustride, num_ghost, &comm_info);
            hypre_CommPkgCreate(comm_info,
                                hypre_StructMatrixDataSpace(smatrix),
                                hypre_StructMatrixDataSpace(smatrix),
                                hypre_StructMatrixNumValues(smatrix), NULL, 1,
                                hypre_StructMatrixComm(smatrix),
                                hypre_StructMatrixMemoryLocation(smatrix),
                                &comm_pkg);
            data = hypre_StructMatrixVData(smatrix);
            hypre_StructCommunicationInitialize(comm_pkg, &data, &data, 1, 0, &comm_handle);
            hypre_StructCommunicationFinalize(comm_handle);

            hypre_CommInfoDestroy(comm_info);
            hypre_CommPkgDestroy(comm_pkg);
         }
      }
   }

   hypre_SStructPMatrixAccumulated(pmatrix) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixAssemble( hypre_SStructPMatrix *pmatrix )
{
   HYPRE_Int              nvars    = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix    *smatrix;
   HYPRE_Int              vi, vj;

   hypre_SStructPMatrixAccumulate(pmatrix);

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            hypre_StructMatrixClearGhostValues(smatrix);
            hypre_StructMatrixAssemble(smatrix);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine may be called at any time.  The boolean 'resize' is returned to
 * indicate whether a MatrixResize() is needed.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixSetTranspose( hypre_SStructPMatrix *pmatrix,
                                  HYPRE_Int             transpose,
                                  HYPRE_Int            *resize )
{
   HYPRE_Int              nvars    = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix    *smatrix;
   HYPRE_Int              vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            hypre_StructMatrixSetTranspose(smatrix, transpose, resize);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * TODO: Deprecate this function. var == -1 or to_var == -1 are never used.
 *       These cases are used only in HYPRE_SStructMatrixSetSymmetric.
 *
 * RDF: The '-1' cases are used in the sstruct driver, and the thought was that
 * it would be a useful way to set everything symmetric.
 *--------------------------------------------------------------------------*/
#if 1
HYPRE_Int
hypre_SStructPMatrixSetSymmetric( hypre_SStructPMatrix *pmatrix,
                                  HYPRE_Int             var,
                                  HYPRE_Int             to_var,
                                  HYPRE_Int             symmetric )
{
   HYPRE_Int **pmsymmetric = hypre_SStructPMatrixSymmetric(pmatrix);

   HYPRE_Int vstart = var;
   HYPRE_Int vsize  = 1;
   HYPRE_Int tstart = to_var;
   HYPRE_Int tsize  = 1;
   HYPRE_Int v, t;

   if (var == -1)
   {
      vstart = 0;
      vsize  = hypre_SStructPMatrixNVars(pmatrix);
   }
   if (to_var == -1)
   {
      tstart = 0;
      tsize  = hypre_SStructPMatrixNVars(pmatrix);
   }

   for (v = vstart; v < vsize; v++)
   {
      for (t = tstart; t < tsize; t++)
      {
         pmsymmetric[v][t] = symmetric;
      }
   }

   return hypre_error_flag;
}

#else
/*--------------------------------------------------------------------------
 * NOTE: Should we have an accessor macro for doing this job?
 *       How would we call it?
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructPMatrixSetSymmetric( hypre_SStructPMatrix *pmatrix,
                                  HYPRE_Int             var,
                                  HYPRE_Int             to_var,
                                  HYPRE_Int             symmetric )
{
   HYPRE_Int **pmsymmetric = hypre_SStructPMatrixSymmetric(pmatrix);

   pmsymmetric[var][to_var] = symmetric;

   return hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixSetCEntries( hypre_SStructPMatrix *pmatrix,
                                 HYPRE_Int             var,
                                 HYPRE_Int             to_var,
                                 HYPRE_Int             num_centries,
                                 HYPRE_Int            *centries )
{
   HYPRE_Int   **pmnum_centries = hypre_SStructPMatrixNumCEntries(pmatrix);
   HYPRE_Int  ***pmcentries     = hypre_SStructPMatrixCEntries(pmatrix);
   HYPRE_Int     i;

   pmnum_centries[var][to_var] = num_centries;
   for (i = 0; i < num_centries; i++)
   {
      pmcentries[var][to_var][i] = centries[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Should we have an accessor macro for doing this job?
 *       How would we call it?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixSetDomainStride( hypre_SStructPMatrix *pmatrix,
                                     hypre_Index           dom_stride )
{
   hypre_CopyIndex(dom_stride, hypre_SStructPMatrixDomainStride(pmatrix));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Should we have an accessor macro for doing this job?
 *       How would we call it?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixSetRangeStride( hypre_SStructPMatrix *pmatrix,
                                    hypre_Index           ran_stride )
{
   hypre_CopyIndex(ran_stride, hypre_SStructPMatrixRangeStride(pmatrix));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixPrint( const char           *filename,
                           hypre_SStructPMatrix *pmatrix,
                           HYPRE_Int             all )
{
   HYPRE_Int           nvars = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix *smatrix;
   HYPRE_Int           vi, vj;
   char                new_filename[255];

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            //            hypre_sprintf(new_filename, "%s.v%1d%1d", filename, vi, vj);
            hypre_sprintf(new_filename, "%s.%1d%1d", filename, vi, vj);
            hypre_StructMatrixPrint(new_filename, smatrix, all);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the diagonal of a SStructPMatrix as a SStructPVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixGetDiagonal( hypre_SStructPMatrix  *pmatrix,
                                 hypre_SStructPVector  *pdiag )
{
   HYPRE_Int               nvars  = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix     *smatrix;
   hypre_StructVector     *sdiag;

   HYPRE_Int               var;

   for (var = 0; var < nvars; var++)
   {
      smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, var);
      sdiag   = hypre_SStructPVectorSVector(pdiag, var);

      hypre_StructMatrixGetDiagonal(smatrix, sdiag);
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructUMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixInitialize( hypre_SStructMatrix  *matrix,
                                HYPRE_MemoryLocation  memory_location )
{
   HYPRE_Int               ndim          = hypre_SStructMatrixNDim(matrix);
   HYPRE_IJMatrix          ijmatrix      = hypre_SStructMatrixIJMatrix(matrix);
   HYPRE_Int               matrix_type   = hypre_SStructMatrixObjectType(matrix);
   hypre_SStructGraph     *graph         = hypre_SStructMatrixGraph(matrix);
   hypre_SStructStencil ***stencils      = hypre_SStructGraphStencils(graph);
   HYPRE_Int               nUventries    = hypre_SStructGraphNUVEntries(graph);
   HYPRE_Int              *iUventries    = hypre_SStructGraphIUVEntries(graph);
   hypre_SStructUVEntry  **Uventries     = hypre_SStructGraphUVEntries(graph);
   HYPRE_Int               nparts        = hypre_SStructGraphNParts(graph);
   hypre_SStructGrid      *grid          = hypre_SStructGraphGrid(graph);
   HYPRE_Int             **nvneighbors   = hypre_SStructGridNVNeighbors(grid);

   hypre_SStructPGrid     *pgrid;
   hypre_StructGrid       *sgrid;
   hypre_SStructStencil   *stencil;
   hypre_BoxArray         *boxes;
   hypre_Box              *box;
   hypre_Box              *ghost_box;
   hypre_IndexRef          start;
   hypre_Index             loop_size, stride;

   HYPRE_Int              *split;
   HYPRE_Int               nvars;
   HYPRE_Int               nrows, nnzrow = 0;
   HYPRE_BigInt            rowstart;
   HYPRE_Int               part, var, entry, b, m, mi;
   HYPRE_Int              *row_sizes;
   HYPRE_Int               max_size = 0;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy   exec  = hypre_GetExecPolicy1(memory_location);
#endif

   HYPRE_IJMatrixSetObjectType(ijmatrix, HYPRE_PARCSR);
   if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
   {
      rowstart = hypre_SStructGridGhstartRank(grid);
      nrows    = hypre_SStructGridGhlocalSize(grid);
   }
   else /* matrix_type == HYPRE_PARCSR */
   {
      rowstart = hypre_SStructGridStartRank(grid);
      nrows    = hypre_SStructGridLocalSize(grid);
   }

   /* Set row_sizes and max_size */
   m = 0;
   ghost_box = hypre_BoxCreate(ndim);
#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      row_sizes = NULL;
   }
   else
#endif
   {
      row_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   }
   hypre_SetIndex(stride, 1);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      /* This part is active in the range grid */
      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, var);

         stencil = stencils[part][var];
         split = hypre_SStructMatrixSplit(matrix, part, var);
         nnzrow = 0;
         for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] == -1)
            {
               nnzrow++;
            }
         }
#if 0
         /* TODO: For now, assume stencil is full/complete */
         if (hypre_SStructMatrixSymmetric(matrix))
         {
            nnzrow = 2 * nnzrow - 1;
         }
#endif

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_HOST)
#endif
         {
            boxes = hypre_StructGridBoxes(sgrid);
            hypre_ForBoxI(b, boxes)
            {
               box = hypre_BoxArrayBox(boxes, b);
               hypre_CopyBox(box, ghost_box);
               if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
               {
                  hypre_BoxGrowByArray(ghost_box, hypre_StructGridNumGhost(sgrid));
               }

               start = hypre_BoxIMin(box);
               hypre_BoxGetSize(box, loop_size);
               hypre_BoxLoop1BeginHost(ndim, loop_size, ghost_box, start, stride, mi);
               {
                  row_sizes[m + mi] = nnzrow;
               }
               hypre_BoxLoop1EndHost(mi);

               m += hypre_BoxVolume(ghost_box);
            }
         }

         max_size = hypre_max(max_size, nnzrow);
         if (nvneighbors[part][var])
         {
            max_size = hypre_max(max_size, hypre_SStructStencilSize(stencil));
         }
      } /* loop on variables */
   } /* loop on parts */
   hypre_BoxDestroy(ghost_box);

   /* GEC0902 essentially for each UVentry we figure out how many
    * extra columns we need to add to the rowsizes */

   /* RDF: THREAD? */
   for (entry = 0; entry < nUventries; entry++)
   {
      mi = iUventries[entry];
      m = hypre_SStructUVEntryRank(Uventries[mi]) - rowstart;
      if ((m > -1) && (m < nrows))
      {
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_HOST)
#endif
         {
            row_sizes[m] += hypre_SStructUVEntryNUEntries(Uventries[mi]);
         }
         max_size = hypre_max(max_size, nnzrow + hypre_SStructUVEntryNUEntries(Uventries[mi]));
      }
   }

#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_HOST)
#endif
   {
      /* ZTODO: Update row_sizes based on neighbor off-part couplings */
      HYPRE_IJMatrixSetRowSizes(ijmatrix, (const HYPRE_Int *) row_sizes);
      hypre_TFree(row_sizes, HYPRE_MEMORY_HOST);
   }

   hypre_SStructMatrixTmpSize(matrix)            = max_size;
   hypre_SStructMatrixTmpRowCoords(matrix)       = hypre_CTAlloc(HYPRE_BigInt,  max_size,
                                                                 HYPRE_MEMORY_HOST);
   hypre_SStructMatrixTmpColCoords(matrix)       = hypre_CTAlloc(HYPRE_BigInt,  max_size,
                                                                 HYPRE_MEMORY_HOST);
   hypre_SStructMatrixTmpCoeffs(matrix)          = hypre_CTAlloc(HYPRE_Complex, max_size,
                                                                 HYPRE_MEMORY_HOST);
#if defined (HYPRE_USING_GPU)
   hypre_SStructMatrixTmpRowCoordsDevice(matrix) = hypre_CTAlloc(HYPRE_BigInt,  max_size,
                                                                 HYPRE_MEMORY_DEVICE);
   hypre_SStructMatrixTmpColCoordsDevice(matrix) = hypre_CTAlloc(HYPRE_BigInt,  max_size,
                                                                 HYPRE_MEMORY_DEVICE);
   hypre_SStructMatrixTmpCoeffsDevice(matrix)    = hypre_CTAlloc(HYPRE_Complex, max_size,
                                                                 HYPRE_MEMORY_DEVICE);
#endif

   HYPRE_IJMatrixInitialize_v2(ijmatrix, memory_location);
   HYPRE_IJMatrixGetObject(ijmatrix,
                           (void **) &hypre_SStructMatrixParCSRMatrix(matrix));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * 9/09 - AB: modified to use the box manager - here we need to check the
 *            neighbor box manager also
 *
 * TODO: Do we really need dom_grid here?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixSetValues( hypre_SStructMatrix *matrix,
                               HYPRE_Int            part,
                               hypre_Index          index,
                               HYPRE_Int            var,
                               HYPRE_Int            nentries,
                               HYPRE_Int           *entries,
                               HYPRE_Complex       *values,
                               HYPRE_Int            action )
{
   HYPRE_Int                ndim        = hypre_SStructMatrixNDim(matrix);
   HYPRE_IJMatrix           ijmatrix    = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph      *graph       = hypre_SStructMatrixGraph(matrix);
   HYPRE_Int                matrix_type = hypre_SStructMatrixObjectType(matrix);
   hypre_SStructGrid       *grid        = hypre_SStructGraphGrid(graph);
   hypre_SStructGrid       *dom_grid    = hypre_SStructGraphDomGrid(graph);
   hypre_SStructStencil    *stencil     = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int               *vars        = hypre_SStructStencilVars(stencil);
   hypre_Index             *shape       = hypre_SStructStencilShape(stencil);
   HYPRE_Int                size        = hypre_SStructStencilSize(stencil);

   HYPRE_Complex           *h_values;
   HYPRE_MemoryLocation     memory_location = hypre_IJMatrixMemoryLocation(ijmatrix);

   hypre_IndexRef           offset;
   hypre_Index              to_index;
   hypre_SStructUVEntry    *Uventry;
   hypre_BoxManEntry       *boxman_entry;
   hypre_SStructBoxManInfo *entry_info;
   HYPRE_BigInt             row_coord;
   HYPRE_BigInt            *col_coords;
   HYPRE_Int                ncoeffs;
   HYPRE_Complex           *coeffs;
   HYPRE_Int                i, entry;
   HYPRE_BigInt             Uverank;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);

   /* if not local, check neighbors */
   if (boxman_entry == NULL)
   {
      hypre_SStructGridFindNborBoxManEntry(grid, part, index, var, &boxman_entry);
   }

   if (boxman_entry == NULL)
   {
      hypre_error_in_arg(1);
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);

      HYPRE_ANNOTATE_FUNC_END;

      return hypre_error_flag;
   }
   else
   {
      hypre_BoxManEntryGetInfo(boxman_entry, (void **) &entry_info);
   }

   hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index,
                                         &row_coord, matrix_type);

   col_coords = hypre_SStructMatrixTmpColCoords(matrix);
   coeffs     = hypre_SStructMatrixTmpCoeffs(matrix);

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      h_values = hypre_TAlloc(HYPRE_Complex, nentries, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_values, values, HYPRE_Complex, nentries,
                    HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_values = values;
   }

   /* RL: TODO Port it to GPU? */
   ncoeffs = 0;
   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];

      if (entry < size)
      {
         /* stencil entries */
         offset = shape[entry];
         hypre_AddIndexes(index, offset, ndim, to_index);

         hypre_SStructGridFindBoxManEntry(dom_grid, part, to_index, vars[entry],
                                          &boxman_entry);

         /* if not local, check neighbors */
         if (boxman_entry == NULL)
         {
            hypre_SStructGridFindNborBoxManEntry(dom_grid, part, to_index,
                                                 vars[entry], &boxman_entry);
         }

         if (boxman_entry != NULL)
         {
            hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, to_index,
                                                  &col_coords[ncoeffs], matrix_type);

            coeffs[ncoeffs] = values[i];
            ncoeffs++;
         }
      }
      else
      {
         /* non-stencil entries */
         entry -= size;
         hypre_SStructGraphGetUVEntryRank(graph, part, var, index, &Uverank);

         if (Uverank > -1)
         {
            Uventry = hypre_SStructGraphUVEntry(graph, Uverank);

            /* Sanity check */
            //hypre_assert(entry < hypre_SStructUVEntryNUEntries(Uventry));

            /* Set column number and coefficient */
            col_coords[ncoeffs] = hypre_SStructUVEntryToRank(Uventry, entry);
            coeffs[ncoeffs] = h_values[i];
            ncoeffs++;
         }
      }
   }

#if defined(HYPRE_USING_GPU)
   if ( hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE )
   {
      if (!hypre_SStructMatrixTmpRowCoordsDevice(matrix))
      {
         hypre_SStructMatrixTmpRowCoordsDevice(matrix) =
            hypre_CTAlloc(HYPRE_BigInt, hypre_SStructMatrixTmpSize(matrix), memory_location);
      }

      if (!hypre_SStructMatrixTmpColCoordsDevice(matrix))
      {
         hypre_SStructMatrixTmpColCoordsDevice(matrix) =
            hypre_CTAlloc(HYPRE_BigInt, hypre_SStructMatrixTmpSize(matrix), memory_location);
      }

      if (!hypre_SStructMatrixTmpCoeffsDevice(matrix))
      {
         hypre_SStructMatrixTmpCoeffsDevice(matrix) =
            hypre_CTAlloc(HYPRE_Complex, hypre_SStructMatrixTmpSize(matrix), memory_location);
      }

      hypreDevice_BigIntFilln(hypre_SStructMatrixTmpRowCoordsDevice(matrix), ncoeffs, row_coord);

      hypre_TMemcpy(hypre_SStructMatrixTmpColCoordsDevice(matrix),
                    col_coords, HYPRE_BigInt, ncoeffs,
                    memory_location, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_SStructMatrixTmpCoeffsDevice(matrix),
                    coeffs, HYPRE_Complex, ncoeffs,
                    memory_location, HYPRE_MEMORY_HOST);

      if (action > 0)
      {
         HYPRE_IJMatrixAddToValues(ijmatrix, ncoeffs, NULL,
                                   hypre_SStructMatrixTmpRowCoordsDevice(matrix),
                                   (const HYPRE_BigInt *) hypre_SStructMatrixTmpColCoordsDevice(matrix),
                                   (const HYPRE_Complex *) hypre_SStructMatrixTmpCoeffsDevice(matrix));
      }
      else if (action > -1)
      {
         HYPRE_IJMatrixSetValues(ijmatrix, ncoeffs, NULL,
                                 hypre_SStructMatrixTmpRowCoordsDevice(matrix),
                                 (const HYPRE_BigInt *) hypre_SStructMatrixTmpColCoordsDevice(matrix),
                                 (const HYPRE_Complex *) hypre_SStructMatrixTmpCoeffsDevice(matrix));
      }
      else
      {
         // RL:TODO
         HYPRE_IJMatrixGetValues(ijmatrix, 1, &ncoeffs, &row_coord, col_coords, values);
      }
   }
   else
#endif
   {
      if (action > 0)
      {
         HYPRE_IJMatrixAddToValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                   (const HYPRE_BigInt *) col_coords,
                                   (const HYPRE_Complex *) coeffs);
      }
      else if (action > -1)
      {
         HYPRE_IJMatrixSetValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                 (const HYPRE_BigInt *) col_coords,
                                 (const HYPRE_Complex *) coeffs);
      }
      else
      {
         HYPRE_IJMatrixGetValues(ijmatrix, 1, &ncoeffs, &row_coord,
                                 col_coords, values);
      }
   }

   if (h_values != values)
   {
      hypre_TFree(h_values, HYPRE_MEMORY_HOST);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note: Entries must all be of type stencil or non-stencil, but not both.
 *
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *
 * 9/09 - AB: modified to use the box manager- here we need to check the
 *            neighbor box manager also
 *
 * To illustrate what is computed below before calling IJSetValues2(), consider
 * the following example of a 5-pt stencil (c,w,e,s,n) on a 3x2 grid (the 'x' in
 * arrays 'cols' and 'ijvalues' indicates "no data"):
 *
 *   nrows       = 6
 *   ncols       = 3         4         3         3         4         3
 *   rows        = 0         1         2         3         4         5
 *   row_indexes = 0         5         10        15        20        25
 *   cols        = . . . x x . . . . x . . . x x . . . x x . . . . x . . . x x
 *   ijvalues    = . . . x x . . . . x . . . x x . . . x x . . . . x . . . x x
 *   entry       = c e n     c w e n   c w n     c e s     c w e s   c w s
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixSetBoxValuesHelper( hypre_SStructMatrix *matrix,
                                        HYPRE_Int            part,
                                        hypre_Box           *set_box,
                                        HYPRE_Int            var,
                                        HYPRE_Int            nentries,
                                        HYPRE_Int           *entries,
                                        hypre_Box           *value_box,
                                        HYPRE_Complex       *values,
                                        HYPRE_Int            action,
                                        HYPRE_IJMatrix       ijmatrix )
{
   HYPRE_Int             ndim        = hypre_SStructMatrixNDim(matrix);
   HYPRE_Int             matrix_type = hypre_SStructMatrixObjectType(matrix);
   hypre_SStructPMatrix *pmatrix     = hypre_SStructMatrixPMatrix(matrix, part);
   hypre_SStructGraph   *graph       = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid        = hypre_SStructGraphGrid(graph);
   hypre_SStructGrid    *dom_grid    = hypre_SStructGraphDomGrid(graph);
   hypre_SStructStencil *stencil     = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int            *vars        = hypre_SStructStencilVars(stencil);
   hypre_Index          *shape       = hypre_SStructStencilShape(stencil);
   HYPRE_Int             size        = hypre_SStructStencilSize(stencil);
   hypre_IndexRef        dom_stride  = hypre_SStructPMatrixDomainStride(pmatrix);
   HYPRE_MemoryLocation  memory_location = hypre_IJMatrixMemoryLocation(ijmatrix);

   hypre_IndexRef        offset;
   hypre_BoxManEntry   **boxman_entries;
   HYPRE_Int             nboxman_entries;
   hypre_BoxManEntry   **boxman_to_entries;
   HYPRE_Int             nboxman_to_entries;
   HYPRE_Int             nrows, num_nonzeros;
   HYPRE_Int            *ncols, *row_indexes;
   HYPRE_BigInt         *rows, *cols;
   HYPRE_Complex        *ijvalues;
   HYPRE_Int            *values_map;
   hypre_Box            *box;
   hypre_Box            *to_box;
   hypre_Box            *map_box;
   hypre_Box            *int_box;
   hypre_Box            *map_vbox;
   hypre_Index           index;
   hypre_Index           unit_stride;
   hypre_Index           loop_size;
   hypre_Index           mstart;
   hypre_IndexRef        vstart;
   hypre_Index           rs, cs;
   HYPRE_BigInt          row_base, col_base;
   HYPRE_Int             ei, entry, i, ii, jj;
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memory_location);
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructUMatrixSetBoxValuesHelper");

   /*------------------------------------------
    * all stencil entries
    *------------------------------------------*/

   hypre_SetIndex(unit_stride, 1);
   if (entries[0] < size)
   {
      box      = hypre_BoxCreate(ndim);
      to_box   = hypre_BoxCreate(ndim);
      map_box  = hypre_BoxCreate(ndim);
      int_box  = hypre_BoxCreate(ndim);
      map_vbox = hypre_BoxCreate(ndim);

      nrows        = hypre_BoxVolume(set_box);
      num_nonzeros = nrows * nentries;
      ncols        = hypre_CTAlloc(HYPRE_Int,     nrows,        memory_location);
      rows         = hypre_CTAlloc(HYPRE_BigInt,  nrows,        memory_location);
      row_indexes  = hypre_CTAlloc(HYPRE_Int,     nrows + 1,    memory_location);
      cols         = hypre_CTAlloc(HYPRE_BigInt,  num_nonzeros, memory_location);
      ijvalues     = hypre_TAlloc(HYPRE_Complex,  num_nonzeros, memory_location);
      values_map   = hypre_TAlloc(HYPRE_Int,      num_nonzeros, memory_location);

      /* TODO (VPM): We could wrap this into a separate function */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypreDevice_IntFilln(values_map, num_nonzeros, -1);
      }
      else
#endif
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_nonzeros; i++)
         {
            values_map[i] = -1;
         }
      }

      hypre_SStructGridIntersect(grid, part, var, set_box, -1,
                                 &boxman_entries, &nboxman_entries);

      for (ii = 0; ii < nboxman_entries; ii++)
      {
         hypre_SStructBoxManEntryGetStrides(boxman_entries[ii], rs, matrix_type);

         hypre_BoxManEntryGetExtents(boxman_entries[ii],
                                     hypre_BoxIMin(map_box),
                                     hypre_BoxIMax(map_box));
         hypre_IntersectBoxes(set_box, map_box, int_box);
         hypre_CopyBox(int_box, box);

         /* For each index in 'box', compute a row of length <= nentries and
          * insert it into an nentries-length segment of 'cols' and 'ijvalues'.
          * This may result in gaps, but IJSetValues2() is designed for that. */
         nrows = hypre_BoxVolume(box);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntFilln(ncols, nrows, 0);
            HYPRE_THRUST_CALL( transform,
                               thrust::counting_iterator<HYPRE_Int>(0),
                               thrust::counting_iterator<HYPRE_Int>(nrows + 1),
                               row_indexes,
                               _1 * nentries );
         }
         else
#endif
         {
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < nrows; i++)
            {
               ncols[i] = 0;
               row_indexes[i] = i * nentries;
            }
         }

         for (ei = 0; ei < nentries; ei++)
         {
            entry  = entries[ei];
            offset = shape[entry];

            hypre_CopyBox(box, to_box);
            hypre_BoxShiftPos(to_box, offset);
            hypre_CoarsenBox(to_box, NULL, dom_stride);

            hypre_SStructGridIntersect(dom_grid, part, vars[entry], to_box, -1,
                                       &boxman_to_entries, &nboxman_to_entries);

            for (jj = 0; jj < nboxman_to_entries; jj++)
            {
               hypre_SStructBoxManEntryGetStrides(boxman_to_entries[jj],
                                                  cs, matrix_type);
               hypre_BoxManEntryGetExtents(boxman_to_entries[jj],
                                           hypre_BoxIMin(map_box),
                                           hypre_BoxIMax(map_box));
               hypre_IntersectBoxes(to_box, map_box, int_box);

               hypre_CopyIndex(hypre_BoxIMin(int_box), index);
               hypre_SStructBoxManEntryGetGlobalRank(boxman_to_entries[jj],
                                                     index, &col_base, matrix_type);

               hypre_RefineBox(int_box, NULL, dom_stride);
               hypre_BoxShiftNeg(int_box, offset);

               hypre_CopyIndex(hypre_BoxIMin(int_box), mstart);
               hypre_SStructBoxManEntryGetGlobalRank(boxman_entries[ii],
                                                     mstart, &row_base, matrix_type);

               hypre_CopyBox(value_box, map_vbox);
               hypre_SStructMatrixMapDataBox(matrix, part, var, vars[entry], map_vbox);
               hypre_SStructMatrixMapDataBox(matrix, part, var, vars[entry], int_box);

               vstart = hypre_BoxIMin(int_box);
               hypre_BoxGetSize(int_box, loop_size);

#if defined(HYPRE_USING_GPU)
               if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
               {
                  //hypre_assert(ndim <= 3);

                  HYPRE_Int rs_0 = 0, rs_1 = 0, rs_2 = 0;
                  HYPRE_Int cs_0 = 0, cs_1 = 0, cs_2 = 0;

                  if (ndim > 0)
                  {
                     rs_0 = rs[0] * dom_stride[0];
                     cs_0 = cs[0];
                  }

                  if (ndim > 1)
                  {
                     rs_1 = rs[1] * dom_stride[1];
                     cs_1 = cs[1];
                  }

                  if (ndim > 2)
                  {
                     rs_2 = rs[2] * dom_stride[2];
                     cs_2 = cs[2];
                  }

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(ncols,rows,cols,ijvalues,values)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      box, mstart, dom_stride, mi,
                                      map_vbox, vstart, unit_stride, vi);
                  {
                     hypre_Index loop_index;
                     HYPRE_Int   ci;

                     hypre_BoxLoopGetIndex(loop_index);

                     ci = mi * nentries + ncols[mi];
                     rows[mi] = row_base;
                     cols[ci] = col_base;

                     if (ndim > 0)
                     {
                        rows[mi] += loop_index[0] * rs_0;
                        cols[ci] += loop_index[0] * cs_0;
                     }

                     if (ndim > 1)
                     {
                        rows[mi] += loop_index[1] * rs_1;
                        cols[ci] += loop_index[1] * cs_1;
                     }

                     if (ndim > 2)
                     {
                        rows[mi] += loop_index[2] * rs_2;
                        cols[ci] += loop_index[2] * cs_2;
                     }

                     ijvalues[ci] = values[ei + vi * nentries];
                     values_map[ei + vi * nentries] = ci;
                     ncols[mi]++;
                  }
                  hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
               }
               else
#endif
               {
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      box, mstart, dom_stride, mi,
                                      map_vbox, vstart, unit_stride, vi);
                  {
                     hypre_Index loop_index;
                     HYPRE_Int   ci, d;

                     hypre_BoxLoopGetIndex(loop_index);
                     //hypre_assert((mi >= 0) && (vi >= 0));

                     ci = mi * nentries + ncols[mi];
                     rows[mi] = row_base;
                     cols[ci] = col_base;
                     for (d = 0; d < ndim; d++)
                     {
                        rows[mi] += loop_index[d] * rs[d] * dom_stride[d];
                        cols[ci] += loop_index[d] * cs[d];
                     }
                     /* WM: todo - if doing a get, don't need to manipulate ijvalues here, right?
                      *            likewise, if doing a set, don't need the values_map */
                     ijvalues[ci] = values[ei + vi * nentries];
                     values_map[ei + vi * nentries] = ci;
                     ncols[mi]++;
                  }
                  hypre_BoxLoop2End(mi, vi);
               }
            } /* end loop through boxman to entries */

            hypre_TFree(boxman_to_entries, HYPRE_MEMORY_HOST);

         } /* end of ei nentries loop */

         if (action > 0)
         {
            HYPRE_IJMatrixAddToValues2(ijmatrix, nrows, ncols,
                                       (const HYPRE_BigInt *) rows,
                                       (const HYPRE_Int *) row_indexes,
                                       (const HYPRE_BigInt *) cols,
                                       (const HYPRE_Complex *) ijvalues);
         }
         else if (action > -1)
         {
            HYPRE_IJMatrixSetValues2(ijmatrix, nrows, ncols,
                                     (const HYPRE_BigInt *) rows,
                                     (const HYPRE_Int *) row_indexes,
                                     (const HYPRE_BigInt *) cols,
                                     (const HYPRE_Complex *) ijvalues);
         }
         else
         {
            /* TODO (VPM): This block is causing random issues with GPUs */
#if !defined(HYPRE_USING_GPU)
            if (action == -2)
            {
               /* Zero out entries gotten */
               HYPRE_IJMatrixGetValuesAndZeroOut(ijmatrix, nrows, ncols,
                                                 rows, row_indexes, cols, ijvalues);
            }
            else
            {
               HYPRE_IJMatrixGetValues2(ijmatrix, nrows, ncols,
                                        rows, row_indexes, cols, ijvalues);
            }
#endif
         }

      } /* end loop through boxman entries */

      /* WM: do backwards mapping from ijvalues to values if doing a get */
      /* WM: todo - put this in a boxloop to avoid unnecessary copies? */
      nrows = hypre_BoxVolume(set_box);
      if (action < 0)
      {
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
            HYPRE_THRUST_CALL( for_each,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(num_nonzeros),
                               [ = ] __device__ (HYPRE_Int i)
            {
               if (values_map[i] >= 0)
               {
                  values[i] = ijvalues[values_map[i]];
               }
            });
#elif defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( for_each,
                               oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                               oneapi::dpl::counting_iterator<HYPRE_Int>(num_nonzeros),
                               [ = ] (HYPRE_Int i)
            {
               if (values_map[i] >= 0)
               {
                  values[i] = ijvalues[values_map[i]];
               }
            });
#endif
         }
         else
#endif
         {
            for (i = 0; i < num_nonzeros; i++)
            {
               if (values_map[i] >= 0)
               {
                  values[i] = ijvalues[values_map[i]];
               }
            }
         }
      }

      hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);

      hypre_TFree(ncols, memory_location);
      hypre_TFree(rows, memory_location);
      hypre_TFree(row_indexes, memory_location);
      hypre_TFree(cols, memory_location);
      hypre_TFree(ijvalues, memory_location);
      hypre_TFree(values_map, memory_location);

      hypre_BoxDestroy(box);
      hypre_BoxDestroy(to_box);
      hypre_BoxDestroy(map_box);
      hypre_BoxDestroy(int_box);
      hypre_BoxDestroy(map_vbox);

#if defined(DEBUG_SETBOX)
      hypre_printf("%s: num_nonzeros: %d\n", __func__, num_nonzeros);
#endif
   }

   /*------------------------------------------
    * non-stencil entries
    *------------------------------------------*/

   else
   {
      /* RDF: THREAD (Check safety on UMatrixSetValues call) */
      hypre_BoxGetSize(set_box, loop_size);
      hypre_SerialBoxLoop0Begin(ndim, loop_size);
      {
         zypre_BoxLoopGetIndex(index);
         hypre_AddIndexes(index, hypre_BoxIMin(set_box), ndim, index);
         hypre_SStructUMatrixSetValues(matrix, part, index, var,
                                       nentries, entries, values, action);
         values += nentries;
      }
      hypre_SerialBoxLoop0End();
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixSetBoxValues( hypre_SStructMatrix *matrix,
                                  HYPRE_Int            part,
                                  hypre_Box           *set_box,
                                  HYPRE_Int            var,
                                  HYPRE_Int            nentries,
                                  HYPRE_Int           *entries,
                                  hypre_Box           *value_box,
                                  HYPRE_Complex       *values,
                                  HYPRE_Int            action )
{
   HYPRE_IJMatrix  ijmatrix = hypre_SStructMatrixIJMatrix(matrix);

   hypre_SStructUMatrixSetBoxValuesHelper(matrix, part, set_box, var,
                                          nentries, entries, value_box, values,
                                          action, ijmatrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixAssemble( hypre_SStructMatrix *matrix )
{
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   HYPRE_IJMatrixAssemble(ijmatrix);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * Maps map_vbox in place to the index space where data is stored for S(vi,vj)
 *
 * Note: Since off-diagonal components of the SStructMatrix are being stored
 *       in the UMatrix, this function does not change map_vbox when vi != vj
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixMapDataBox( hypre_SStructMatrix  *matrix,
                               HYPRE_Int             part,
                               HYPRE_Int             vi,
                               HYPRE_Int             vj,
                               hypre_Box            *map_vbox )
{
   HYPRE_Int             matrix_type = hypre_SStructMatrixObjectType(matrix);
   hypre_SStructPMatrix *pmatrix;
   hypre_StructMatrix   *smatrix;

   if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
      if (vi == vj)
      {
         hypre_StructMatrixMapDataBox(smatrix, map_vbox);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixRef( hypre_SStructMatrix  *matrix,
                        hypre_SStructMatrix **matrix_ref )
{
   hypre_SStructMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixSplitEntries( hypre_SStructMatrix *matrix,
                                 HYPRE_Int            part,
                                 HYPRE_Int            var,
                                 HYPRE_Int            nentries,
                                 HYPRE_Int           *entries,
                                 HYPRE_Int           *nSentries_ptr,
                                 HYPRE_Int          **Sentries_ptr,
                                 HYPRE_Int           *nUentries_ptr,
                                 HYPRE_Int          **Uentries_ptr )
{
   hypre_SStructGraph   *graph   = hypre_SStructMatrixGraph(matrix);
   HYPRE_Int            *split   = hypre_SStructMatrixSplit(matrix, part, var);
   hypre_SStructStencil *stencil = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int             entry;
   HYPRE_Int             i;

   HYPRE_Int             nSentries = 0;
   HYPRE_Int            *Sentries  = hypre_SStructMatrixSEntries(matrix);
   HYPRE_Int             nUentries = 0;
   HYPRE_Int            *Uentries  = hypre_SStructMatrixUEntries(matrix);

   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];
      if (entry < hypre_SStructStencilSize(stencil))
      {
         /* stencil entries */
         if (split[entry] > -1)
         {
            Sentries[nSentries] = split[entry];
            nSentries++;
         }
         else
         {
            Uentries[nUentries] = entry;
            nUentries++;
         }
      }
      else
      {
         /* non-stencil entries */
         Uentries[nUentries] = entry;
         nUentries++;
      }
   }

   *nSentries_ptr = nSentries;
   *Sentries_ptr  = Sentries;
   *nUentries_ptr = nUentries;
   *Uentries_ptr  = Uentries;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixSetValues( HYPRE_SStructMatrix  matrix,
                              HYPRE_Int            part,
                              HYPRE_Int           *index,
                              HYPRE_Int            var,
                              HYPRE_Int            nentries,
                              HYPRE_Int           *entries,
                              HYPRE_Complex       *values,
                              HYPRE_Int            action )
{
   HYPRE_Int             ndim  = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph   *graph = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid  = hypre_SStructGraphGrid(graph);
   HYPRE_Int           **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   HYPRE_Int            *Sentries;
   HYPRE_Int            *Uentries;
   HYPRE_Int             nSentries;
   HYPRE_Int             nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cindex;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(index, ndim, cindex);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetValues(pmatrix, cindex, var,
                                    nSentries, Sentries, values, action);
      /* put inter-part couplings in UMatrix and zero them out in PMatrix
       * (possibly in ghost zones) */
      if (nvneighbors[part][var] > 0)
      {
         hypre_Box  *set_box;
         HYPRE_Int   d;
         /* This creates boxes with zeroed-out extents */
         set_box = hypre_BoxCreate(ndim);
         for (d = 0; d < ndim; d++)
         {
            hypre_BoxIMinD(set_box, d) = cindex[d];
            hypre_BoxIMaxD(set_box, d) = cindex[d];
         }
         hypre_SStructMatrixSetInterPartValues(matrix, part, set_box, var, nSentries, entries,
                                               set_box, values, action);
         hypre_BoxDestroy(set_box);
      }
   }

   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, action);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixSetBoxValues( HYPRE_SStructMatrix  matrix,
                                 HYPRE_Int            part,
                                 hypre_Box           *set_box,
                                 HYPRE_Int            var,
                                 HYPRE_Int            nentries,
                                 HYPRE_Int           *entries,
                                 hypre_Box           *value_box,
                                 HYPRE_Complex       *values,
                                 HYPRE_Int            action )
{
   hypre_SStructGraph      *graph = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid       *grid  = hypre_SStructGraphGrid(graph);
   HYPRE_Int              **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   HYPRE_Int               *Sentries;
   HYPRE_Int               *Uentries;
   HYPRE_Int                nSentries;
   HYPRE_Int                nUentries;
   hypre_SStructPMatrix    *pmatrix;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetBoxValues(pmatrix, set_box, var, nSentries, Sentries,
                                       value_box, values, action);

      /* put inter-part couplings in UMatrix and zero them out in PMatrix
       * (possibly in ghost zones) */
      if (nvneighbors[part][var] > 0)
      {
         hypre_SStructMatrixSetInterPartValues(matrix, part, set_box, var, nSentries, entries,
                                               value_box, values, action);
      }
   }

   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetBoxValues(matrix, part, set_box, var, nUentries, Uentries,
                                       value_box, values, action);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Put inter-part couplings in UMatrix and zero them out in PMatrix (possibly in
 * ghost zones).  Assumes that all entries are stencil entries.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixSetInterPartValues( HYPRE_SStructMatrix  matrix,
                                       HYPRE_Int            part,
                                       hypre_Box           *set_box,
                                       HYPRE_Int            var,
                                       HYPRE_Int            nentries,
                                       HYPRE_Int           *entries,
                                       hypre_Box           *value_box,
                                       HYPRE_Complex       *values,
                                       HYPRE_Int            action )
{
   HYPRE_Int                ndim       = hypre_SStructMatrixNDim(matrix);
   hypre_IJMatrix          *ij_matrix  = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph      *graph      = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid       *grid       = hypre_SStructGraphGrid(graph);
   hypre_SStructPMatrix    *pmatrix    = hypre_SStructMatrixPMatrix(matrix, part);
   hypre_SStructPGrid      *pgrid      = hypre_SStructPMatrixPGrid(pmatrix);
   hypre_BoxArrayArray     *pbnd_boxaa = hypre_SStructPGridPBndBoxArrayArray(pgrid, var);
   HYPRE_Int               *smap       = hypre_SStructPMatrixSMap(pmatrix, var);
   hypre_SStructVariable    frvartype  = hypre_SStructPGridVarType(pgrid, var);
   hypre_SStructStencil    *stencil    = hypre_SStructPMatrixStencil(pmatrix, var);
   hypre_Index             *shape      = hypre_SStructStencilShape(stencil);
   HYPRE_Int               *vars       = hypre_SStructStencilVars(stencil);
   HYPRE_MemoryLocation     memloc     = hypre_IJMatrixMemoryLocation(ij_matrix);

   hypre_SStructVariable    tovartype;
   hypre_StructMatrix      *smatrix;
   hypre_StructGrid        *sgrid;
   hypre_BoxArray          *pbnd_boxa;
   hypre_BoxArray          *grid_boxes;
   hypre_Box               *grid_box, *box, *ibox0, *ibox1, *ibox2, *tobox, *frbox;
   hypre_Index              ustride, loop_size;
   hypre_IndexRef           offset, start;
   hypre_BoxManEntry      **frentries, **toentries;
   hypre_SStructBoxManInfo *frinfo, *toinfo;
   HYPRE_Complex           *tvalues;
   HYPRE_Int                box_id;
   HYPRE_Int                nfrentries, ntoentries, frpart, topart;
   HYPRE_Int                entry, sentry, ei, fri, toi, i;
   HYPRE_Int                volume, tvalues_size = 16384;

   hypre_SetIndex(ustride, 1);
   box   = hypre_BoxCreate(ndim);
   ibox0 = hypre_BoxCreate(ndim);
   ibox1 = hypre_BoxCreate(ndim);
   ibox2 = hypre_BoxCreate(ndim);
   tobox = hypre_BoxCreate(ndim);
   frbox = hypre_BoxCreate(ndim);

   /* Allocate memory */
   tvalues = hypre_TAlloc(HYPRE_Complex, tvalues_size, memloc);

   for (ei = 0; ei < nentries; ei++)
   {
      entry  = entries[ei];
      sentry = smap[entry];
      offset = shape[entry];
      tovartype = hypre_SStructPGridVarType(pgrid, vars[entry]);
      smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entry]);
      sgrid = hypre_StructMatrixGrid(smatrix);
      grid_boxes = hypre_StructGridBoxes(sgrid);

      /* shift box in the stencil offset direction */
      hypre_CopyBox(set_box, box);

      hypre_AddIndexes(hypre_BoxIMin(box), offset, ndim, hypre_BoxIMin(box));
      hypre_AddIndexes(hypre_BoxIMax(box), offset, ndim, hypre_BoxIMax(box));

      /* get "to" entries */
      hypre_SStructGridIntersect(grid, part, vars[entry], box, -1,
                                 &toentries, &ntoentries);

      for (toi = 0; toi < ntoentries; toi++)
      {
         hypre_BoxManEntryGetExtents(toentries[toi],
                                     hypre_BoxIMin(tobox),
                                     hypre_BoxIMax(tobox));
         hypre_IntersectBoxes(box, tobox, ibox0);
         if (hypre_BoxVolume(ibox0))
         {
            hypre_SStructBoxManEntryGetPart(toentries[toi], part, &topart);

            /* shift ibox0 back */
            hypre_SubtractIndexes(hypre_BoxIMin(ibox0), offset, ndim,
                                  hypre_BoxIMin(ibox0));
            hypre_SubtractIndexes(hypre_BoxIMax(ibox0), offset, ndim,
                                  hypre_BoxIMax(ibox0));

            /* get "from" entries */
            hypre_SStructGridIntersect(grid, part, var, ibox0, -1,
                                       &frentries, &nfrentries);
            for (fri = 0; fri < nfrentries; fri++)
            {
               /* don't set couplings within the same part unless possibly for
                * cell data (to simplify periodic conditions for users) */
               hypre_SStructBoxManEntryGetPart(frentries[fri], part, &frpart);
               if (topart == frpart)
               {
                  if ( (frvartype != HYPRE_SSTRUCT_VARIABLE_CELL) ||
                       (tovartype != HYPRE_SSTRUCT_VARIABLE_CELL) )
                  {
                     continue;
                  }

                  hypre_BoxManEntryGetInfo(frentries[fri], (void **) &frinfo);
                  hypre_BoxManEntryGetInfo(toentries[toi], (void **) &toinfo);
                  if ( hypre_SStructBoxManInfoType(frinfo) ==
                       hypre_SStructBoxManInfoType(toinfo) )
                  {
                     continue;
                  }
               }

               hypre_BoxManEntryGetExtents(frentries[fri],
                                           hypre_BoxIMin(frbox),
                                           hypre_BoxIMax(frbox));
               hypre_IntersectBoxes(ibox0, frbox, ibox1);
               if (hypre_BoxVolume(ibox1))
               {
                  volume = hypre_BoxVolume(ibox1);
                  if (tvalues_size < volume)
                  {
                     tvalues = hypre_TReAlloc_v2(tvalues, HYPRE_Complex,
                                                 tvalues_size, HYPRE_Complex,
                                                 volume, memloc);
                     tvalues_size = volume;
                  }

                  if (action >= 0)
                  {
                     /* Update list of part boundaries */
                     hypre_ForBoxI(i, grid_boxes)
                     {
                        box_id = hypre_StructGridID(sgrid, i);
                        grid_box = hypre_BoxArrayBox(grid_boxes, i);
                        hypre_IntersectBoxes(grid_box, ibox1, ibox2);

                        pbnd_boxa = hypre_BoxArrayArrayBoxArray(pbnd_boxaa, box_id);
                        hypre_AppendBox(ibox2, pbnd_boxa);
                     }

                     /* set or add */
                     /* copy values into tvalues */
                     start = hypre_BoxIMin(ibox1);
                     hypre_BoxGetSize(ibox1, loop_size);
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         ibox1, start, ustride, mi,
                                         value_box, start, ustride, vi);
                     {
                        tvalues[mi] = values[ei + vi * nentries];
                     }
                     hypre_BoxLoop2End(mi, vi);

                     /* put values into UMatrix */
                     hypre_SStructUMatrixSetBoxValues(matrix, part, ibox1, var, 1,
                                                      &entry, ibox1, tvalues, action);

                     /* zero out values in PMatrix (possibly in ghost) */
                     hypre_StructMatrixClearBoxValues(smatrix, ibox1, 1, &sentry, -1, 1);
                  }
                  else
                  {
                     /* get values from UMatrix */
                     hypre_SStructUMatrixSetBoxValues(matrix, part, ibox1, var, 1,
                                                      &entry, ibox1, tvalues, action);

                     /* copy tvalues into values */
                     start = hypre_BoxIMin(ibox1);
                     hypre_BoxGetSize(ibox1, loop_size);
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         ibox1, start, ustride, mi,
                                         value_box, start, ustride, vi);
                     {
                        values[ei + vi * nentries] = tvalues[mi];
                     }
                     hypre_BoxLoop2End(mi, vi);

                  } /* end if action */
               } /* end if nonzero ibox1 */
            } /* end of "from" boxman entries loop */

            hypre_TFree(frentries, HYPRE_MEMORY_HOST);
         } /* end if nonzero ibox0 */
      } /* end of "to" boxman entries loop */

      hypre_TFree(toentries, HYPRE_MEMORY_HOST);
   } /* end of entries loop */

   /* Free memory */
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(ibox0);
   hypre_BoxDestroy(ibox1);
   hypre_BoxDestroy(ibox2);
   hypre_BoxDestroy(tobox);
   hypre_BoxDestroy(frbox);
   hypre_TFree(tvalues, memloc);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Add to or set (overwrite) entries in S with entries from U where
 * the sparsity pattern permits.
 *
 * (action > 0): add-to values
 * (action = 0): set values
 *
 * WM: TODO - what if there are constant stencil entries?
 * Not sure what the expected behavior should be. For now, avoid this case.
 * WM: TODO - does this potentially screw up pre-existing communication packages?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixCompressUToS( HYPRE_SStructMatrix A,
                                 HYPRE_Int           action )
{
   HYPRE_Int                nparts       = hypre_SStructMatrixNParts(A);
   HYPRE_Int               *Sentries     = hypre_SStructMatrixSEntries(A);
   HYPRE_SStructGraph       graph        = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid       *grid         = hypre_SStructGraphGrid(graph);
   HYPRE_Int              **nvneighbors  = hypre_SStructGridNVNeighbors(grid);
   HYPRE_Int                ndim         = hypre_SStructGridNDim(grid);

   hypre_ParCSRMatrix      *A_u          = hypre_SStructMatrixParCSRMatrix(A);
   hypre_CSRMatrix         *A_ud         = hypre_ParCSRMatrixDiag(A_u);
   hypre_CSRMatrix         *A_uo         = hypre_ParCSRMatrixOffd(A_u);
   HYPRE_Int                num_rows     = hypre_CSRMatrixNumRows(A_ud);

   hypre_SStructPMatrix    *pmatrix;
   hypre_StructMatrix      *smatrix;
   hypre_StructGrid        *sgrid;
   hypre_SStructStencil    *stencil;
   HYPRE_Int               *split;
   hypre_Index              start, stride, loop_size;
   hypre_BoxArray          *grid_boxes;
   hypre_Box               *grid_box;
   HYPRE_Int               *num_ghost;
   HYPRE_Int                i, j, offset, volume, var, entry, part, nvars, nSentries, num_indices;
   HYPRE_Real               threshold = 0.9;
   HYPRE_Int               *indices[3]       = {NULL, NULL, NULL};
   HYPRE_Int               *indices_0        = NULL;
   HYPRE_Int               *indices_1        = NULL;
   HYPRE_Int               *indices_2        = NULL;
   hypre_BoxArray          *indices_boxa     = NULL;
   HYPRE_Int                size;
   HYPRE_Complex           *values;

#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation     memory_location  = hypre_SStructMatrixMemoryLocation(A);
   HYPRE_ExecutionPolicy    exec             = hypre_GetExecPolicy1(memory_location);

   HYPRE_Int                max_num_rownnz;
   HYPRE_Int               *nonzero_rows     = NULL;
   HYPRE_Int               *nonzero_rows_end = NULL;
   HYPRE_Int               *all_indices_0    = NULL;
   HYPRE_Int               *all_indices_1    = NULL;
   HYPRE_Int               *all_indices_2    = NULL;
   HYPRE_Int               *box_nnzrows      = NULL;
   HYPRE_Int               *box_nnzrows_end  = NULL;
#endif

#if defined(DEBUG_U2S)
   char                     msg[128];
#endif

   /* Return in the case of a trivial unstructured component */
   /* WM: TODO - safe to return based on local info? That is, no collective calls below? */
   if (!(hypre_CSRMatrixNumNonzeros(A_ud) + hypre_CSRMatrixNumNonzeros(A_uo)))
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructMatrixCompressUToS");

   /* Create work variables */
   grid_box = hypre_BoxCreate(ndim);

#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
#if defined(HYPRE_USING_SYCL)
      /* WM: todo - sycl */
#else
      max_num_rownnz = hypre_min(num_rows,
                                 hypre_CSRMatrixNumRownnz(A_ud) +
                                 hypre_CSRMatrixNumRownnz(A_uo));

      if (hypre_CSRMatrixRownnz(A_ud) && hypre_CSRMatrixRownnz(A_uo))
      {
         nonzero_rows = hypre_TAlloc(HYPRE_Int, max_num_rownnz, HYPRE_MEMORY_DEVICE);
         HYPRE_THRUST_CALL( merge,
                            hypre_CSRMatrixRownnz(A_ud),
                            hypre_CSRMatrixRownnz(A_ud) + hypre_CSRMatrixNumRownnz(A_ud),
                            hypre_CSRMatrixRownnz(A_uo),
                            hypre_CSRMatrixRownnz(A_uo) + hypre_CSRMatrixNumRownnz(A_uo),
                            nonzero_rows );
      }
      else if (hypre_CSRMatrixRownnz(A_ud))
      {
         nonzero_rows = hypre_CSRMatrixRownnz(A_ud);
      }

      if (nonzero_rows)
      {
         nonzero_rows_end = HYPRE_THRUST_CALL(unique,
                                              nonzero_rows,
                                              nonzero_rows + max_num_rownnz);
      }
      else
      {
         nonzero_rows = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         HYPRE_THRUST_CALL( sequence, nonzero_rows, nonzero_rows + num_rows );
         nonzero_rows_end = nonzero_rows + num_rows;
      }
#endif
   }
#endif

   /* Set work arrays */
#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      if (ndim > 0)
      {
         all_indices_0 = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         indices_0     = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         indices[0]    = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
      }

      if (ndim > 1)
      {
         all_indices_1 = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         indices_1     = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         indices[1]    = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
      }

      if (ndim > 2)
      {
         all_indices_2 = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         indices_2     = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
         indices[2]    = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
      }

      box_nnzrows = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
   }
   else
#endif
   {
      if (ndim > 0) { indices_0 = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST); }
      if (ndim > 1) { indices_1 = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST); }
      if (ndim > 2) { indices_2 = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST); }
   }

   /* Set entries of ij_Ahat */
   offset = 0;
   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(A, part);
      nvars   = hypre_SStructPMatrixNVars(pmatrix);

      for (var = 0; var < nvars; var++)
      {
         split    = hypre_SStructMatrixSplit(A, part, var);
         smatrix  = hypre_SStructPMatrixSMatrix(pmatrix, var, var);
         stencil  = hypre_SStructGraphStencil(graph, part, var);
         sgrid    = hypre_StructMatrixGrid(smatrix);

         nSentries = 0;
         for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] > -1)
            {
               Sentries[nSentries] = split[entry];
               nSentries++;
            }
         }

         grid_boxes = hypre_StructGridBoxes(sgrid);

         /* Loop over boxes */
         hypre_ForBoxI(i, grid_boxes)
         {
            /* WM: todo - I'm using the struct grid box grown by
               num_ghosts instead of the matrix data space again here */
            hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, i), grid_box);
            num_ghost = hypre_StructGridNumGhost(sgrid);
            hypre_BoxGrowByArray(grid_box, num_ghost);
            hypre_BoxGetSize(grid_box, loop_size);
            volume = hypre_BoxVolume(grid_box);
            hypre_SetIndex(stride, 1);
            hypre_CopyToIndex(hypre_BoxIMin(grid_box), ndim, start);

#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               /* Get ALL the indices */
               hypre_BoxLoop1Begin(ndim, loop_size, grid_box, start, stride, ii);
               {
                  hypre_Index index;
                  hypre_BoxLoopGetIndex(index);
                  if (ndim > 0)
                  {
                     all_indices_0[ii] = index[0] + start[0];
                  }
                  if (ndim > 1)
                  {
                     all_indices_1[ii] = index[1] + start[1];
                  }
                  if (ndim > 2)
                  {
                     all_indices_2[ii] = index[2] + start[2];
                  }
               }
               hypre_BoxLoop1End(ii);

#if defined(HYPRE_USING_SYCL)
               /* WM: todo - sycl */
#else
               /* Get the nonzero rows for this box */
               box_nnzrows_end = HYPRE_THRUST_CALL( copy_if,
                                                    nonzero_rows,
                                                    nonzero_rows_end,
                                                    box_nnzrows,
                                                    in_range<HYPRE_Int>(offset, offset + volume) );
               HYPRE_THRUST_CALL( transform,
                                  box_nnzrows,
                                  box_nnzrows_end,
                                  thrust::make_constant_iterator(offset),
                                  box_nnzrows,
                                  thrust::minus<HYPRE_Int>() );
               num_indices = box_nnzrows_end - box_nnzrows;

               /* Gather indices at non-zero rows of A_u */
               if (ndim > 0)
               {
                  HYPRE_THRUST_CALL( gather, box_nnzrows, box_nnzrows_end, all_indices_0, indices_0 );
               }

               if (ndim > 1)
               {
                  HYPRE_THRUST_CALL( gather, box_nnzrows, box_nnzrows_end, all_indices_1, indices_1 );
               }

               if (ndim > 2)
               {
                  HYPRE_THRUST_CALL( gather, box_nnzrows, box_nnzrows_end, all_indices_2, indices_2 );
               }
#endif // defined(HYPRE_USING_SYCL)
            }
            else
#endif // defined(HYPRE_USING_GPU)
            {
               num_indices = 0;
               hypre_BoxLoop1ReductionBeginHost(ndim, loop_size, grid_box, start, stride, ii, num_indices);
               {
                  if (hypre_CSRMatrixI(A_ud)[offset + ii + 1] -
                      hypre_CSRMatrixI(A_ud)[offset + ii] +
                      hypre_CSRMatrixI(A_uo)[offset + ii + 1] -
                      hypre_CSRMatrixI(A_uo)[offset + ii] > 0)
                  {
                     hypre_Index index;
                     hypre_BoxLoopGetIndexHost(index);
                     if (ndim > 0) { indices_0[num_indices] = index[0] + start[0]; }
                     if (ndim > 1) { indices_1[num_indices] = index[1] + start[1]; }
                     if (ndim > 2) { indices_2[num_indices] = index[2] + start[2]; }
                     num_indices++;
                  }
               }
               hypre_BoxLoop1ReductionEndHost(ii, num_indices);
            }

            /* WM: todo - these offsets for the unstructured indices only work
               with no inter-variable couplings? */
            offset += volume;

            /* WM: todo - make sure threshold is set such that
               there are no extra rows here! */
            if (num_indices)
            {
#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  if (ndim > 0) hypre_TMemcpy(indices[0], indices_0, HYPRE_Int, num_indices,
                                              HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
                  if (ndim > 1) hypre_TMemcpy(indices[1], indices_1, HYPRE_Int, num_indices,
                                              HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
                  if (ndim > 2) hypre_TMemcpy(indices[2], indices_2, HYPRE_Int, num_indices,
                                              HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
               }
               else
#endif
               {
                  indices[0] = indices_0;
                  indices[1] = indices_1;
                  indices[2] = indices_2;
               }

               /* Create array of boxes from set of indices */
               hypre_BoxArrayCreateFromIndices(ndim, num_indices, indices,
                                               threshold, &indices_boxa);
               hypre_ForBoxI(j, indices_boxa)
               {
#if defined (DEBUG_U2S)
                  hypre_sprintf(msg, "Part %d, Box %d - ", part, j);
                  hypre_BoxPrintDebug(msg, hypre_BoxArrayBox(indices_boxa, j));
#endif
                  size   = hypre_BoxVolume(hypre_BoxArrayBox(indices_boxa, j)) * nSentries;
                  values = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_DEVICE);

                  /* INIT values from the structured matrix if action = 0
                     (need current stencil values for entries that don't exist in U matrix) */
                  if (action == 0)
                  {
                     hypre_SStructPMatrixSetBoxValues(pmatrix,
                                                      hypre_BoxArrayBox(indices_boxa, j),
                                                      var, nSentries, Sentries,
                                                      hypre_BoxArrayBox(indices_boxa, j),
                                                      values, -1);
                  }

                  /* GET values from unstructured matrix */
                  /* WM: note - I'm passing the entire box here, so I expect to get back
                                ALL intra-part connections in A_u */
                  /* WM: question - What about inter-part connections? I hope that they are
                                    always excluded here? Double check this. */
                  hypre_SStructUMatrixSetBoxValues(A, part,
                                                   hypre_BoxArrayBox(indices_boxa, j),
                                                   var, nSentries, Sentries,
                                                   hypre_BoxArrayBox(indices_boxa, j),
                                                   values, -2);

                  /* ADD values to structured matrix */
                  /* WM: todo - just call to hypre_SStructMatrixSetBoxValues() instead of
                   * hypre_SStructPMatrixSetBoxValues() and hypre_SStructMatrixSetInterPartValues()? */
                  hypre_SStructPMatrixSetBoxValues(pmatrix,
                                                   hypre_BoxArrayBox(indices_boxa, j),
                                                   var, nSentries, Sentries,
                                                   hypre_BoxArrayBox(indices_boxa, j),
                                                   values, action);
                  if (nvneighbors[part][var] > 0)
                  {
                     hypre_SStructMatrixSetInterPartValues(A, part,
                                                           hypre_BoxArrayBox(indices_boxa, j),
                                                           var, nSentries, Sentries,
                                                           hypre_BoxArrayBox(indices_boxa, j),
                                                           values, 1);
                  }

                  /* Free memory */
                  hypre_TFree(values, HYPRE_MEMORY_DEVICE);
               }
               hypre_BoxArrayDestroy(indices_boxa);
               indices_boxa = NULL;
            }
         } /* Loop over boxes */
      } /* Loop over vars */
   } /* Loop over parts */
   hypre_BoxDestroy(grid_box);

   /* Free memory */
#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      if (nonzero_rows != hypre_CSRMatrixRownnz(A_ud))
      {
         hypre_TFree(nonzero_rows, HYPRE_MEMORY_DEVICE);
      }
      hypre_TFree(all_indices_0, HYPRE_MEMORY_DEVICE);
      hypre_TFree(all_indices_1, HYPRE_MEMORY_DEVICE);
      hypre_TFree(all_indices_2, HYPRE_MEMORY_DEVICE);
      hypre_TFree(indices_0, HYPRE_MEMORY_DEVICE);
      hypre_TFree(indices_1, HYPRE_MEMORY_DEVICE);
      hypre_TFree(indices_2, HYPRE_MEMORY_DEVICE);
      hypre_TFree(box_nnzrows, HYPRE_MEMORY_DEVICE);
      hypre_TFree(indices[0], HYPRE_MEMORY_HOST);
      hypre_TFree(indices[1], HYPRE_MEMORY_HOST);
      hypre_TFree(indices[2], HYPRE_MEMORY_HOST);
   }
   else
#endif
   {
      hypre_TFree(indices_0, HYPRE_MEMORY_HOST);
      hypre_TFree(indices_1, HYPRE_MEMORY_HOST);
      hypre_TFree(indices_2, HYPRE_MEMORY_HOST);
   }

   /* WM: TODO: insert a check here that ensures the matrix A doesn't change in the case of action > 0 */
   /*           what about if action = 0? Then A does change... is there some other way to check correctness? */

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Notes (VPM):
 *       1) Use part and var as arguments to this function?
 *       2) We are not converting the whole SStructMatrix, only the
 *          structured part. Change function's name?
 *       3) This converts only A(vi, vi). Need to expand to other variables.
 *--------------------------------------------------------------------------*/

hypre_IJMatrix *
hypre_SStructMatrixToUMatrix( HYPRE_SStructMatrix  matrix,
                              HYPRE_Int            fill_diagonal )
{
   MPI_Comm                 comm            = hypre_SStructMatrixComm(matrix);
   HYPRE_Int                ndim            = hypre_SStructMatrixNDim(matrix);
   HYPRE_Int                nparts          = hypre_SStructMatrixNParts(matrix);
   HYPRE_IJMatrix           ij_A            = hypre_SStructMatrixIJMatrix(matrix);
   HYPRE_MemoryLocation     memory_location = hypre_IJMatrixMemoryLocation(ij_A);
   hypre_SStructGraph      *graph           = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid       *grid            = hypre_SStructGraphGrid(graph);
   hypre_SStructPGrid      *pgrid;
   hypre_StructGrid        *sgrid;
   hypre_BoxArray          *grid_boxes;
   hypre_Box               *grid_box;
   HYPRE_Int                i, part, var, nvars, nrows;

   hypre_IJMatrix          *ij_Ahat = NULL;
   HYPRE_BigInt             sizes[4];
   HYPRE_Int               *ncols, *rowidx;
   HYPRE_BigInt            *rows, *cols;
   HYPRE_Complex           *values;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructMatrixToUMatrix");

   /* Set beggining/end of rows and columns that belong to this process */
   HYPRE_IJMatrixGetLocalRange(ij_A, &sizes[0], &sizes[1], &sizes[2], &sizes[3]);
   nrows = (HYPRE_Int) (sizes[1] - sizes[0] + 1);

   /* Set all diagonal entries to 1 */
   if (fill_diagonal)
   {
      /* Create and initialize ij_Ahat */
      HYPRE_IJMatrixCreate(comm, sizes[0], sizes[1], sizes[2], sizes[3], &ij_Ahat);
      HYPRE_IJMatrixSetObjectType(ij_Ahat, HYPRE_PARCSR);
      HYPRE_IJMatrixInitialize_v2(ij_Ahat, memory_location);

      ncols  = hypre_TAlloc(HYPRE_Int, nrows, memory_location);
      rows   = hypre_TAlloc(HYPRE_BigInt, nrows, memory_location);
      rowidx = hypre_TAlloc(HYPRE_Int, nrows, memory_location);
      cols   = hypre_TAlloc(HYPRE_BigInt, nrows, memory_location);
      values = hypre_TAlloc(HYPRE_Complex, nrows, memory_location);

#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      HYPRE_THRUST_CALL( fill, ncols, ncols + nrows, 1 );
      HYPRE_THRUST_CALL( fill, values, values + nrows, 1.0 );
      HYPRE_THRUST_CALL( sequence, rowidx, rowidx + nrows );
      HYPRE_THRUST_CALL( sequence, rows, rows + nrows, sizes[0] );
      HYPRE_THRUST_CALL( sequence, cols, cols + nrows, sizes[2] );

#elif defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( std::fill, ncols, ncols + nrows, 1 );
      HYPRE_ONEDPL_CALL( std::fill, values, values + nrows, 1.0 );
      hypreSycl_sequence( rowidx, rowidx + nrows, 0 );
      hypreSycl_sequence( rows, rows + nrows, sizes[0] );
      hypreSycl_sequence( cols, cols + nrows, sizes[2] );
#endif
#else
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         ncols[i]  = 1;
         rows[i]   = sizes[0] + i;
         cols[i]   = sizes[2] + i;
         rowidx[i] = i;
         values[i] = 1.0;
      }
#endif

      HYPRE_IJMatrixSetValues2(ij_Ahat, nrows, ncols,
                               (const HYPRE_BigInt *) rows,
                               (const HYPRE_Int *) rowidx,
                               (const HYPRE_BigInt *) cols,
                               (const HYPRE_Complex *) values);

      hypre_TFree(ncols, memory_location);
      hypre_TFree(rows, memory_location);
      hypre_TFree(rowidx, memory_location);
      hypre_TFree(cols, memory_location);
      hypre_TFree(values, memory_location);
   }

   hypre_BoxArray ***convert_boxa;
   convert_boxa = hypre_TAlloc(hypre_BoxArray **, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      convert_boxa[part] = hypre_CTAlloc(hypre_BoxArray *, nvars, HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         /* WM: question - what about connections between variables? Note the var, var arguments below */
         sgrid      = hypre_SStructPGridSGrid(pgrid, var);
         grid_boxes = hypre_StructGridBoxes(sgrid);
         convert_boxa[part][var] = hypre_BoxArrayCreate(0, ndim);
         hypre_ForBoxI(i, grid_boxes)
         {
            grid_box = hypre_BoxArrayBox(grid_boxes, i);
            hypre_AppendBox(grid_box, convert_boxa[part][var]);
         }
      }
   }
   hypre_SStructMatrixBoxesToUMatrix(matrix, grid, &ij_Ahat, convert_boxa);

   /* Free memory */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (var = 0; var < nvars; var++)
      {
         hypre_BoxArrayDestroy(convert_boxa[part][var]);
      }
      hypre_TFree(convert_boxa[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(convert_boxa, HYPRE_MEMORY_HOST);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return ij_Ahat;
}

/*--------------------------------------------------------------------------
 * Notes:
 *         *) Consider a single variable type for now.
 *         *) Input grid comes from the matrix we are multiplying to A
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixBoxesToUMatrix( hypre_SStructMatrix   *A,
                                   hypre_SStructGrid     *grid,
                                   hypre_IJMatrix       **ij_Ahat_ptr,
                                   hypre_BoxArray      ***convert_boxa)
{
   HYPRE_Int              ndim     = hypre_SStructMatrixNDim(A);
   HYPRE_Int              nparts   = hypre_SStructMatrixNParts(A);
   HYPRE_Int             *Sentries = hypre_SStructMatrixSEntries(A);
   HYPRE_IJMatrix         ij_A     = hypre_SStructMatrixIJMatrix(A);
   HYPRE_MemoryLocation   memory_location = hypre_IJMatrixMemoryLocation(ij_A);
   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;
   hypre_SStructStencil  *stencil;

   hypre_SStructPMatrix  *pA;
   hypre_IJMatrix        *ij_Ahat;
   hypre_AuxParCSRMatrix *aux_matrix;

   HYPRE_Int             *split;
   hypre_BoxArray        *grid_boxes;
   hypre_Box             *box = hypre_BoxCreate(ndim);
   hypre_Box             *grid_box;
   hypre_Box             *ghost_box = hypre_BoxCreate(ndim);
   hypre_Box             *convert_box;

   hypre_Index            stride;
   HYPRE_Int              nrows;
   HYPRE_Int              ncols;
   HYPRE_Int             *row_sizes = NULL;
   HYPRE_Complex         *values;

   HYPRE_BigInt           sizes[4];
   HYPRE_Int              entry, part, var, nvars;
   HYPRE_Int              nnzrow;
   HYPRE_Int              nvalues, i, j;
   HYPRE_Int             *num_ghost;
   HYPRE_Int              nSentries;
#if !defined(HYPRE_USING_GPU)
   HYPRE_Int              m = 0;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
#endif

#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
   char                   msg[512];
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructMatrixBoxesToUMatrix");

   /* Get row and column ranges */
   HYPRE_IJMatrixGetLocalRange(ij_A, &sizes[0], &sizes[1], &sizes[2], &sizes[3]);
   nrows = (HYPRE_Int) (sizes[1] - sizes[0] + 1);
   ncols = (HYPRE_Int) (sizes[3] - sizes[2] + 1);

   /* Set row sizes */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Set rowsizes");
   nvalues = 0;
   hypre_SetIndex(stride, 1);
#if !defined(HYPRE_USING_GPU)
   if (!*ij_Ahat_ptr)
   {
      row_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   }
#endif
   for (part = 0; part < nparts; part++)
   {
      pA    = hypre_SStructMatrixPMatrix(A, part);
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPMatrixNVars(pA);

      for (var = 0; var < nvars; var++)
      {
         sgrid   = hypre_SStructPGridSGrid(pgrid, var);
         split   = hypre_SStructMatrixSplit(A, part, var);
         stencil = hypre_SStructPMatrixStencil(pA, var);
         grid_boxes = hypre_StructGridBoxes(sgrid);
         num_ghost  = hypre_StructGridNumGhost(sgrid);

         nnzrow = hypre_SStructStencilSize(stencil);

         hypre_ForBoxI(i, grid_boxes)
         {
            grid_box = hypre_BoxArrayBox(grid_boxes, i);
            hypre_CopyBox(grid_box, ghost_box);
            hypre_BoxGrowByArray(ghost_box, num_ghost);

            hypre_ForBoxI(j, convert_boxa[part][var])
            {
               convert_box = hypre_BoxArrayBox(convert_boxa[part][var], j);

#if !defined(HYPRE_USING_GPU)
               if (!*ij_Ahat_ptr)
               {
                  /* WM: do I need to check whether there is a non-trivial
                         intersection, or is that handled automatically? */
                  hypre_IntersectBoxes(grid_box, convert_box, box);
                  start = hypre_BoxIMin(box);
                  hypre_BoxGetSize(box, loop_size);
                  hypre_BoxLoop1Begin(ndim, loop_size, ghost_box, start, stride, mi);
                  {
                     row_sizes[m + mi] = nnzrow;
                  }
                  hypre_BoxLoop1End(mi);
               }
#endif
               nvalues = hypre_max(nvalues, nnzrow * hypre_BoxVolume(convert_box));
            } /* Loop over convert_boxa[part][var] */

#if !defined(HYPRE_USING_GPU)
            m += hypre_BoxVolume(ghost_box);
#endif
         } /* Loop over grid_boxes */
      } /* Loop over vars */
   } /* Loop over parts */
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(ghost_box);
   HYPRE_ANNOTATE_REGION_END("%s", "Set rowsizes");

   /* Create and initialize ij_Ahat */
   if (*ij_Ahat_ptr)
   {
      ij_Ahat = *ij_Ahat_ptr;
   }
   else
   {
      HYPRE_ANNOTATE_REGION_BEGIN("%s", "Create Matrix");
      HYPRE_IJMatrixPartialClone(ij_A, &ij_Ahat);
      if (row_sizes)
      {
         hypre_AuxParCSRMatrixCreate(&aux_matrix, nrows, ncols, row_sizes);
         hypre_IJMatrixTranslator(ij_Ahat) = aux_matrix;
      }
      HYPRE_IJMatrixInitialize(ij_Ahat);
      HYPRE_ANNOTATE_REGION_END("%s", "Create Matrix");
   }

   /* Allocate memory */
   values = hypre_CTAlloc(HYPRE_Complex, nvalues, memory_location);

   /* Set entries of ij_Ahat */
   for (part = 0; part < nparts; part++)
   {
      pA    = hypre_SStructMatrixPMatrix(A, part);
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPMatrixNVars(pA);

      for (var = 0; var < nvars; var++)
      {
         split   = hypre_SStructMatrixSplit(A, part, var);
         stencil = hypre_SStructPMatrixStencil(pA, var);

         nSentries = 0;
         for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] > -1)
            {
               Sentries[nSentries] = split[entry];
               nSentries++;
            }
         }

         hypre_ForBoxI(i, convert_boxa[part][var])
         {
            convert_box = hypre_BoxArrayBox(convert_boxa[part][var], i);

            hypre_assert(hypre_BoxVolume(convert_box) > 0);
            hypre_assert(hypre_BoxVolume(convert_box) <= nvalues);

#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
            hypre_sprintf(msg, "Part %d - Box %d - ", part, i);
            hypre_BoxPrintDebug(convert_box);
            HYPRE_ANNOTATE_REGION_BEGIN("%s %d %s %d", "Get values part", part, "convert_box", i);
#endif
            /* GET values from this box */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Get entries");
            hypre_SStructPMatrixSetBoxValues(pA, convert_box, var,
                                             nSentries, Sentries, convert_box, values, -1);
            HYPRE_ANNOTATE_REGION_END("%s", "Get entries");

#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
            HYPRE_ANNOTATE_REGION_END("%s %d %s %d", "Get values part", part, "convert_box", i);
            HYPRE_ANNOTATE_REGION_BEGIN("%s %d %s %d", "Set values part", part, "convert_box", i);
#endif
            /* SET values to ij_Ahat */
            HYPRE_ANNOTATE_REGION_BEGIN("%s", "Set entries");
            hypre_SStructUMatrixSetBoxValuesHelper(A, part, convert_box,
                                                   var, nSentries, Sentries,
                                                   convert_box, values, 0, ij_Ahat);
            HYPRE_ANNOTATE_REGION_END("%s", "Set entries");
#if defined(HYPRE_DEBUG) && defined(DEBUG_MATCONV)
            HYPRE_ANNOTATE_REGION_END("%s %d %s %d", "Set values part", part, "convert_box", i);
#endif
         } /* Loop over convert_boxa */
      } /* Loop over vars */
   } /* Loop over parts */

   /* Free memory */
   hypre_TFree(values, memory_location);

   /* Assemble ij_A */
   HYPRE_IJMatrixAssemble(ij_Ahat);

   /* Set pointer to ij_Ahat */
   *ij_Ahat_ptr = ij_Ahat;

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Notes:
 *   *) Consider a single variable type for now.
 *   *) Input grid comes from the matrix we are multiplying to A
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixHaloToUMatrix( hypre_SStructMatrix   *A,
                                  hypre_SStructGrid     *grid,
                                  hypre_IJMatrix       **ij_Ahat_ptr,
                                  HYPRE_Int              halo_size)
{
   HYPRE_Int              ndim     = hypre_SStructMatrixNDim(A);
   HYPRE_Int              nparts   = hypre_SStructMatrixNParts(A);
   HYPRE_IJMatrix         ij_A     = hypre_SStructMatrixIJMatrix(A);
   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;

   hypre_BoxArrayArray ***convert_boxaa;
   hypre_BoxArrayArray   *pbnd_boxaa;
   hypre_BoxArray        *convert_boxa;
   hypre_BoxArray        *pbnd_boxa;
   hypre_BoxArray        *grid_boxes;
   hypre_Box             *box;
   hypre_Box             *grow_box;
   hypre_Box             *grid_box;
   hypre_Box             *convert_box;

   HYPRE_BigInt           sizes[4];
   HYPRE_Int              part, var, nvars;
   HYPRE_Int              i, j, k, kk;
   HYPRE_Int              num_boxes;
   HYPRE_Int              pbnd_boxaa_size;
   HYPRE_Int              convert_boxaa_size;
   HYPRE_Int              grid_box_id;
   HYPRE_Int              convert_box_id;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Get row and column ranges */
   HYPRE_IJMatrixGetLocalRange(ij_A, &sizes[0], &sizes[1], &sizes[2], &sizes[3]);

   /* Find boxes to be converted */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Find boxes");
   convert_box   = hypre_BoxCreate(ndim);
   grow_box      = hypre_BoxCreate(ndim);
   convert_boxaa = hypre_TAlloc(hypre_BoxArrayArray **, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      convert_boxaa[part] = hypre_TAlloc(hypre_BoxArrayArray *, nvars, HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrid, var);
         num_boxes  = hypre_StructGridNumBoxes(sgrid);
         grid_boxes = hypre_StructGridBoxes(sgrid);

         /* Exit this loop if there are no grid boxes */
         if (!num_boxes)
         {
            convert_boxaa[part][var] = hypre_BoxArrayArrayCreate(0, ndim);
            continue;
         }

         pbnd_boxaa = hypre_SStructPGridPBndBoxArrayArray(pgrid, var);
         pbnd_boxaa_size = hypre_BoxArrayArraySize(pbnd_boxaa);

         convert_boxaa_size = hypre_min(num_boxes, pbnd_boxaa_size) + 1;
         convert_boxaa[part][var] = hypre_BoxArrayArrayCreate(convert_boxaa_size, ndim);

         k = kk = 0;
         hypre_ForBoxArrayI(i, pbnd_boxaa)
         {
            pbnd_boxa      = hypre_BoxArrayArrayBoxArray(pbnd_boxaa, i);
            convert_box_id = hypre_BoxArrayArrayID(pbnd_boxaa, i);
            grid_box_id    = hypre_StructGridID(sgrid, k);
            convert_boxa   = hypre_BoxArrayArrayBoxArray(convert_boxaa[part][var], kk);

            /* Find matching box id */
            while (convert_box_id != grid_box_id)
            {
               k++;
               //hypre_assert(k < hypre_StructGridNumBoxes(sgrid));
               grid_box_id = hypre_StructGridID(sgrid, k);
            }
            grid_box = hypre_BoxArrayBox(grid_boxes, k);

            if (hypre_BoxArraySize(pbnd_boxa))
            {
               hypre_ForBoxI(j, pbnd_boxa)
               {
                  box = hypre_BoxArrayBox(pbnd_boxa, j);
                  hypre_CopyBox(box, grow_box);
                  hypre_BoxGrowByValue(grow_box, halo_size - 1);
                  hypre_IntersectBoxes(grow_box, grid_box, convert_box);

                  hypre_AppendBox(convert_box, convert_boxa);
               }

               /* Eliminate duplicated entries */
               hypre_UnionBoxes(convert_boxa);

               /* Update convert_boxaa */
               hypre_BoxArrayArrayID(convert_boxaa[part][var], kk) = convert_box_id;
               kk++;
            }
         } /* loop over grid_boxes */

         hypre_BoxArrayArraySize(convert_boxaa[part][var]) = kk;
      } /* loop over vars */
   } /* loop over parts */
   hypre_BoxDestroy(grow_box);
   hypre_BoxDestroy(convert_box);
   HYPRE_ANNOTATE_REGION_END("%s", "Find boxes");

   /* Flatten the convert_boxaa to convert_boxa_flattened (that is ArrayArray to Array). */
   hypre_BoxArray ***convert_boxaa_flattened = hypre_TAlloc(hypre_BoxArray **, nparts,
                                                            HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      convert_boxaa_flattened[part] = hypre_TAlloc(hypre_BoxArray *, nvars, HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         convert_boxaa_flattened[part][var] = hypre_BoxArrayCreate(0, ndim);
         hypre_ForBoxArrayI(i, convert_boxaa[part][var])
         {
            grid_boxes = hypre_BoxArrayArrayBoxArray(convert_boxaa[part][var], i);
            hypre_ForBoxI(j, grid_boxes)
            {
               box = hypre_BoxArrayBox(grid_boxes, j);
               hypre_AppendBox(box, convert_boxaa_flattened[part][var]);
            }
         }
      }
   }

   hypre_SStructMatrixBoxesToUMatrix(A, grid, ij_Ahat_ptr, convert_boxaa_flattened);

   /* Free memory */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (var = 0; var < nvars; var++)
      {
         hypre_BoxArrayArrayDestroy(convert_boxaa[part][var]);
         hypre_BoxArrayDestroy(convert_boxaa_flattened[part][var]);
      }
      hypre_TFree(convert_boxaa[part], HYPRE_MEMORY_HOST);
      hypre_TFree(convert_boxaa_flattened[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(convert_boxaa, HYPRE_MEMORY_HOST);
   hypre_TFree(convert_boxaa_flattened, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the diagonal of a SStructMatrix as a SStructVector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixGetDiagonal( hypre_SStructMatrix   *matrix,
                                hypre_SStructVector  **diag_ptr )
{
   MPI_Comm                comm   = hypre_SStructMatrixComm(matrix);
   hypre_SStructGraph     *graph  = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid      *grid   = hypre_SStructGraphGrid(graph);
   HYPRE_Int               nparts = hypre_SStructMatrixNParts(matrix);
   HYPRE_Int               object_type = hypre_SStructMatrixObjectType(matrix);

   hypre_SStructVector    *diag;
   hypre_ParVector        *par_d;
   hypre_SStructPMatrix   *pmatrix;
   hypre_SStructPVector   *pdiag;
   HYPRE_Int               part;

   hypre_ParCSRMatrix     *A;
   hypre_CSRMatrix        *A_diag;

   /* Create vector */
   HYPRE_SStructVectorCreate(comm, grid, &diag);
   HYPRE_SStructVectorInitialize(diag);
   HYPRE_SStructVectorAssemble(diag);

   /* Fill vector with the diagonal of the matrix */
   if (object_type == HYPRE_SSTRUCT || object_type == HYPRE_STRUCT)
   {
      for (part = 0; part < nparts; part++)
      {
         pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
         pdiag   = hypre_SStructVectorPVector(diag, part);

         hypre_SStructPMatrixGetDiagonal(pmatrix, pdiag);
      }
   }

   /* Unstructured diagonal component */
   if (object_type == HYPRE_SSTRUCT || object_type == HYPRE_PARCSR)
   {
      A = hypre_SStructMatrixParCSRMatrix(matrix);
      A_diag = hypre_ParCSRMatrixDiag(A);

      hypre_SStructVectorConvert(diag, &par_d);
      hypre_CSRMatrixExtractDiagonal(A_diag, hypre_ParVectorLocalData(par_d), 10);
      hypre_SStructVectorRestore(diag, par_d);
   }

   *diag_ptr = diag;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_MemoryLocation
hypre_SStructMatrixMemoryLocation(hypre_SStructMatrix *matrix)
{
   HYPRE_Int   type = hypre_SStructMatrixObjectType(matrix);
   void       *object;

   HYPRE_SStructMatrixGetObject(matrix, &object);

   if (type == HYPRE_SSTRUCT)
   {
      return hypre_ParCSRMatrixMemoryLocation(hypre_SStructMatrixParCSRMatrix(matrix));
   }

   if (type == HYPRE_PARCSR)
   {
      return hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix *) object);
   }

   if (type == HYPRE_STRUCT)
   {
      return hypre_StructMatrixMemoryLocation((hypre_StructMatrix *) object);
   }

   return HYPRE_MEMORY_UNDEFINED;
}
