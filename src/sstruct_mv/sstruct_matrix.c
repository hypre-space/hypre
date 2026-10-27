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

/* Forward declarations of static functions */
static HYPRE_Int hypre_SStructMatrixCompressUToSBox( hypre_SStructMatrix *A,
                                                     HYPRE_Int            action );
static HYPRE_Int hypre_SStructBoundaryGridMatches( hypre_SStructGrid *a,
                                                   hypre_SStructGrid *b );

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

   /* WM: todo - using vars[entries[0]] below assumes all the entries point to the same var */
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

   /* WM: todo - using vars[entries[0]] below assumes all the entries point to the same var */
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
   if (hypre_GetExecPolicy1(hypre_StructMatrixMemoryLocation(smatrix)) == HYPRE_EXEC_DEVICE)
   {
      hypre_SyncDevice();
   }
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

   /* if values already accumulated, just return */
   if (hypre_SStructPMatrixAccumulated(pmatrix))
   {
      return hypre_error_flag;
   }

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
            hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost, &comm_info);
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

   for (v = vstart; v < (vstart + vsize); v++)
   {
      for (t = tstart; t < (tstart + tsize); t++)
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
   HYPRE_BigInt           *iUventries    = hypre_SStructGraphIUVEntries(graph);
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
   HYPRE_Int               part, var, entry, b, m;
   HYPRE_BigInt            iUventry;
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
      iUventry = iUventries[entry];
      m = (HYPRE_Int) (hypre_SStructUVEntryRank(Uventries[iUventry]) - rowstart);
      if ((m > -1) && (m < nrows))
      {
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_HOST)
#endif
         {
            row_sizes[m] += hypre_SStructUVEntryNUEntries(Uventries[iUventry]);
         }
         max_size = hypre_max(max_size,
                              nnzrow + hypre_SStructUVEntryNUEntries(Uventries[iUventry]));
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
   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_SStructMatrixTmpRowCoordsDevice(matrix) = hypre_CTAlloc(HYPRE_BigInt,  max_size,
                                                                    HYPRE_MEMORY_DEVICE);
      hypre_SStructMatrixTmpColCoordsDevice(matrix) = hypre_CTAlloc(HYPRE_BigInt,  max_size,
                                                                    HYPRE_MEMORY_DEVICE);
      hypre_SStructMatrixTmpCoeffsDevice(matrix)    = hypre_CTAlloc(HYPRE_Complex, max_size,
                                                                    HYPRE_MEMORY_DEVICE);
   }
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

            coeffs[ncoeffs] = h_values[i];
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

      hypre_BigIntFillnDevice(hypre_SStructMatrixTmpRowCoordsDevice(matrix), ncoeffs, row_coord);

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

#if defined(HYPRE_USING_GPU)
/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixSetArrayValuesDevice( hypre_SStructMatrix *matrix,
                                          HYPRE_Int            part,
                                          HYPRE_Int            var,
                                          HYPRE_Int            nvalues,
                                          HYPRE_Int           *indexes,
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

   hypre_IndexRef           offset;
   hypre_Index              index;
   hypre_Index              to_index;
   hypre_SStructUVEntry    *Uventry;
   hypre_BoxManEntry       *boxman_entry;
   hypre_SStructBoxManInfo *entry_info;
   HYPRE_Int                i, d, entry;
   HYPRE_Int                nvalues_set = 0;
   HYPRE_Complex           *values_set;
   HYPRE_BigInt             Uverank;

   HYPRE_Int               *skip_val;
   HYPRE_Int               *skip_val_h;
   HYPRE_Int               *indexes_h;
   HYPRE_BigInt            *rows_h;
   HYPRE_BigInt            *cols_h;
   HYPRE_Int               *scatter_locations = NULL;

   HYPRE_Int                nrows;
   HYPRE_Int               *ncols;
   HYPRE_BigInt            *rows = hypre_TAlloc(HYPRE_BigInt, nvalues, HYPRE_MEMORY_DEVICE);
   HYPRE_BigInt            *cols = hypre_TAlloc(HYPRE_BigInt, nvalues, HYPRE_MEMORY_DEVICE);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* WM: todo - for now, I am copying the indexes to the host, then getting all
    *            the row/col info on the host, then copying back to the device
    *            a more complete GPU port may be needed if depending on performance
    *            if there are many Uentries to set (though this should be small
    *            compared to the struct component of the matrix) */

   skip_val = hypre_CTAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE);
   skip_val_h = hypre_CTAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_HOST);
   indexes_h = hypre_TAlloc(HYPRE_Int, ndim * nvalues, HYPRE_MEMORY_HOST);
   rows_h = hypre_TAlloc(HYPRE_BigInt, nvalues, HYPRE_MEMORY_HOST);
   cols_h = hypre_TAlloc(HYPRE_BigInt, nvalues, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(indexes_h, indexes, HYPRE_Int, ndim * nvalues, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);

   /* Loop over indexes/entries and get row/col for IJ matrix */
   for (i = 0; i < nvalues; i++)
   {
      for (d = 0; d < ndim; d++)
      {
         index[d] = indexes_h[i * ndim + d];
      }

      hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);

      /* if not local, check neighbors */
      if (boxman_entry == NULL)
      {
         hypre_SStructGridFindNborBoxManEntry(grid, part, index, var, &boxman_entry);
      }

      /* WM: todo - note that I am not allowing the user to pass bad rows */
      if (boxman_entry == NULL)
      {
         hypre_error_in_arg(1);
         hypre_error_in_arg(2);
         hypre_error_in_arg(4);

         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }
      else
      {
         hypre_BoxManEntryGetInfo(boxman_entry, (void **) &entry_info);
      }

      hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index,
                                            &(rows_h[nvalues_set]), matrix_type);

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

         if (boxman_entry == NULL)
         {
            /* WM: note - I am allowing for col indices that reach outside the grid. These will be skipped.
             *            This can make it easier for the user at boundary conditions. That is,
             *            when looping over the box, the stencils will look different at the boundary, but they
             *            can just pass the same stencil as the interior, and we will skip the invalid entries here.
             *            But this also silently skips potentially "bad" arguments from the user... maybe revisit. */
            skip_val_h[i] = 1;
         }
         else
         {
            hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, to_index,
                                                  &(cols_h[nvalues_set++]), matrix_type);
         }
      }
      else
      {
         /* non-stencil entries */
         entry -= size;
         hypre_SStructGraphGetUVEntryRank(graph, part, var, index, &Uverank);

         if (Uverank < 0)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Non-stencil entry not found in SStruct Graph.");

            HYPRE_ANNOTATE_FUNC_END;

            return hypre_error_flag;
         }
         else
         {
            Uventry = hypre_SStructGraphUVEntry(graph, Uverank);
            cols_h[nvalues_set++] = hypre_SStructUVEntryToRank(Uventry, entry);
         }
      }
   }

   /* Copy rows/cols to device */
   hypre_TMemcpy(rows, rows_h, HYPRE_BigInt, nvalues_set, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(cols, cols_h, HYPRE_BigInt, nvalues_set, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(skip_val, skip_val_h, HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* Sort and reduce_by_key to get rows/nrows/ncols to pass to IJMatrixSet/AddTo/GetValues */
   values_set = hypre_TAlloc(HYPRE_Complex, nvalues_set, HYPRE_MEMORY_DEVICE);
   ncols = hypre_TAlloc(HYPRE_Int, nvalues_set, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   if (action < 0)
   {
      scatter_locations = hypre_TAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE);
      hypre_SequenceSycl( scatter_locations,
                          scatter_locations + nvalues,
                          0 );
      hypre_CopyIfSycl( oneapi::dpl::make_zip_iterator(values, scatter_locations),
                        oneapi::dpl::make_zip_iterator(values, scatter_locations) + nvalues,
                        skip_val,
                        oneapi::dpl::make_zip_iterator(values_set, scatter_locations),
                        equal<HYPRE_Int>(0) );
      HYPRE_ONEDPL_CALL( oneapi::dpl::sort_by_key,
                         oneapi::dpl::make_zip_iterator(rows, scatter_locations),
                         oneapi::dpl::make_zip_iterator(rows, scatter_locations) + nvalues_set,
                         oneapi::dpl::make_zip_iterator(cols, values_set) );
   }
   hypre_CopyIfSycl( values,
                     values + nvalues,
                     skip_val,
                     values_set,
                     equal<HYPRE_Int>(0) );
   HYPRE_ONEDPL_CALL( oneapi::dpl::sort_by_key,
                      rows,
                      rows + nvalues_set,
                      oneapi::dpl::make_zip_iterator(cols, values_set) );
   auto new_end = HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     rows,
                                     rows + nvalues_set,
                                     dpct::make_constant_iterator(1),
                                     rows,
                                     ncols );
   nrows = new_end.first - rows;
#else
   if (action < 0)
   {
      scatter_locations = hypre_TAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( sequence,
                         scatter_locations,
                         scatter_locations + nvalues,
                         0 );
      HYPRE_THRUST_CALL( copy_if,
                         thrust::make_zip_iterator(thrust::make_tuple(values, scatter_locations)),
                         thrust::make_zip_iterator(thrust::make_tuple(values, scatter_locations)) + nvalues,
                         skip_val,
                         thrust::make_zip_iterator(thrust::make_tuple(values_set, scatter_locations)),
                         equal<HYPRE_Int>(0) );
      HYPRE_THRUST_CALL( sort_by_key,
                         thrust::make_zip_iterator(thrust::make_tuple(rows, scatter_locations)),
                         thrust::make_zip_iterator(thrust::make_tuple(rows, scatter_locations)) + nvalues_set,
                         thrust::make_zip_iterator(thrust::make_tuple(cols, values_set)) );
   }
   HYPRE_THRUST_CALL( copy_if,
                      values,
                      values + nvalues,
                      skip_val,
                      values_set,
                      equal<HYPRE_Int>(0) );
   HYPRE_THRUST_CALL( sort_by_key,
                      rows,
                      rows + nvalues_set,
                      thrust::make_zip_iterator(thrust::make_tuple(cols, values_set)) );
   auto new_end = HYPRE_THRUST_CALL( reduce_by_key,
                                     rows,
                                     rows + nvalues_set,
                                     thrust::make_constant_iterator(1),
                                     rows,
                                     ncols );
   nrows = new_end.first - rows;
#endif

   /* Call IJMatrixSet/AddTo/GetValues */
   if (action > 0)
   {
      HYPRE_IJMatrixAddToValues(ijmatrix, nrows, ncols, rows, cols, values_set);
   }
   else if (action > -1)
   {
      HYPRE_IJMatrixSetValues(ijmatrix, nrows, ncols, rows, cols, values_set);
   }
   else
   {
      HYPRE_IJMatrixGetValues(ijmatrix, nrows, ncols, rows, cols, values_set);
#if defined(HYPRE_USING_SYCL)
      hypre_ScatterSycl( values_set,
                         values_set + nvalues_set,
                         scatter_locations,
                         values );
#else
      HYPRE_THRUST_CALL( scatter,
                         values_set,
                         values_set + nvalues_set,
                         scatter_locations,
                         values );
#endif
      hypre_TFree(scatter_locations, HYPRE_MEMORY_DEVICE);
   }

   hypre_TFree(values_set, HYPRE_MEMORY_DEVICE);
   hypre_TFree(rows, HYPRE_MEMORY_DEVICE);
   hypre_TFree(cols, HYPRE_MEMORY_DEVICE);
   hypre_TFree(skip_val, HYPRE_MEMORY_DEVICE);
   hypre_TFree(ncols, HYPRE_MEMORY_DEVICE);
   hypre_TFree(skip_val_h, HYPRE_MEMORY_HOST);
   hypre_TFree(indexes_h, HYPRE_MEMORY_HOST);
   hypre_TFree(rows_h, HYPRE_MEMORY_HOST);
   hypre_TFree(cols_h, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
#endif // defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * Returns the stored entry of U at local row and global column col, or NULL
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
HYPRE_Complex *
hypre_SStructUMatrixFindEntry( HYPRE_Int       row,
                               HYPRE_BigInt    col,
                               HYPRE_BigInt    first_col,
                               HYPRE_BigInt    last_col,
                               HYPRE_Int      *di,
                               HYPRE_Int      *dj,
                               HYPRE_Complex  *da,
                               HYPRE_Int      *oi,
                               HYPRE_Int      *oj,
                               HYPRE_Complex  *oa,
                               HYPRE_BigInt   *col_map,
                               HYPRE_Int       num_cols_offd )
{
   HYPRE_Int  lo = 0;
   HYPRE_Int  hi = num_cols_offd - 1;
   HYPRE_Int  mid, j;

   if (col >= first_col && col <= last_col)
   {
      for (j = di[row]; j < di[row + 1]; j++)
      {
         if (dj[j] == (HYPRE_Int) (col - first_col)) { return da + j; }
      }
   }
   else if (oi[row] < oi[row + 1])
   {
      while (lo < hi)
      {
         mid = (lo + hi) / 2;
         if (col_map[mid] < col) { lo = mid + 1; }
         else { hi = mid; }
      }
      for (j = oi[row]; j < oi[row + 1] && col_map[lo] == col; j++)
      {
         if (oj[j] == lo) { return oa + j; }
      }
   }

   return NULL;
}

/*--------------------------------------------------------------------------
 * Reads and clears stored U entries, retaining the supplied value when an
 * entry is absent. Unlike IJ GetValues, this implements the sparse overwrite
 * needed by U-to-S compression without replacing missing entries by zero.
 *--------------------------------------------------------------------------*/

static void
hypre_SStructUMatrixGetExistingValues( hypre_IJMatrix *ij,
                                       HYPRE_Int       nrows,
                                       HYPRE_Int      *ncols,
                                       HYPRE_BigInt   *rows,
                                       HYPRE_Int      *row_indexes,
                                       HYPRE_BigInt   *cols,
                                       HYPRE_Complex  *values )
{
   hypre_ParCSRMatrix  *U             = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij);
   hypre_CSRMatrix     *diag          = hypre_ParCSRMatrixDiag(U);
   hypre_CSRMatrix     *offd          = hypre_ParCSRMatrixOffd(U);
   HYPRE_Int           *di            = hypre_CSRMatrixI(diag);
   HYPRE_Int           *dj            = hypre_CSRMatrixJ(diag);
   HYPRE_Complex       *da            = hypre_CSRMatrixData(diag);
   HYPRE_Int           *oi            = hypre_CSRMatrixI(offd);
   HYPRE_Int           *oj            = hypre_CSRMatrixJ(offd);
   HYPRE_Complex       *oa            = hypre_CSRMatrixData(offd);
   HYPRE_BigInt         first_row     = hypre_ParCSRMatrixFirstRowIndex(U);
   HYPRE_BigInt         first_col     = hypre_ParCSRMatrixFirstColDiag(U);
   HYPRE_BigInt         last_col      = hypre_ParCSRMatrixLastColDiag(U);
   HYPRE_BigInt        *col_map       = hypre_ParCSRMatrixColMapOffd(U);
   HYPRE_Int            num_rows      = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int            num_cols_offd = hypre_CSRMatrixNumCols(offd);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(U)) == HYPRE_EXEC_DEVICE)
   {
      hypre_ParCSRMatrixCopyColMapOffdToDevice(U);
      col_map = hypre_ParCSRMatrixDeviceColMapOffd(U);
   }
#endif

   hypre_LoopBegin(nrows, r)
   {
      HYPRE_BigInt local_row = rows[r] - first_row;
      if (local_row >= 0 && local_row < num_rows)
      {
         for (HYPRE_Int k = row_indexes[r]; k < row_indexes[r] + ncols[r]; k++)
         {
            HYPRE_Complex *u = hypre_SStructUMatrixFindEntry((HYPRE_Int) local_row, cols[k],
                                                             first_col, last_col, di, dj, da,
                                                             oi, oj, oa, col_map, num_cols_offd);
            if (u)
            {
               values[k] = *u;
               *u = 0.0;
            }
         }
      }
   }
   hypre_LoopEnd();
}

/*--------------------------------------------------------------------------
 * Note: Entries must all be of type stencil or non-stencil, but not both.
 *
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 * (action =-3): get stored values and zero out; preserve absent stencil entries
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
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_IntFillnDevice(values_map, num_nonzeros, -1);
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

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_IntFillnDevice(ncols, nrows, 0);
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::transform,
                               oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                               oneapi::dpl::counting_iterator<HYPRE_Int>(nrows + 1),
                               row_indexes,
            [const_val = nentries] (const auto & x) { return x * const_val; } );
#else
            HYPRE_THRUST_CALL( transform,
                               thrust::counting_iterator<HYPRE_Int>(0),
                               thrust::counting_iterator<HYPRE_Int>(nrows + 1),
                               row_indexes,
                               _1 * nentries );
#endif
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
            if (action == -3)
            {
               hypre_SStructUMatrixGetExistingValues(ijmatrix, nrows, ncols, rows,
                                                      row_indexes, cols, ijvalues);
            }
            else if (action == -2)
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
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::for_each,
                               oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                               oneapi::dpl::counting_iterator<HYPRE_Int>(num_nonzeros),
                               [ = ] (HYPRE_Int i)
            {
               if (values_map[i] >= 0)
               {
                  values[i] = ijvalues[values_map[i]];
               }
            });
#else
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
         /* WM: todo - shouldn't it be Sentries instead of entries below??? */
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
         /* WM: todo - shouldn't it be Sentries instead of entries below??? */
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

#if defined(HYPRE_USING_GPU)
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* WM: todo - make this more generally available? Useful elsewhere? */
__global__ void
hypre_GPUKernelCopyIndexes(hypre_DeviceItem &item,
                           HYPRE_Int  ndim,
                           HYPRE_Int  n_copy_indexes,
                           HYPRE_Int *indexes_in,
                           HYPRE_Int *locations,
                           HYPRE_Int *indexes_out)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int j;

   if (i < n_copy_indexes)
   {
      for (j = 0; j < ndim; j++)
      {
         indexes_out[i * ndim + j] = indexes_in[ locations[i] * ndim + j ];
      }
   }
}

/*--------------------------------------------------------------------------
 * Split indexes, entries, and values arrays into struct and unstruct
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SStructMatrixSplitArrayEntriesDevice( HYPRE_SStructMatrix matrix,
                                            HYPRE_Int           part,
                                            HYPRE_Int           var,
                                            HYPRE_Int           nvalues,
                                            HYPRE_Int          *indexes,
                                            HYPRE_Int          *entries,
                                            HYPRE_Complex      *values,
                                            HYPRE_Int          *nSentries_out,
                                            HYPRE_Int          *nUentries_out,
                                            HYPRE_Int         **Sindexes_out,
                                            HYPRE_Int         **Uindexes_out,
                                            HYPRE_Int         **Sentries_out,
                                            HYPRE_Int         **Uentries_out,
                                            HYPRE_Complex     **Svalues_out,
                                            HYPRE_Complex     **Uvalues_out,
                                            HYPRE_Int         **Sentry_locations_out,
                                            HYPRE_Int         **Uentry_locations_out )
{
   HYPRE_Int             entry;
   HYPRE_Int             ndim      = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph   *graph     = hypre_SStructMatrixGraph(matrix);
   HYPRE_Int            *split     = hypre_SStructMatrixSplit(matrix, part, var);
   hypre_SStructStencil *stencil   = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int             nSentries;
   HYPRE_Int             nUentries;
   HYPRE_Int            *Sindexes;
   HYPRE_Int            *Uindexes;
   HYPRE_Int            *Sentries;
   HYPRE_Int            *Uentries;
   HYPRE_Complex        *Svalues;
   HYPRE_Complex        *Uvalues;

   HYPRE_Int *Sentry_locations = hypre_TAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *Uentry_locations = hypre_TAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim;

   /* entries_split_arr[i] = 0 if entries[i] is Sentry, 1 if entries[i] is Uentry */
   HYPRE_Int *entries_split_arr = hypre_CTAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   /* If entries[i] >= hypre_SStructStencilSize(stencil), then it is a Uentry */
   HYPRE_Int stencil_size = hypre_SStructStencilSize(stencil);
   hypre_TransformIfSycl(entries_split_arr,
                         entries_split_arr + nvalues,
                         entries,
                         entries_split_arr,
   [] (const auto & x) {return 1;},
   [stencil_size] (const auto & x) {return x >= stencil_size;} );
   /* If entries[i] == entry, where split[entry] < 0, then it is a Uentry */
   for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
   {
      if (split[entry] < 0)
      {
         hypre_TransformIfSycl(entries_split_arr,
                               entries_split_arr + nvalues,
                               entries,
                               entries_split_arr,
         [] (const auto & x) {return 1;},
         [entry] (const auto & x) {return x == entry;} );
      }
   }
   /* Get locations (indices) in the array where Sentries and Uentries live */
   auto Sentries_locations_end = hypre_CopyIfSycl( oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                                                   oneapi::dpl::counting_iterator<HYPRE_Int>(0) + nvalues,
                                                   entries_split_arr,
                                                   Sentry_locations,
                                                   equal<HYPRE_Int>(0) );
   auto Uentries_locations_end = hypre_CopyIfSycl( oneapi::dpl::counting_iterator<HYPRE_Int>(0),
                                                   oneapi::dpl::counting_iterator<HYPRE_Int>(0) + nvalues,
                                                   entries_split_arr,
                                                   Uentry_locations,
                                                   equal<HYPRE_Int>(1) );
#else
   /* If entries[i] >= hypre_SStructStencilSize(stencil), then it is a Uentry */
   HYPRE_THRUST_CALL(replace_if, entries_split_arr, entries_split_arr + nvalues, entries,
                     HYPRE_THRUST_NOT(less_than<HYPRE_Int>(hypre_SStructStencilSize(stencil))), 1);
   /* If entries[i] == entry, where split[entry] < 0, then it is a Uentry */
   for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
   {
      if (split[entry] < 0)
      {
         HYPRE_THRUST_CALL( replace_if,
                            entries_split_arr,
                            entries_split_arr + nvalues, entries,
                            equal<HYPRE_Int>(entry),
                            1);
      }
   }
   /* Get locations (indices) in the array where Sentries and Uentries live */
   auto Sentries_locations_end = HYPRE_THRUST_CALL( copy_if,
                                                    thrust::make_counting_iterator<HYPRE_Int>(0),
                                                    thrust::make_counting_iterator<HYPRE_Int>(0) + nvalues,
                                                    entries_split_arr,
                                                    Sentry_locations,
                                                    equal<HYPRE_Int>(0) );
   auto Uentries_locations_end = HYPRE_THRUST_CALL( copy_if,
                                                    thrust::make_counting_iterator<HYPRE_Int>(0),
                                                    thrust::make_counting_iterator<HYPRE_Int>(0) + nvalues,
                                                    entries_split_arr,
                                                    Uentry_locations,
                                                    equal<HYPRE_Int>(1) );
#endif
   /* Get number of Sentries and Uentries */
   nSentries = Sentries_locations_end - Sentry_locations;
   nUentries = Uentries_locations_end - Uentry_locations;

   /* Allocate split arrays for entries, indexes, values */
   Sentries = hypre_TAlloc(HYPRE_Int, nSentries, HYPRE_MEMORY_DEVICE);
   Uentries = hypre_TAlloc(HYPRE_Int, nUentries, HYPRE_MEMORY_DEVICE);
   Sindexes = hypre_TAlloc(HYPRE_Int, ndim * nSentries, HYPRE_MEMORY_DEVICE);
   Uindexes = hypre_TAlloc(HYPRE_Int, ndim * nUentries, HYPRE_MEMORY_DEVICE);
   Svalues = hypre_TAlloc(HYPRE_Complex, nSentries, HYPRE_MEMORY_DEVICE);
   Uvalues = hypre_TAlloc(HYPRE_Complex, nUentries, HYPRE_MEMORY_DEVICE);

   /* Copy from indexes into Sindexes and Uindexes */
   gDim = hypre_GetDefaultDeviceGridDimension(nSentries, "thread", bDim);
   HYPRE_GPU_LAUNCH( hypre_GPUKernelCopyIndexes, gDim, bDim, ndim, nSentries, indexes,
                     Sentry_locations, Sindexes );
   gDim = hypre_GetDefaultDeviceGridDimension(nUentries, "thread", bDim);
   HYPRE_GPU_LAUNCH( hypre_GPUKernelCopyIndexes, gDim, bDim, ndim, nUentries, indexes,
                     Uentry_locations, Uindexes );
#if defined(HYPRE_USING_SYCL)
   /* Copy from entries into Sentries and Uentries */
   hypre_CopyIfSycl( entries,
                     entries + nvalues,
                     entries_split_arr,
                     Sentries,
                     equal<HYPRE_Int>(0) );
   hypre_CopyIfSycl( entries,
                     entries + nvalues,
                     entries_split_arr,
                     Uentries,
                     equal<HYPRE_Int>(1) );

   /* Copy from values into Svalues and Uvalues */
   hypre_CopyIfSycl( values,
                     values + nvalues,
                     entries_split_arr,
                     Svalues,
                     equal<HYPRE_Int>(0) );
   hypre_CopyIfSycl( values,
                     values + nvalues,
                     entries_split_arr,
                     Uvalues,
                     equal<HYPRE_Int>(1) );
#else
   /* Copy from entries into Sentries and Uentries */
   HYPRE_THRUST_CALL( copy_if,
                      entries,
                      entries + nvalues,
                      entries_split_arr,
                      Sentries,
                      equal<HYPRE_Int>(0) );
   HYPRE_THRUST_CALL( copy_if,
                      entries,
                      entries + nvalues,
                      entries_split_arr,
                      Uentries,
                      equal<HYPRE_Int>(1) );

   /* Copy from values into Svalues and Uvalues */
   HYPRE_THRUST_CALL( copy_if,
                      values,
                      values + nvalues,
                      entries_split_arr,
                      Svalues,
                      equal<HYPRE_Int>(0) );
   HYPRE_THRUST_CALL( copy_if,
                      values,
                      values + nvalues,
                      entries_split_arr,
                      Uvalues,
                      equal<HYPRE_Int>(1) );
#endif

   /* Clean up memory */
   hypre_TFree(entries_split_arr, HYPRE_MEMORY_DEVICE);

   /* Assign outputs */
   *nSentries_out = nSentries;
   *nUentries_out = nUentries;
   *Sindexes_out = Sindexes;
   *Uindexes_out = Uindexes;
   *Sentries_out = Sentries;
   *Uentries_out = Uentries;
   *Svalues_out = Svalues;
   *Uvalues_out = Uvalues;
   *Sentry_locations_out = Sentry_locations;
   *Uentry_locations_out = Uentry_locations;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixSetArrayValuesDevice( HYPRE_SStructMatrix  matrix,
                                         HYPRE_Int            part,
                                         HYPRE_Int            var,
                                         HYPRE_Int            nvalues,
                                         HYPRE_Int           *indexes,
                                         HYPRE_Int           *entries,
                                         HYPRE_Complex       *values,
                                         HYPRE_Int            action )
{
   HYPRE_Int             i, entry;
   hypre_SStructGraph   *graph       = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid        = hypre_SStructGraphGrid(graph);
   HYPRE_Int           **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_SStructStencil *stencil     = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int            *vars        = hypre_SStructStencilVars(stencil);
   HYPRE_Int             nSentries;
   HYPRE_Int             nUentries;
   HYPRE_Int            *Sindexes;
   HYPRE_Int            *Uindexes;
   HYPRE_Int            *Sentries;
   HYPRE_Int            *Uentries;
   HYPRE_Complex        *Svalues;
   HYPRE_Complex        *Uvalues;
   HYPRE_Int            *Sentry_locations;
   HYPRE_Int            *Uentry_locations;
   hypre_SStructPMatrix *pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
   HYPRE_Int            *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   hypre_StructMatrix   *smatrix;

   HYPRE_Int            *Sentries_h;
   HYPRE_Int            *struct_stencil_indices;
   HYPRE_Int            *struct_stencil_indices_h;

   /* WM: todo - using vars[entries[0]] below assumes all the entries point to the same var */
   hypre_TMemcpy(&entry, entries, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entry]);

   hypre_SStructMatrixSplitArrayEntriesDevice( matrix,
                                               part,
                                               var,
                                               nvalues,
                                               indexes,
                                               entries,
                                               values,
                                               &nSentries,
                                               &nUentries,
                                               &Sindexes,
                                               &Uindexes,
                                               &Sentries,
                                               &Uentries,
                                               &Svalues,
                                               &Uvalues,
                                               &Sentry_locations,
                                               &Uentry_locations );

   /* S-matrix */
   if (nSentries > 0)
   {
      /* WM: todo - for now, doing the smap stuff on the host... */
      Sentries_h = hypre_TAlloc(HYPRE_Int, nSentries, HYPRE_MEMORY_HOST);
      struct_stencil_indices = hypre_TAlloc(HYPRE_Int, nSentries, HYPRE_MEMORY_DEVICE);
      struct_stencil_indices_h = hypre_TAlloc(HYPRE_Int, nSentries, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(Sentries_h, Sentries, HYPRE_Int, nSentries, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < nSentries; i++)
      {
         struct_stencil_indices_h[i] = smap[Sentries_h[i]];
      }
      hypre_TMemcpy(struct_stencil_indices, struct_stencil_indices_h, HYPRE_Int, nSentries,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TFree(Sentries_h, HYPRE_MEMORY_HOST);
      hypre_TFree(struct_stencil_indices_h, HYPRE_MEMORY_HOST);

      /* SetValues */
      if (action == 0)
      {
         hypre_StructMatrixSetArrayValuesDevice(smatrix,
                                                nSentries,
                                                Sindexes,
                                                struct_stencil_indices,
                                                Svalues,
                                                1);
      }
      /* AddToValues */
      else if (action > 0)
      {
         hypre_StructMatrixAddToArrayValuesDevice(smatrix,
                                                  nSentries,
                                                  Sindexes,
                                                  struct_stencil_indices,
                                                  Svalues,
                                                  1);
      }
      /* GetValues */
      else
      {
         hypre_StructMatrixGetArrayValuesDevice(smatrix,
                                                nSentries,
                                                Sindexes,
                                                struct_stencil_indices,
                                                Svalues);
      }

      hypre_TFree(struct_stencil_indices, HYPRE_MEMORY_DEVICE);

      /* put inter-part couplings in UMatrix and zero them out in PMatrix
       * (possibly in ghost zones) */
      if (nvneighbors[part][var] > 0)
      {
         hypre_SStructMatrixSetArrayInterPartValuesDevice(matrix, part, var, nSentries, Sindexes, Sentries,
                                                          Svalues, action);
      }
   }

   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetArrayValuesDevice(matrix, part, var, nUentries, Uindexes, Uentries, Uvalues,
                                               action);
   }

   /* Scatter struct and parcsr values back to input values array in the case of Get */
   if (action < 0)
   {
#if defined(HYPRE_USING_SYCL)
#else
      HYPRE_THRUST_CALL( scatter,
                         Svalues,
                         Svalues + nSentries,
                         Sentry_locations,
                         values );
      HYPRE_THRUST_CALL( scatter,
                         Uvalues,
                         Uvalues + nUentries,
                         Uentry_locations,
                         values );
#endif
   }

   /* Clean up memory */
   hypre_TFree(Sindexes, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Uindexes, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Sentries, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Uentries, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Svalues, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Uvalues, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Sentry_locations, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Uentry_locations, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}
#endif // defined(HYPRE_USING_GPU)

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

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * Put inter-part couplings in UMatrix and zero them out in PMatrix (possibly in
 * ghost zones).  Assumes that all entries are stencil entries.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixSetArrayInterPartValuesDevice( HYPRE_SStructMatrix  matrix,
                                                  HYPRE_Int            part,
                                                  HYPRE_Int            var,
                                                  HYPRE_Int            nvalues,
                                                  HYPRE_Int           *indexes,
                                                  HYPRE_Int           *entries,
                                                  HYPRE_Complex       *values,
                                                  HYPRE_Int            action )
{
   HYPRE_Int                ndim       = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph      *graph      = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid       *grid       = hypre_SStructGraphGrid(graph);
   hypre_SStructPMatrix    *pmatrix    = hypre_SStructMatrixPMatrix(matrix, part);
   hypre_SStructPGrid      *pgrid      = hypre_SStructPMatrixPGrid(pmatrix);
   HYPRE_Int               *smap       = hypre_SStructPMatrixSMap(pmatrix, var);
   hypre_SStructVariable    frvartype  = hypre_SStructPGridVarType(pgrid, var);
   hypre_SStructStencil    *stencil    = hypre_SStructPMatrixStencil(pmatrix, var);
   hypre_Index             *shape      = hypre_SStructStencilShape(stencil);
   HYPRE_Int               *vars       = hypre_SStructStencilVars(stencil);

   hypre_SStructVariable    tovartype;
   hypre_StructMatrix      *smatrix;
   hypre_Box               *set_box, *box, *ibox0, *ibox1, *tobox, *frbox;
   hypre_IndexRef           offset;
   hypre_BoxManEntry      **frentries, **toentries;
   hypre_SStructBoxManInfo *frinfo, *toinfo;
   HYPRE_Int                nfrentries, ntoentries, frpart, topart, ntvalues = 0;
   HYPRE_Int                entry, fri, toi, i, d;

   HYPRE_Int               *tindexes;
   HYPRE_Int               *tentries;
   HYPRE_Int               *tentries_h;
   HYPRE_Complex           *tvalues;
   HYPRE_Int               *tlocations;
   HYPRE_Int               *tlocations_h;
   HYPRE_Int               *indexes_h;
   HYPRE_Int               *entries_h;
   HYPRE_Int               *struct_stencil_indices;
   HYPRE_Int               *struct_stencil_indices_h;

   set_box   = hypre_BoxCreate(ndim);
   box       = hypre_BoxCreate(ndim);
   ibox0     = hypre_BoxCreate(ndim);
   ibox1     = hypre_BoxCreate(ndim);
   tobox     = hypre_BoxCreate(ndim);
   frbox     = hypre_BoxCreate(ndim);

   /* WM: todo - using vars[entries[0]] below assumes all the entries point to the same var */
   hypre_TMemcpy(&entry, entries, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entry]);

   /* Copy indexes and entries to the host */
   tlocations_h = hypre_TAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_HOST);
   indexes_h = hypre_TAlloc(HYPRE_Int, ndim * nvalues, HYPRE_MEMORY_HOST);
   entries_h = hypre_TAlloc(HYPRE_Int, nvalues, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(indexes_h, indexes, HYPRE_Int, ndim * nvalues, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(entries_h, entries, HYPRE_Int, nvalues, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

   for (i = 0; i < nvalues; i++)
   {
      entry  = entries_h[i];
      offset = shape[entry];
      tovartype = hypre_SStructPGridVarType(pgrid, vars[entry]);

      /* Copy index to a box to take advantage of existing box algebra fom SetInterPartValues */
      for (d = 0; d < ndim; d++)
      {
         hypre_BoxIMinD(set_box, d) = indexes_h[i * ndim + d];
         hypre_BoxIMaxD(set_box, d) = indexes_h[i * ndim + d];
      }

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
                  /* ibox1 should be a single index */
                  for (d = 0; d < ndim; d++)
                  {
                     hypre_assert(hypre_BoxIMinD(ibox1, d) == hypre_BoxIMaxD(ibox1, d));
                  }
                  tlocations_h[ntvalues++] = i;
               } /* end if nonzero ibox1 */
            } /* end of "from" boxman entries loop */

            hypre_TFree(frentries, HYPRE_MEMORY_HOST);
         } /* end if nonzero ibox0 */
      } /* end of "to" boxman entries loop */

      hypre_TFree(toentries, HYPRE_MEMORY_HOST);
   } /* end of entries loop */

   /* Copy tlocations to GPU */
   tlocations = hypre_TAlloc(HYPRE_Int, ntvalues, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(tlocations, tlocations_h, HYPRE_Int, ntvalues, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_HOST);

   /* Copy from indexes/entries/values into tindexes/tentries/tvalues */
   tindexes = hypre_TAlloc(HYPRE_Int, ndim * ntvalues, HYPRE_MEMORY_DEVICE);
   tentries = hypre_TAlloc(HYPRE_Int, ntvalues, HYPRE_MEMORY_DEVICE);
   tvalues = hypre_TAlloc(HYPRE_Complex, ntvalues, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   hypre_GatherSycl( tlocations,
                     tlocations + ntvalues,
                     entries,
                     tentries );
   hypre_GatherSycl( tlocations,
                     tlocations + ntvalues,
                     values,
                     tvalues );
#else
   HYPRE_THRUST_CALL( gather,
                      tlocations,
                      tlocations + ntvalues,
                      entries,
                      tentries );
   HYPRE_THRUST_CALL( gather,
                      tlocations,
                      tlocations + ntvalues,
                      values,
                      tvalues );
#endif
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(ntvalues, "thread", bDim);
   HYPRE_GPU_LAUNCH( hypre_GPUKernelCopyIndexes, gDim, bDim, ndim, ntvalues, indexes, tlocations,
                     tindexes );

   if (action >= 0)
   {
      /* set or add */
      /* copy values into tvalues */
#if defined(HYPRE_USING_SYCL)
      hypre_GatherSycl( tlocations,
                        tlocations + ntvalues,
                        values,
                        tvalues );
#else
      HYPRE_THRUST_CALL( gather,
                         tlocations,
                         tlocations + ntvalues,
                         values,
                         tvalues );
#endif

      /* put values into UMatrix */
      hypre_SStructUMatrixSetArrayValuesDevice(matrix, part, var, ntvalues, tindexes,
                                               tentries, tvalues, action);

      /* WM: todo - for now, doing the smap stuff on the host... */
      struct_stencil_indices = hypre_TAlloc(HYPRE_Int, ntvalues, HYPRE_MEMORY_DEVICE);
      struct_stencil_indices_h = hypre_TAlloc(HYPRE_Int, ntvalues, HYPRE_MEMORY_HOST);
      tentries_h = hypre_TAlloc(HYPRE_Int, ntvalues, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(tentries_h, tentries, HYPRE_Int, ntvalues, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < ntvalues; i++)
      {
         struct_stencil_indices_h[i] = smap[tentries_h[i]];
      }
      hypre_TMemcpy(struct_stencil_indices, struct_stencil_indices_h, HYPRE_Int, ntvalues,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TFree(tentries_h, HYPRE_MEMORY_HOST);
      hypre_TFree(struct_stencil_indices_h, HYPRE_MEMORY_HOST);

      /* zero out values in PMatrix (possibly in ghost) */
      hypre_StructMatrixSetArrayValuesDevice(smatrix, ntvalues, tindexes, struct_stencil_indices, NULL,
                                             1);
      hypre_TFree(struct_stencil_indices, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      /* get values from UMatrix */
      hypre_SStructUMatrixSetArrayValuesDevice(matrix, part, var, ntvalues, tindexes,
                                               tentries, tvalues, action);

      /* copy tvalues into values */
#if defined(HYPRE_USING_SYCL)
      hypre_ScatterSycl( tvalues,
                         tvalues + ntvalues,
                         tlocations,
                         values );
#else
      HYPRE_THRUST_CALL( scatter,
                         tvalues,
                         tvalues + ntvalues,
                         tlocations,
                         values );
#endif

   } /* end if action */

   /* Free memory */
   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(box);
   hypre_BoxDestroy(ibox0);
   hypre_BoxDestroy(ibox1);
   hypre_BoxDestroy(tobox);
   hypre_BoxDestroy(frbox);
   hypre_TFree(indexes_h, HYPRE_MEMORY_HOST);
   hypre_TFree(entries_h, HYPRE_MEMORY_HOST);
   hypre_TFree(tlocations, HYPRE_MEMORY_DEVICE);
   hypre_TFree(tlocations_h, HYPRE_MEMORY_HOST);
   hypre_TFree(tindexes, HYPRE_MEMORY_DEVICE);
   hypre_TFree(tentries, HYPRE_MEMORY_DEVICE);
   hypre_TFree(tvalues, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}
#endif

/*--------------------------------------------------------------------------
 * Add to or set (overwrite) entries in S with entries from U where
 * the sparsity pattern permits.
 *
 * (action > 0): add-to values
 * (action = 0): replace stencil values in selected rows (missing U entries are zero)
 * (action < 0): overwrite stored U entries only, retaining S at absent entries
 *
 * WM: TODO - what if there are constant stencil entries?
 * Not sure what the expected behavior should be. For now, avoid this case.
 * WM: TODO - does this potentially screw up pre-existing communication packages?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixCompressUToS( HYPRE_SStructMatrix A,
                                 HYPRE_Int           action )
{
   HYPRE_MemoryLocation     memory_location = hypre_SStructMatrixMemoryLocation(A);
   HYPRE_Int                nparts          = hypre_SStructMatrixNParts(A);
   HYPRE_Int               *Sentries        = hypre_SStructMatrixSEntries(A);
   HYPRE_SStructGraph       graph           = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid       *grid            = hypre_SStructGraphGrid(graph);
   HYPRE_Int              **nvneighbors     = hypre_SStructGridNVNeighbors(grid);
   HYPRE_Int                ndim            = hypre_SStructGridNDim(grid);

   hypre_ParCSRMatrix      *A_u             = hypre_SStructMatrixParCSRMatrix(A);
   hypre_CSRMatrix         *A_ud            = hypre_ParCSRMatrixDiag(A_u);
   hypre_CSRMatrix         *A_uo            = hypre_ParCSRMatrixOffd(A_u);
   HYPRE_Int                num_rows        = hypre_CSRMatrixNumRows(A_ud);

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
   HYPRE_Int                sparse_set;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy    exec = hypre_GetExecPolicy1(memory_location);

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

   /* Row replacement (action = 0) is used by interpolation; sparse overwrite
      (action < 0) is used by boundary PtAP. Try the box backend first */
   sparse_set = action < 0;
   if (action && hypre_SStructMatrixCompressUToSBox(A, sparse_set ? 0 : action))
   {
      return hypre_error_flag;
   }
   if (sparse_set)
   {
      action = 0;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructMatrixCompressUToS");

   /* Create work variables */
   grid_box = hypre_BoxCreate(ndim);

#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      max_num_rownnz = hypre_min(num_rows,
                                 hypre_CSRMatrixNumRownnz(A_ud) +
                                 hypre_CSRMatrixNumRownnz(A_uo));

      if (hypre_CSRMatrixRownnz(A_ud) && hypre_CSRMatrixRownnz(A_uo))
      {
         nonzero_rows = hypre_TAlloc(HYPRE_Int, max_num_rownnz, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::merge,
                            hypre_CSRMatrixRownnz(A_ud),
                            hypre_CSRMatrixRownnz(A_ud) + hypre_CSRMatrixNumRownnz(A_ud),
                            hypre_CSRMatrixRownnz(A_uo),
                            hypre_CSRMatrixRownnz(A_uo) + hypre_CSRMatrixNumRownnz(A_uo),
                            nonzero_rows );
#else
         HYPRE_THRUST_CALL( merge,
                            hypre_CSRMatrixRownnz(A_ud),
                            hypre_CSRMatrixRownnz(A_ud) + hypre_CSRMatrixNumRownnz(A_ud),
                            hypre_CSRMatrixRownnz(A_uo),
                            hypre_CSRMatrixRownnz(A_uo) + hypre_CSRMatrixNumRownnz(A_uo),
                            nonzero_rows );
#endif
      }
      else if (hypre_CSRMatrixRownnz(A_ud))
      {
         nonzero_rows = hypre_CSRMatrixRownnz(A_ud);
      }

      if (nonzero_rows)
      {
#if defined(HYPRE_USING_SYCL)
         nonzero_rows_end = HYPRE_ONEDPL_CALL(std::unique,
                                              nonzero_rows,
                                              nonzero_rows + max_num_rownnz);
#else
         nonzero_rows_end = HYPRE_THRUST_CALL(unique,
                                              nonzero_rows,
                                              nonzero_rows + max_num_rownnz);
#endif
      }
      else
      {
         nonzero_rows = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
         hypre_SequenceSycl( nonzero_rows, nonzero_rows + num_rows, 0 );
#else
         HYPRE_THRUST_CALL( sequence, nonzero_rows, nonzero_rows + num_rows );
#endif
         nonzero_rows_end = nonzero_rows + num_rows;
      }
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
                  if (ndim > 0) { all_indices_0[ii] = index[0] + start[0]; }
                  if (ndim > 1) { all_indices_1[ii] = index[1] + start[1]; }
                  if (ndim > 2) { all_indices_2[ii] = index[2] + start[2]; }
               }
               hypre_BoxLoop1End(ii);

#if defined(HYPRE_USING_SYCL)
               /* Get the nonzero rows for this box */
               box_nnzrows_end = HYPRE_ONEDPL_CALL( std::copy_if,
                                                    nonzero_rows,
                                                    nonzero_rows_end,
                                                    box_nnzrows,
                                                    in_range<HYPRE_Int>(offset, offset + volume) );
               HYPRE_ONEDPL_CALL( std::transform,
                                  box_nnzrows,
                                  box_nnzrows_end,
                                  box_nnzrows,
               [const_val = offset] (const auto & x) { return x - const_val; } );
               num_indices = box_nnzrows_end - box_nnzrows;

               /* Gather indices at non-zero rows of A_u */
               if (ndim > 0)
               {
                  hypre_GatherSycl( box_nnzrows, box_nnzrows_end, all_indices_0, indices_0 );
               }

               if (ndim > 1)
               {
                  hypre_GatherSycl( box_nnzrows, box_nnzrows_end, all_indices_1, indices_1 );
               }

               if (ndim > 2)
               {
                  hypre_GatherSycl( box_nnzrows, box_nnzrows_end, all_indices_2, indices_2 );
               }
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
                     hypre_SetIndex(index, 0);
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
                  values = hypre_CTAlloc(HYPRE_Complex, size, memory_location);

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
                                                   values, sparse_set ? -3 : -2);

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
                  hypre_TFree(values, memory_location);
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
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy    exec  = hypre_GetExecPolicy1(memory_location);
#endif

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
      if (exec == HYPRE_EXEC_DEVICE)
      {
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::fill, ncols, ncols + nrows, 1 );
         HYPRE_ONEDPL_CALL( std::fill, values, values + nrows, 1.0 );
         hypre_SequenceSycl( rowidx, rowidx + nrows, 0 );
         hypre_SequenceSycl( rows, rows + nrows, sizes[0] );
         hypre_SequenceSycl( cols, cols + nrows, sizes[2] );
#else
         HYPRE_THRUST_CALL( fill, ncols, ncols + nrows, 1 );
         HYPRE_THRUST_CALL( fill, values, values + nrows, 1.0 );
         HYPRE_THRUST_CALL( sequence, rowidx, rowidx + nrows );
         HYPRE_THRUST_CALL( sequence, rows, rows + nrows, sizes[0] );
         HYPRE_THRUST_CALL( sequence, cols, cols + nrows, sizes[2] );
#endif
      }
      else
#endif
      {
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
      }

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

typedef struct
{
   HYPRE_Int       size[3], row_stride[3], col_stride[3], data_stride[3];
   HYPRE_Int       data_start, constant, volume, offset;
   HYPRE_BigInt    row_base, col_base;
   HYPRE_Complex  *data;
} hypre_SStructMatrixCopySegment;

/*--------------------------------------------------------------------------
 * Sets (add = 0) or adds (add = 1) value in column col of the row stored in
 * [start, *end) of cols and values, appending the column when absent. Repeated
 * structured entries are set, keeping the last one as IJ SetValues does.
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
void
hypre_SStructRowAccumulate( HYPRE_BigInt    *cols,
                            HYPRE_Complex   *values,
                            HYPRE_Int        start,
                            HYPRE_Int       *end,
                            HYPRE_BigInt     col,
                            HYPRE_Complex    value,
                            HYPRE_Int        add )
{
   HYPRE_Int  k = start;

   while (k < *end && cols[k] != col)
   {
      k++;
   }
   if (k == *end)
   {
      cols[(*end)++] = col;
      values[k] = 0.0;
   }
   values[k] = add ? values[k] + value : value;
}

/* Assemble the structured subset as row-packed CSR, avoiding
 * IJ's global COO sorting. */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__global__ void
hypre_GPUKernelSStructCountU( hypre_DeviceItem &item,
                              HYPRE_Int         nrows,
                              HYPRE_Int        *diag,
                              HYPRE_Int        *offd,
                              HYPRE_Int        *counts )
{
   HYPRE_Int  row = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (row < nrows)
   {
      counts[row + 1] += (diag ? diag[row + 1] - diag[row] : 0) +
                         (offd ? offd[row + 1] - offd[row] : 0);
   }
}

__global__ void
hypre_GPUKernelSStructValidCounts( hypre_DeviceItem &item,
                                   HYPRE_Int         nrows,
                                   HYPRE_Int        *old_i,
                                   HYPRE_Int        *cursor,
                                   HYPRE_Int        *new_i )
{
   HYPRE_Int  row = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (row < nrows)
   {
      new_i[row + 1] = cursor[row] - old_i[row];
   }
}

__global__ void
hypre_GPUKernelSStructPackRows( hypre_DeviceItem &item,
                                HYPRE_Int         nrows,
                                HYPRE_Int        *old_i,
                                HYPRE_Int        *new_i,
                                HYPRE_BigInt     *old_j,
                                HYPRE_BigInt     *new_j,
                                HYPRE_Complex    *old_data,
                                HYPRE_Complex    *new_data )
{
   HYPRE_Int  row = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int  j, k;

   if (row < nrows)
   {
      for (j = new_i[row]; j < new_i[row + 1]; j++)
      {
         k = old_i[row] + j - new_i[row];
         new_j[j] = old_j[k];
         new_data[j] = old_data[k];
      }
   }
}

__global__ void
hypre_GPUKernelSStructCopyCSR( hypre_DeviceItem               &item,
                               hypre_SStructMatrixCopySegment *segments,
                               HYPRE_Int                       nsegments,
                               HYPRE_Int                       nnz,
                               HYPRE_BigInt                    first,
                               HYPRE_Int                      *cursor,
                               HYPRE_BigInt                   *cols,
                               HYPRE_Complex                  *values,
                               HYPRE_Int                      *order,
                               HYPRE_Int                       fill )
{
   HYPRE_Int                        i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   hypre_SStructMatrixCopySegment  *s;
   HYPRE_Int                        lo, hi, mid;
   HYPRE_Int                        x, y, z, row, j, k;

   if (i < nnz)
   {
      /* Find the segment of this entry by binary search */
      lo = 0;
      hi = nsegments;
      while (lo + 1 < hi)
      {
         mid = lo + (hi - lo) / 2;
         if (segments[mid].offset <= i)
         {
            lo = mid;
         }
         else
         {
            hi = mid;
         }
      }

      s  = segments + lo;
      i -= s->offset;
      x  = i % s->size[0];
      y  = (i / s->size[0]) % s->size[1];
      z  = i / (s->size[0] * s->size[1]);

      row = (HYPRE_Int) (s->row_base - first + (HYPRE_BigInt) x * s->row_stride[0] +
                         (HYPRE_BigInt) y * s->row_stride[1] +
                         (HYPRE_BigInt) z * s->row_stride[2]);
      j   = atomicAdd(cursor + row + (fill ? 0 : 1), 1);
      if (fill)
      {
         order[j] = s->offset + i;
         cols[j]  = s->col_base + (HYPRE_BigInt) x * s->col_stride[0] +
                    (HYPRE_BigInt) y * s->col_stride[1] + (HYPRE_BigInt) z * s->col_stride[2];
         k = s->constant ? 0 : s->data_start + x * s->data_stride[0] +
             y * s->data_stride[1] + z * s->data_stride[2];
         values[j] = s->data[k];
      }
   }
}

/*--------------------------------------------------------------------------
 * Atomic row reservations are unordered. Restore segment order within each
 * short stencil row, then merge repeated entries and U as on the host, so
 * rows and summation order are repeatable.
 *--------------------------------------------------------------------------*/

__global__ void
hypre_GPUKernelSStructFinishRows( hypre_DeviceItem &item,
                                  HYPRE_Int         nrows,
                                  HYPRE_BigInt      firstcol,
                                  HYPRE_Int        *rowptr,
                                  HYPRE_Int        *cursor,
                                  HYPRE_Int        *order,
                                  HYPRE_BigInt     *cols,
                                  HYPRE_Complex    *values,
                                  HYPRE_Int        *di,
                                  HYPRE_Int        *dj,
                                  HYPRE_Complex    *da,
                                  HYPRE_Int        *oi,
                                  HYPRE_Int        *oj,
                                  HYPRE_Complex    *oa,
                                  HYPRE_BigInt     *map,
                                  HYPRE_Int        *duplicates )
{
   HYPRE_Int       row = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int       start, end;
   HYPRE_Int       j, k, key, block;
   HYPRE_Int      *ii, *jj;
   HYPRE_BigInt    col;
   HYPRE_Complex   value;
   HYPRE_Complex  *data;

   if (row < nrows)
   {
      start = rowptr[row];
      end   = start;

      /* Insertion sort of the entries of the row by their segment order */
      for (j = start + 1; j < cursor[row]; j++)
      {
         key   = order[j];
         col   = cols[j];
         value = values[j];
         k     = j;
         while (k > start && order[k - 1] > key)
         {
            order[k]  = order[k - 1];
            cols[k]   = cols[k - 1];
            values[k] = values[k - 1];
            k--;
         }
         order[k]  = key;
         cols[k]   = col;
         values[k] = value;
      }

      /* Merge repeated structured entries */
      for (j = start; j < cursor[row]; j++)
      {
         hypre_SStructRowAccumulate(cols, values, start, &end, cols[j], values[j], 0);
      }

      /* Add U (diagonal block first, then off-diagonal block) */
      for (block = 0; block < 2; block++)
      {
         ii   = block ? oi : di;
         jj   = block ? oj : dj;
         data = block ? oa : da;
         if (!ii)
         {
            continue;
         }

         for (j = ii[row]; j < ii[row + 1]; j++)
         {
            hypre_SStructRowAccumulate(cols, values, start, &end,
                                       block ? map[jj[j]] : firstcol + jj[j], data[j], 1);
         }
      }

      cursor[row] = end;
      if (end != rowptr[row + 1])
      {
         atomicExch(duplicates, 1);
      }
   }
}

/*--------------------------------------------------------------------------
 * Device CSR assembly: count the entries of each row, scan, and then fill
 * the rows with the entries of the segments and of U
 *--------------------------------------------------------------------------*/

static void
hypre_SStructMatrixCopyBoxesDevice( hypre_IJMatrix                 *ij,
                                    hypre_SStructMatrixCopySegment *segments,
                                    HYPRE_Int                       count,
                                    HYPRE_Int                       nnz,
                                    hypre_ParCSRMatrix             *U )
{
   HYPRE_Int                        structured_nnz = nnz;
   HYPRE_BigInt                     range[4];
   HYPRE_Int                        nrows, num_cols_offd, duplicate_count;

   hypre_ParCSRMatrix              *par;
   hypre_CSRMatrix                 *csr, *diag, *offd;
   hypre_CSRMatrix                 *ud, *uo;
   hypre_SStructMatrixCopySegment  *device_segments;
   HYPRE_Int                       *rowptr, *cursor, *order, *duplicates, *map, *new_i;
   HYPRE_BigInt                    *cols, *new_j, *colmap, *umap = NULL;
   HYPRE_Complex                   *values, *new_data;
   HYPRE_Int                       *di = NULL, *dj = NULL, *oi = NULL, *oj = NULL;
   HYPRE_Complex                   *da = NULL, *oa = NULL;
   dim3                             block, launch, rows_launch;

   if (U)
   {
      nnz += hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(U)) +
             hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(U));
   }
   HYPRE_IJMatrixGetLocalRange(ij, range, range + 1, range + 2, range + 3);
   nrows = (HYPRE_Int) (range[1] - range[0] + 1);

   /* Allocate memory */
   csr             = hypre_CSRMatrixCreate(nrows, 0, nnz);
   rowptr          = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);
   cursor          = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE);
   order           = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   duplicates      = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);
   cols            = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
   values          = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);
   device_segments = hypre_TAlloc(hypre_SStructMatrixCopySegment, count, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(device_segments, segments, hypre_SStructMatrixCopySegment, count,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   block       = hypre_GetDefaultDeviceBlockDimension();
   launch      = hypre_GetDefaultDeviceGridDimension(structured_nnz, "thread", block);
   rows_launch = hypre_GetDefaultDeviceGridDimension(nrows, "thread", block);

   /* Count the entries of each row */
   if (structured_nnz)
   {
      HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructCopyCSR, launch, block, device_segments, count,
                       structured_nnz, range[0], rowptr, cols, values, order, 0);
   }
   if (U)
   {
      ud = hypre_ParCSRMatrixDiag(U);
      uo = hypre_ParCSRMatrixOffd(U);
      di = hypre_CSRMatrixI(ud);
      dj = hypre_CSRMatrixJ(ud);
      da = hypre_CSRMatrixData(ud);
      oi = hypre_CSRMatrixI(uo);
      oj = hypre_CSRMatrixJ(uo);
      oa = hypre_CSRMatrixData(uo);
      hypre_ParCSRMatrixCopyColMapOffdToDevice(U);
      umap = hypre_ParCSRMatrixDeviceColMapOffd(U);
      HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructCountU, rows_launch, block, nrows, di, oi, rowptr);
   }
   hypre_IntegerInclusiveScanDevice(nrows + 1, rowptr);
   hypre_TMemcpy(cursor, rowptr, HYPRE_Int, nrows, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   /* Fill the rows */
   if (structured_nnz)
   {
      HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructCopyCSR, launch, block, device_segments, count,
                       structured_nnz, range[0], cursor, cols, values, order, 1);
   }
   HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructFinishRows, rows_launch, block, nrows, range[2],
                    rowptr, cursor, order, cols, values, di, dj, da, oi, oj, oa, umap,
                    duplicates);
   hypre_TFree(order, HYPRE_MEMORY_DEVICE);

   /* Pack the rows if there were repeated entries */
   hypre_TMemcpy(&duplicate_count, duplicates, HYPRE_Int, 1,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_TFree(duplicates, HYPRE_MEMORY_DEVICE);
   if (duplicate_count)
   {
      new_i = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_DEVICE);
      HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructValidCounts, rows_launch, block, nrows, rowptr,
                       cursor, new_i);
      hypre_IntegerInclusiveScanDevice(nrows + 1, new_i);
      hypre_TMemcpy(&nnz, new_i + nrows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

      new_j    = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_DEVICE);
      new_data = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_DEVICE);
      HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructPackRows, rows_launch, block, nrows, rowptr,
                       new_i, cols, new_j, values, new_data);
      hypre_TFree(rowptr, HYPRE_MEMORY_DEVICE);
      hypre_TFree(cols, HYPRE_MEMORY_DEVICE);
      hypre_TFree(values, HYPRE_MEMORY_DEVICE);

      rowptr = new_i;
      cols   = new_j;
      values = new_data;
   }

   /* Split the CSR matrix into diag and offd and set up the ParCSR matrix */
   hypre_CSRMatrixNumNonzeros(csr)    = nnz;
   hypre_CSRMatrixMemoryLocation(csr) = HYPRE_MEMORY_DEVICE;
   hypre_CSRMatrixI(csr)              = rowptr;
   hypre_CSRMatrixBigJ(csr)           = cols;
   hypre_CSRMatrixData(csr)           = values;
   hypre_CSRMatrixSplitDevice(csr, range[2], range[3], 0, NULL, &map,
                              &num_cols_offd, &colmap, &diag, &offd);

   hypre_IJMatrixCreateParCSR(ij);
   par = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij);
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(par));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(par));
   hypre_ParCSRMatrixDiag(par)             = diag;
   hypre_ParCSRMatrixOffd(par)             = offd;
   hypre_ParCSRMatrixDeviceColMapOffd(par) = colmap;
   hypre_ParCSRMatrixColMapOffd(par)       = hypre_TAlloc(HYPRE_BigInt, num_cols_offd,
                                                          HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(par), colmap, HYPRE_BigInt, num_cols_offd,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixSetRownnz(diag);
   hypre_CSRMatrixSetRownnz(offd);
   hypre_ParCSRMatrixSetNumNonzeros(par);
   hypre_IJMatrixAssembleFlag(ij) = 1;

   /* Free memory */
   hypre_CSRMatrixDestroy(csr);
   hypre_TFree(map, HYPRE_MEMORY_DEVICE);
   hypre_TFree(cursor, HYPRE_MEMORY_DEVICE);
   hypre_TFree(device_segments, HYPRE_MEMORY_DEVICE);
}
#endif

/* Host CSR assembly; structured and U coefficients share one pass. */
/*--------------------------------------------------------------------------
 * Host CSR assembly: structured and U coefficients share one pass
 *--------------------------------------------------------------------------*/

static void
hypre_SStructMatrixCopyBoxesHost( hypre_IJMatrix                 *ij,
                                  hypre_SStructMatrixCopySegment *segments,
                                  HYPRE_Int                       count,
                                  HYPRE_Int                       nnz,
                                  hypre_ParCSRMatrix             *U )
{
   hypre_CSRMatrix                 *ud = U ? hypre_ParCSRMatrixDiag(U) : NULL;
   hypre_CSRMatrix                 *uo = U ? hypre_ParCSRMatrixOffd(U) : NULL;
   hypre_SStructMatrixCopySegment  *s;
   hypre_ParCSRMatrix              *par;
   hypre_CSRMatrix                 *csr, *diag, *offd, *ucsr;
   HYPRE_BigInt                     range[4];
   HYPRE_BigInt                    *colmap, *jj;
   HYPRE_BigInt                     col;
   HYPRE_Complex                   *data;
   HYPRE_Int                       *ii, *cursor;
   HYPRE_Int                        nrows, noffd, packed, start, end;
   HYPRE_Int                        e, r, x, y, z, j, block, row, ucol;

   HYPRE_IJMatrixGetLocalRange(ij, range, range + 1, range + 2, range + 3);
   nrows = (HYPRE_Int) (range[1] - range[0] + 1);
   ii    = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);

   /* Count the entries of each row */
   if (U)
   {
      nnz += hypre_CSRMatrixNumNonzeros(ud) + hypre_CSRMatrixNumNonzeros(uo);
      for (r = 0; r < nrows; r++)
      {
         ii[r + 1] = hypre_CSRMatrixI(ud)[r + 1] - hypre_CSRMatrixI(ud)[r] +
                     hypre_CSRMatrixI(uo)[r + 1] - hypre_CSRMatrixI(uo)[r];
      }
   }
   for (e = 0; e < count; e++)
   {
      s = segments + e;
      for (z = 0; z < s->size[2]; z++)
      {
         for (y = 0; y < s->size[1]; y++)
         {
            row = (HYPRE_Int) (s->row_base - range[0] + (HYPRE_BigInt) y * s->row_stride[1] +
                               (HYPRE_BigInt) z * s->row_stride[2]);
            for (x = 0; x < s->size[0]; x++)
            {
               ii[row + x * s->row_stride[0] + 1]++;
            }
         }
      }
   }
   for (r = 0; r < nrows; r++)
   {
      ii[r + 1] += ii[r];
   }

   /* Fill the rows with the entries of the segments */
   cursor = hypre_TAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(cursor, ii, HYPRE_Int, nrows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   jj   = hypre_TAlloc(HYPRE_BigInt, nnz, HYPRE_MEMORY_HOST);
   data = hypre_TAlloc(HYPRE_Complex, nnz, HYPRE_MEMORY_HOST);
   for (e = 0; e < count; e++)
   {
      s = segments + e;
      for (z = 0; z < s->size[2]; z++)
      {
         for (y = 0; y < s->size[1]; y++)
         {
            row   = (HYPRE_Int) (s->row_base - range[0] + (HYPRE_BigInt) y * s->row_stride[1] +
                                 (HYPRE_BigInt) z * s->row_stride[2]);
            col   = s->col_base + (HYPRE_BigInt) y * s->col_stride[1] +
                    (HYPRE_BigInt) z * s->col_stride[2];
            start = s->data_start + y * s->data_stride[1] + z * s->data_stride[2];
            for (x = 0; x < s->size[0]; x++)
            {
               j = cursor[row + x * s->row_stride[0]]++;
               jj[j]   = col + (HYPRE_BigInt) x * s->col_stride[0];
               data[j] = s->data[s->constant ? 0 : start + x * s->data_stride[0]];
            }
         }
      }
   }

   /* Merge repeated structured entries and add U */
   for (r = 0; r < nrows; r++)
   {
      end = ii[r];
      for (j = ii[r]; j < cursor[r]; j++)
      {
         hypre_SStructRowAccumulate(jj, data, ii[r], &end, jj[j], data[j], 0);
      }

      for (block = 0; U && block < 2; block++)
      {
         ucsr = block ? uo : ud;
         for (j = hypre_CSRMatrixI(ucsr)[r]; j < hypre_CSRMatrixI(ucsr)[r + 1]; j++)
         {
            ucol = hypre_CSRMatrixJ(ucsr)[j];
            hypre_SStructRowAccumulate(jj, data, ii[r], &end,
                                       block ? hypre_ParCSRMatrixColMapOffd(U)[ucol] :
                                       range[2] + ucol,
                                       hypre_CSRMatrixData(ucsr)[j], 1);
         }
      }
      cursor[r] = end;
   }

   /* Pack the rows */
   packed = 0;
   for (r = 0; r < nrows; r++)
   {
      start = ii[r];
      ii[r] = packed;
      for (j = start; j < cursor[r]; j++)
      {
         jj[packed]     = jj[j];
         data[packed++] = data[j];
      }
   }
   ii[nrows] = packed;

   /* Split the CSR matrix into diag and offd and set up the ParCSR matrix */
   csr = hypre_CSRMatrixCreate(nrows, 0, packed);
   hypre_CSRMatrixMemoryLocation(csr) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI(csr)              = ii;
   hypre_CSRMatrixBigJ(csr)           = jj;
   hypre_CSRMatrixData(csr)           = data;
   hypre_CSRMatrixSplit(csr, range[2], range[3], 0, NULL, &noffd, &colmap, &diag, &offd);

   hypre_IJMatrixCreateParCSR(ij);
   par = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij);
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(par));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(par));
   hypre_ParCSRMatrixDiag(par)       = diag;
   hypre_ParCSRMatrixOffd(par)       = offd;
   hypre_ParCSRMatrixColMapOffd(par) = colmap;
   hypre_CSRMatrixSetRownnz(diag);
   hypre_CSRMatrixSetRownnz(offd);
   hypre_ParCSRMatrixSetNumNonzeros(par);
   hypre_IJMatrixAssembleFlag(ij) = 1;

   /* Free memory */
   hypre_CSRMatrixDestroy(csr);
   hypre_TFree(cursor, HYPRE_MEMORY_HOST);
}

/* Reject unsupported layouts before collecting support or constructing
 * segments. Compression additionally requires a square, nonconstant stencil. */
static HYPRE_Int
hypre_SStructMatrixCopyLayout( hypre_SStructMatrix *A,
                               hypre_SStructGrid   *grid,
                               HYPRE_Int            compress )
{
   HYPRE_Int              ndim   = hypre_SStructMatrixNDim(A);
   HYPRE_Int              nparts = hypre_SStructMatrixNParts(A);
   hypre_SStructGraph    *graph  = hypre_SStructMatrixGraph(A);

   hypre_SStructPMatrix  *pA;
   hypre_SStructPGrid    *pgrid;
   hypre_SStructStencil  *stencil;
   hypre_StructMatrix    *sA;
   hypre_IndexRef         stride;
   HYPRE_Int             *split;
   HYPRE_Int             *smap;
   HYPRE_Int              part, d, e, se;

   if (ndim < 1 || ndim > 3 ||
       hypre_SStructGridNParts(grid) != nparts ||
       (compress && hypre_SStructGraphDomGrid(graph) != grid))
   {
      return 0;
   }

   for (part = 0; part < nparts; part++)
   {
      pA    = hypre_SStructMatrixPMatrix(A, part);
      pgrid = hypre_SStructGridPGrid(grid, part);

      if (hypre_SStructPMatrixNVars(pA) != 1 ||
          hypre_SStructPGridNVars(pgrid) != 1 ||
          hypre_SStructPGridVarType(pgrid, 0) != HYPRE_SSTRUCT_VARIABLE_CELL)
      {
         return 0;
      }

      sA = hypre_SStructPMatrixSMatrix(pA, 0, 0);
      if (!sA)
      {
         return 0;
      }

      stride = hypre_SStructPMatrixDomainStride(pA);
      for (d = 0; d < ndim; d++)
      {
         if (stride[d] < 1 ||
             hypre_SStructPMatrixRangeStride(pA)[d] != 1 ||
             hypre_StructMatrixDataStride(sA)[d] != stride[d] ||
             (compress && stride[d] != 1))
         {
            return 0;
         }
      }

      stencil = hypre_SStructGraphStencil(graph, part, 0);
      split   = hypre_SStructMatrixSplit(A, part, 0);
      smap    = hypre_SStructPMatrixSMap(pA, 0);
      for (e = 0; e < hypre_SStructStencilSize(stencil); e++)
      {
         if (split[e] < 0 || hypre_SStructStencilVars(stencil)[e] != 0)
         {
            return 0;
         }

         se = smap[split[e]];
         if (hypre_StructMatrixSymmEntries(sA)[se] >= 0 ||
             (compress && hypre_StructMatrixConstEntry(sA, se)))
         {
            return 0;
         }
      }
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * Builds the segments of the structured component of A over the rows in boxes:
 * one segment per stencil entry and box manager intersection of rows and
 * columns. col_action selects the column box managers as in
 * hypre_SStructGridIntersect. The layout must pass hypre_SStructMatrixCopyLayout.
 * Returns zero (locally) for selections outside the local boxes or for too many
 * entries.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SStructMatrixBoxSegments( hypre_SStructMatrix              *A,
                                hypre_SStructGrid                *grid,
                                hypre_BoxArray                ***boxes,
                                HYPRE_Int                        col_action,
                                hypre_SStructMatrixCopySegment  **segments_ptr,
                                HYPRE_Int                        *count_ptr,
                                HYPRE_Int                        *nnz_ptr )
{
   HYPRE_Int                        ndim     = hypre_SStructMatrixNDim(A);
   HYPRE_Int                        nparts   = hypre_SStructMatrixNParts(A);
   HYPRE_Int                        type     = hypre_SStructMatrixObjectType(A);
   hypre_SStructGraph              *graph    = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid               *dom_grid = hypre_SStructGraphDomGrid(graph);

   /* Segments */
   hypre_SStructMatrixCopySegment  *segments = NULL;
   hypre_SStructMatrixCopySegment   s;
   HYPRE_Int                        count    = 0;
   HYPRE_Int                        capacity = 0;
   HYPRE_Int                        nnz      = 0;
   HYPRE_Int                        valid    = 1;

   /* Matrix data of each part */
   hypre_SStructPMatrix            *pA;
   hypre_SStructPGrid              *pgrid;
   hypre_SStructStencil            *stencil;
   hypre_StructMatrix              *sA;
   hypre_BoxArray                  *grid_boxes;
   hypre_IndexRef                   stride;
   hypre_IndexRef                   offset;
   HYPRE_Int                       *split;
   HYPRE_Int                       *smap;

   /* Boxes and box manager entries */
   hypre_BoxArray                  *outside, *scratch, *selected;
   hypre_Box                       *box, *data_box;
   hypre_Box                       *clipped, *region, *map_box, *to_box, *intersection;
   hypre_BoxManEntry              **row_entries;
   hypre_BoxManEntry              **col_entries;
   hypre_SStructBoxManInfo         *from_info, *to_info;
   hypre_Index                      rs, cs, data_index;
   HYPRE_Int                        nrow_entries, ncol_entries;
   HYPRE_Int                        from_part, to_part;
   HYPRE_Int                        data_step, sentry;
   HYPRE_Int                        part, b, g, e, i, j, d;

   clipped      = hypre_BoxCreate(ndim);
   region       = hypre_BoxCreate(ndim);
   map_box      = hypre_BoxCreate(ndim);
   to_box       = hypre_BoxCreate(ndim);
   intersection = hypre_BoxCreate(ndim);

   for (part = 0; valid && part < nparts; part++)
   {
      pA         = hypre_SStructMatrixPMatrix(A, part);
      pgrid      = hypre_SStructGridPGrid(grid, part);
      sA         = hypre_SStructPMatrixSMatrix(pA, 0, 0);
      stride     = hypre_SStructPMatrixDomainStride(pA);
      stencil    = hypre_SStructGraphStencil(graph, part, 0);
      split      = hypre_SStructMatrixSplit(A, part, 0);
      smap       = hypre_SStructPMatrixSMap(pA, 0);
      grid_boxes = hypre_StructGridBoxes(hypre_SStructPGridSGrid(pgrid, 0));

      /* Clipping must not discard selected ghost rows. The general backend
         handles selections outside the owned boxes. Selections spanning more
         than one owned box are handled by the per-source union below. */
      outside = hypre_BoxArrayClone(boxes[part][0]);
      scratch = hypre_BoxArrayCreate(0, ndim);
      hypre_SubtractBoxArrays(outside, grid_boxes, scratch);
      valid = (hypre_BoxArraySize(outside) == 0);
      hypre_BoxArrayDestroy(outside);
      hypre_BoxArrayDestroy(scratch);
      if (!valid)
      {
         break;
      }

      selected = hypre_BoxArrayCreate(0, ndim);
      for (g = 0; valid && g < hypre_BoxArraySize(grid_boxes); g++)
      {
         hypre_BoxArraySetSize(selected, 0);
         hypre_ForBoxI(b, boxes[part][0])
         {
            hypre_IntersectBoxes(hypre_BoxArrayBox(boxes[part][0], b),
                                 hypre_BoxArrayBox(grid_boxes, g), clipped);
            if (hypre_BoxVolume(clipped))
            {
               hypre_AppendBox(clipped, selected);
            }
         }

         /* Union within each source box so overlapping selections neither
            duplicate coefficients nor force a slow IJ assembly */
         hypre_UnionBoxes(selected);

         data_box = hypre_StructMatrixBoxDataBox(sA, g);
         for (b = 0; valid && b < hypre_BoxArraySize(selected); b++)
         {
            box = hypre_BoxArrayBox(selected, b);
            hypre_SStructGridIntersect(grid, part, 0, box, -1, &row_entries, &nrow_entries);
            for (i = 0; valid && i < nrow_entries; i++)
            {
               hypre_SStructBoxManEntryGetStrides(row_entries[i], rs, type);
               hypre_BoxManEntryGetExtents(row_entries[i], hypre_BoxIMin(map_box),
                                           hypre_BoxIMax(map_box));
               hypre_IntersectBoxes(box, map_box, region);

               for (e = 0; valid && e < hypre_SStructStencilSize(stencil); e++)
               {
                  offset = hypre_SStructStencilShape(stencil)[e];
                  sentry = smap[split[e]];

                  hypre_CopyBox(region, to_box);
                  hypre_BoxShiftPos(to_box, offset);
                  hypre_CoarsenBox(to_box, NULL, stride);
                  hypre_SStructGridIntersect(dom_grid, part, 0, to_box, col_action,
                                             &col_entries, &ncol_entries);
                  for (j = 0; valid && j < ncol_entries; j++)
                  {
                     /* Skip the couplings within the same part (and box manager
                        info type) when looking for the inter-part ones */
                     if (col_action == 1)
                     {
                        hypre_SStructBoxManEntryGetPart(row_entries[i], part, &from_part);
                        hypre_SStructBoxManEntryGetPart(col_entries[j], part, &to_part);
                        hypre_BoxManEntryGetInfo(row_entries[i], (void **) &from_info);
                        hypre_BoxManEntryGetInfo(col_entries[j], (void **) &to_info);
                        if (from_part == to_part &&
                            hypre_SStructBoxManInfoType(from_info) ==
                            hypre_SStructBoxManInfoType(to_info))
                        {
                           continue;
                        }
                     }

                     hypre_BoxManEntryGetExtents(col_entries[j], hypre_BoxIMin(map_box),
                                                 hypre_BoxIMax(map_box));
                     hypre_IntersectBoxes(to_box, map_box, intersection);
                     if (!hypre_BoxVolume(intersection))
                     {
                        continue;
                     }

                     hypre_SStructBoxManEntryGetStrides(col_entries[j], cs, type);
                     hypre_SStructBoxManEntryGetGlobalRank(col_entries[j],
                                                           hypre_BoxIMin(intersection),
                                                           &s.col_base, type);
                     s.volume = hypre_BoxVolume(intersection);
                     if (s.volume > HYPRE_INT_MAX - nnz)
                     {
                        valid = 0;
                        break;
                     }

                     data_step = 1;
                     for (d = 0; d < 3; d++)
                     {
                        s.size[d]        = d < ndim ? hypre_BoxSizeD(intersection, d) : 1;
                        s.row_stride[d]  = d < ndim ? rs[d] * stride[d] : 0;
                        s.col_stride[d]  = d < ndim ? cs[d] : 0;
                        s.data_stride[d] = d < ndim ? data_step : 0;
                        if (d < ndim)
                        {
                           data_step *= hypre_BoxSizeD(data_box, d);
                        }
                     }

                     hypre_RefineBox(intersection, NULL, stride);
                     hypre_BoxShiftNeg(intersection, offset);
                     hypre_SStructBoxManEntryGetGlobalRank(row_entries[i],
                                                           hypre_BoxIMin(intersection),
                                                           &s.row_base, type);
                     hypre_CopyIndex(hypre_BoxIMin(intersection), data_index);
                     hypre_StructMatrixMapDataIndex(sA, data_index);
                     s.data_start = hypre_BoxIndexRank(data_box, data_index);
                     s.constant   = hypre_StructMatrixConstEntry(sA, sentry);
                     s.data       = hypre_StructMatrixBoxData(sA, g, sentry);
                     s.offset     = nnz;
                     nnz         += s.volume;

                     if (count == capacity)
                     {
                        capacity = capacity > HYPRE_INT_MAX / 2 ? HYPRE_INT_MAX :
                                   hypre_max(16, 2 * capacity);
                        segments = hypre_TReAlloc(segments, hypre_SStructMatrixCopySegment,
                                                  capacity, HYPRE_MEMORY_HOST);
                     }
                     segments[count++] = s;
                  }
                  hypre_TFree(col_entries, HYPRE_MEMORY_HOST);
               }
            }
            hypre_TFree(row_entries, HYPRE_MEMORY_HOST);
         }
      }
      hypre_BoxArrayDestroy(selected);
   }

   /* Free memory */
   hypre_BoxDestroy(clipped);
   hypre_BoxDestroy(region);
   hypre_BoxDestroy(map_box);
   hypre_BoxDestroy(to_box);
   hypre_BoxDestroy(intersection);

   *segments_ptr = segments;
   *count_ptr    = count;
   *nnz_ptr      = nnz;

   return valid;
}

/*--------------------------------------------------------------------------
 * Assembles the structured component of A on the rows in boxes (plus U, if
 * not NULL) directly as a ParCSR matrix. Returns zero on all ranks when the
 * layout is not supported or when result is already initialized.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SStructMatrixCopyBoxes( hypre_SStructMatrix  *A,
                              hypre_SStructGrid    *grid,
                              hypre_IJMatrix      **result,
                              hypre_BoxArray     ***boxes,
                              hypre_ParCSRMatrix   *U )
{
   MPI_Comm                         comm = hypre_SStructMatrixComm(A);
   HYPRE_IJMatrix                   ij_A = hypre_SStructMatrixIJMatrix(A);
   HYPRE_IJMatrix                   ij;
   hypre_SStructMatrixCopySegment  *segments = NULL;
   HYPRE_Int                        count    = 0;
   HYPRE_Int                        nnz      = 0;
   HYPRE_Int                        diag_nnz, offd_nnz;
   HYPRE_Int                        valid, all_valid;
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation             location = hypre_IJMatrixMemoryLocation(ij_A);
#endif

   valid = !*result &&
           !hypre_IJMatrixGlobalFirstRow(ij_A) &&
           !hypre_IJMatrixGlobalFirstCol(ij_A) &&
           hypre_SStructMatrixCopyLayout(A, grid, 0);
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      valid = 0;
   }
#endif
   if (valid)
   {
      valid = hypre_SStructMatrixBoxSegments(A, grid, boxes, -1, &segments, &count, &nnz);
   }

   if (U)
   {
      diag_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(U));
      offd_nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(U));
      if (diag_nnz > HYPRE_INT_MAX - nnz ||
          offd_nnz > HYPRE_INT_MAX - nnz - diag_nnz)
      {
         valid = 0;
      }
   }

   /* All ranks must select the same path before IJ's collective assembly */
   hypre_MPI_Allreduce(&valid, &all_valid, 1, hypre_MPI_INT, hypre_MPI_MIN, comm);
   if (!all_valid)
   {
      hypre_TFree(segments, HYPRE_MEMORY_HOST);

      return 0;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      HYPRE_IJMatrixPartialClone(ij_A, &ij);
      hypre_SStructMatrixCopyBoxesDevice(ij, segments, count, nnz, U);
      *result = ij;
      hypre_TFree(segments, HYPRE_MEMORY_HOST);

      return 1;
   }
#endif

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_HOST)
#endif
   {
      HYPRE_IJMatrixPartialClone(ij_A, &ij);
      hypre_SStructMatrixCopyBoxesHost(ij, segments, count, nnz, U);
      *result = ij;
      hypre_TFree(segments, HYPRE_MEMORY_HOST);

      return 1;
   }

   hypre_TFree(segments, HYPRE_MEMORY_HOST);

   return 0;
}

static HYPRE_Int
hypre_SStructMatrixBoxesToUMatrixGeneral( hypre_SStructMatrix   *A,
                                          hypre_SStructGrid     *grid,
                                          hypre_IJMatrix       **ij_Ahat_ptr,
                                          hypre_BoxArray      ***convert_boxa)
{
   HYPRE_Int              ndim            = hypre_SStructMatrixNDim(A);
   HYPRE_Int              nparts          = hypre_SStructMatrixNParts(A);
   HYPRE_Int             *Sentries        = hypre_SStructMatrixSEntries(A);
   HYPRE_IJMatrix         ij_A            = hypre_SStructMatrixIJMatrix(A);
   HYPRE_MemoryLocation   memory_location = hypre_IJMatrixMemoryLocation(ij_A);

   hypre_SStructPGrid    *pgrid;
   hypre_StructGrid      *sgrid;
   hypre_SStructStencil  *stencil;

   hypre_SStructPMatrix  *pA;
   hypre_IJMatrix        *ij_Ahat;
   hypre_AuxParCSRMatrix *aux_matrix;

   HYPRE_Int             *split;
   hypre_BoxArray        *grid_boxes;
   hypre_Box             *box       = hypre_BoxCreate(ndim);
   hypre_Box             *ghost_box = hypre_BoxCreate(ndim);
   hypre_Box             *grid_box;
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
   HYPRE_Int              m = 0;
   hypre_Index            loop_size;
   hypre_IndexRef         start;
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy  exec  = hypre_GetExecPolicy1(memory_location);
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
#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_HOST)
#endif
   {
      if (!*ij_Ahat_ptr)
      {
         row_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
      }
   }

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

#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_HOST)
#endif
               {
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
               }

               nvalues = hypre_max(nvalues, nnzrow * hypre_BoxVolume(convert_box));
            } /* Loop over convert_boxa[part][var] */

#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_HOST)
#endif
            {
               m += hypre_BoxVolume(ghost_box);
            }
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
      HYPRE_IJMatrixInitialize_v2(ij_Ahat, memory_location);
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

HYPRE_Int
hypre_SStructMatrixBoxesToUMatrix( hypre_SStructMatrix   *A,
                                   hypre_SStructGrid     *grid,
                                   hypre_IJMatrix       **result,
                                   hypre_BoxArray      ***boxes )
{
   if (!hypre_SStructMatrixCopyBoxes(A, grid, result, boxes, NULL))
   {
      hypre_SStructMatrixBoxesToUMatrixGeneral(A, grid, result, boxes);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Group equal x spans in a plane so interleaved face runs can merge along y
 *--------------------------------------------------------------------------*/

static hypre_int
hypre_SStructCompareNonzeroRuns( const void *left,
                                 const void *right )
{
   const hypre_Box  *a    = (const hypre_Box *) left;
   const hypre_Box  *b    = (const hypre_Box *) right;
   HYPRE_Int         ndim = hypre_BoxNDim(a);
   HYPRE_Int         keys_a[6];
   HYPRE_Int         keys_b[6];
   HYPRE_Int         k;

   keys_a[0] = keys_b[0] = 0;
   keys_a[1] = keys_b[1] = 0;
   if (ndim > 2)
   {
      keys_a[0] = hypre_BoxIMinD(a, 2);
      keys_a[1] = hypre_BoxIMaxD(a, 2);
      keys_b[0] = hypre_BoxIMinD(b, 2);
      keys_b[1] = hypre_BoxIMaxD(b, 2);
   }
   keys_a[2] = hypre_BoxIMinD(a, 0);
   keys_a[3] = hypre_BoxIMaxD(a, 0);
   keys_a[4] = hypre_BoxIMinD(a, 1);
   keys_a[5] = hypre_BoxIMaxD(a, 1);
   keys_b[2] = hypre_BoxIMinD(b, 0);
   keys_b[3] = hypre_BoxIMaxD(b, 0);
   keys_b[4] = hypre_BoxIMinD(b, 1);
   keys_b[5] = hypre_BoxIMaxD(b, 1);

   for (k = 0; k < 6; k++)
   {
      if (keys_a[k] < keys_b[k]) { return -1; }
      if (keys_a[k] > keys_b[k]) { return 1; }
   }

   return 0;
}

#if defined(HYPRE_USING_GPU)
/*--------------------------------------------------------------------------
 * Marks the rows that have entries in diag or offd
 *--------------------------------------------------------------------------*/

struct hypre_SStructNonzeroRunBoundary
{
   HYPRE_Int *marker;
   HYPRE_Int nrows;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   __host__ __device__
#endif
   bool operator()(HYPRE_Int i) const
   {
      return (i < nrows ? marker[i] : 0) != (i > 0 ? marker[i - 1] : 0);
   }
};

__global__ void
hypre_GPUKernelMarkNonzeroRows(hypre_DeviceItem &item,
                               HYPRE_Int  nrows,
                               HYPRE_Int *diag_i,
                               HYPRE_Int *offd_i,
                               HYPRE_Int *marker)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < nrows)
   {
      marker[i] = (diag_i[i + 1] > diag_i[i]) || (offd_i && offd_i[i + 1] > offd_i[i]);
   }
}
#endif

/*--------------------------------------------------------------------------
 * Shared support markers for halo conversion and boundary PtAP.
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU)
__global__ void
hypre_GPUKernelSStructMarkerValues(hypre_DeviceItem &item, HYPRE_Int n, HYPRE_Int *marker,
                                   HYPRE_Complex *values)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (i < n) { values[i] = (HYPRE_Complex) marker[i]; }
}
#endif


/*--------------------------------------------------------------------------
 * Marks the columns of P (including the ones owned by other ranks that are
 * needed for communication) in marker
 *--------------------------------------------------------------------------*/

static void
hypre_SStructMarkColumns( hypre_ParCSRMatrix *P,
                          HYPRE_Int          *marker )
{
   hypre_CSRMatrix      *diag = hypre_ParCSRMatrixDiag(P);
   HYPRE_Int            *cols = hypre_CSRMatrixJ(diag);
   HYPRE_Int             nnz  = hypre_CSRMatrixNumNonzeros(diag);
   hypre_ParCSRCommPkg  *pkg;
   HYPRE_Int             sends, j;

   if (!hypre_ParCSRMatrixCommPkg(P))
   {
      hypre_MatvecCommPkgCreate(P);
   }
   pkg   = hypre_ParCSRMatrixCommPkg(P);
   sends = hypre_ParCSRCommPkgSendMapStart(pkg, hypre_ParCSRCommPkgNumSends(pkg));

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(P)) == HYPRE_EXEC_DEVICE)
   {
      hypre_ScatterConstantDevice(marker, nnz, cols, (HYPRE_Int) 1);
      hypre_ParCSRCommPkgCopySendMapElmtsToDevice(pkg);
      hypre_ScatterConstantDevice(marker, sends,
                                  hypre_ParCSRCommPkgDeviceSendMapElmts(pkg), (HYPRE_Int) 1);
      return;
   }
#endif

   for (j = 0; j < nnz; j++)
   {
      marker[cols[j]] = 1;
   }
   for (j = 0; j < sends; j++)
   {
      marker[hypre_ParCSRCommPkgSendMapElmt(pkg, j)] = 1;
   }
}

static hypre_int
hypre_SStructCompareNonzeroPlanes( const void *left,
                                   const void *right )
{
   const hypre_Box  *a = (const hypre_Box *) left;
   const hypre_Box  *b = (const hypre_Box *) right;
   HYPRE_Int         d;

   for (d = 0; d < 3; d++)
   {
      if (hypre_BoxIMinD(a, d) < hypre_BoxIMinD(b, d)) { return -1; }
      if (hypre_BoxIMinD(a, d) > hypre_BoxIMinD(b, d)) { return 1; }
      if (hypre_BoxIMaxD(a, d) < hypre_BoxIMaxD(b, d)) { return -1; }
      if (hypre_BoxIMaxD(a, d) > hypre_BoxIMaxD(b, d)) { return 1; }
   }

   return 0;
}


/*--------------------------------------------------------------------------
 * Appends box to boxes, merging it along y with one of the last few boxes when
 * they are contiguous and share the same x (and z) extents
 *--------------------------------------------------------------------------*/

static void
hypre_SStructAppendNonzeroBox( hypre_Box      *box,
                               hypre_BoxArray *boxes )
{
   HYPRE_Int   ndim = hypre_BoxNDim(box);
   HYPRE_Int   size = hypre_BoxArraySize(boxes);
   hypre_Box  *prev;
   HYPRE_Int   t;

   for (t = size - 1; ndim > 1 && t >= hypre_max(0, size - 8); t--)
   {
      prev = hypre_BoxArrayBox(boxes, t);
      if (hypre_BoxIMinD(box, 0) == hypre_BoxIMinD(prev, 0) &&
          hypre_BoxIMaxD(box, 0) == hypre_BoxIMaxD(prev, 0) &&
          hypre_BoxIMinD(box, 1) == hypre_BoxIMaxD(prev, 1) + 1 &&
          (ndim < 3 || (hypre_BoxIMinD(box, 2) == hypre_BoxIMinD(prev, 2) &&
                        hypre_BoxIMaxD(box, 2) == hypre_BoxIMaxD(prev, 2))))
      {
         hypre_BoxIMaxD(prev, 1) = hypre_BoxIMaxD(box, 1);
         return;
      }
   }
   hypre_AppendBox(box, boxes);
}

/*--------------------------------------------------------------------------
 * Merges contiguous boxes along each axis, then eliminates duplicated boxes
 *--------------------------------------------------------------------------*/

static void
hypre_SStructUnionNonzeroBoxes( hypre_BoxArray *boxes )
{
   HYPRE_Int   ndim = hypre_BoxArrayNDim(boxes);
   hypre_Box  *box;
   hypre_Box  *prev;
   HYPRE_Int   axis, out, b, d, matches;

   if (hypre_BoxArraySize(boxes) < 2)
   {
      return;
   }

   for (axis = 1; axis < ndim; axis++)
   {
      out = 0;
      qsort(hypre_BoxArrayBoxes(boxes), hypre_BoxArraySize(boxes), sizeof(hypre_Box),
            axis == 1 ? hypre_SStructCompareNonzeroRuns : hypre_SStructCompareNonzeroPlanes);

      for (b = 0; b < hypre_BoxArraySize(boxes); b++)
      {
         box     = hypre_BoxArrayBox(boxes, b);
         prev    = out ? hypre_BoxArrayBox(boxes, out - 1) : NULL;
         matches = prev != NULL;
         for (d = 0; matches && d < ndim; d++)
         {
            if (d != axis && (hypre_BoxIMinD(box, d) != hypre_BoxIMinD(prev, d) ||
                              hypre_BoxIMaxD(box, d) != hypre_BoxIMaxD(prev, d)))
            {
               matches = 0;
            }
         }

         if (matches && hypre_BoxIMinD(box, axis) == hypre_BoxIMaxD(prev, axis) + 1)
         {
            hypre_BoxIMaxD(prev, axis) = hypre_BoxIMaxD(box, axis);
         }
         else
         {
            hypre_CopyBox(box, hypre_BoxArrayBox(boxes, out++));
         }
      }
      hypre_BoxArraySetSize(boxes, out);
   }
   hypre_UnionBoxes(boxes);
}



/*--------------------------------------------------------------------------
 * Returns the boxes of the rows marked in marker. SStruct local numbering is
 * ordered by part, variable and ghost box.
 *--------------------------------------------------------------------------*/

static hypre_BoxArray ***
hypre_SStructMarkerBoxes( hypre_SStructGrid   *grid,
                          HYPRE_Int           *marker,
                          HYPRE_Int            nrows,
                          HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int             ndim   = hypre_SStructGridNDim(grid);
   HYPRE_Int             nparts = hypre_SStructGridNParts(grid);

   hypre_SStructPGrid   *pgrid;
   hypre_StructGrid     *sgrid;
   hypre_BoxArray     ***convert_boxa;
   hypre_BoxArray       *grid_boxes;
   hypre_BoxArray       *run_boxa;
   hypre_Box            *ghost_box;
   hypre_Box            *run_box;
   hypre_Index           index;
   HYPRE_Int            *rows;
   HYPRE_Int             num_rows = 0;
   HYPRE_Int             part, var, nvars, b, d, i, k, m, r, volume;

   HYPRE_UNUSED_VAR(memory_location);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      HYPRE_Int *device_rows = hypre_TAlloc(HYPRE_Int, nrows + 1, memory_location);
      HYPRE_Int *end;
#if defined(HYPRE_USING_SYCL)
      oneapi::dpl::counting_iterator<HYPRE_Int> count(0);
      end = HYPRE_ONEDPL_CALL(std::copy_if, count, count + nrows + 1, device_rows,
                              (hypre_SStructNonzeroRunBoundary {marker, nrows}));
#else
      end = HYPRE_THRUST_CALL(copy_if, thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(nrows + 1), device_rows,
                              (hypre_SStructNonzeroRunBoundary {marker, nrows}));
#endif
      num_rows = (HYPRE_Int) (end - device_rows);
      rows = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(rows, device_rows, HYPRE_Int, num_rows, HYPRE_MEMORY_HOST, memory_location);
      hypre_TFree(device_rows, memory_location);
   }
   else
#endif
   {
      rows = hypre_TAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
      for (i = 0; i <= nrows; i++)
      {
         if ((i < nrows ? marker[i] : 0) != (i > 0 ? marker[i - 1] : 0))
         {
            rows[num_rows++] = i;
         }
      }
   }
   /* Local rows are ordered by part, variable and ghost box. Transfer only
    * the endpoints of nonzero runs, then tile each interval with aligned
    * rectangles, without visiting every selected row on the host. */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Find boxes");
   ghost_box    = hypre_BoxCreate(ndim);
   run_box      = hypre_BoxCreate(ndim);
   run_boxa     = hypre_BoxArrayCreate(0, ndim);
   convert_boxa = hypre_TAlloc(hypre_BoxArray **, nparts, HYPRE_MEMORY_HOST);
   hypre_SetIndex(index, 0);
   m = r = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      convert_boxa[part] = hypre_TAlloc(hypre_BoxArray *, nvars, HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         sgrid      = hypre_SStructPGridSGrid(pgrid, var);
         grid_boxes = hypre_StructGridBoxes(sgrid);
         convert_boxa[part][var] = hypre_BoxArrayCreate(0, ndim);

         hypre_ForBoxI(b, grid_boxes)
         {
            hypre_CopyBox(hypre_BoxArrayBox(grid_boxes, b), ghost_box);
            hypre_BoxGrowByArray(ghost_box, hypre_StructGridNumGhost(sgrid));
            volume = hypre_BoxVolume(ghost_box);

            hypre_BoxArraySetSize(run_boxa, 0);
            while (r < num_rows && rows[r] < m + volume)
            {
               HYPRE_Int end = hypre_min(rows[r + 1], m + volume) - m;
               k = hypre_max(rows[r], m) - m;
               while (k < end)
               {
                  HYPRE_Int linear = k, span;
                  hypre_Index extent;
                  hypre_SetIndex(extent, 1);
                  for (d = 0; d < ndim; d++)
                  {
                     hypre_IndexD(index, d) = hypre_BoxIMinD(ghost_box, d) +
                                              linear % hypre_BoxSizeD(ghost_box, d);
                     linear /= hypre_BoxSizeD(ghost_box, d);
                  }
                  extent[0] = hypre_min(hypre_BoxIMaxD(ghost_box, 0) - index[0] + 1, end - k);
                  span = extent[0];
                  for (d = 1; d < ndim; d++)
                  {
                     if (index[d - 1] != hypre_BoxIMinD(ghost_box, d - 1) ||
                         extent[d - 1] != hypre_BoxSizeD(ghost_box, d - 1)) { break; }
                     extent[d] = hypre_min(hypre_BoxIMaxD(ghost_box, d) - index[d] + 1,
                                           (end - k) / span);
                     span *= extent[d];
                  }
                  hypre_BoxSetExtents(run_box, index, index);
                  for (d = 0; d < ndim; d++)
                  {
                     hypre_BoxIMaxD(run_box, d) += extent[d] - 1;
                  }
                  hypre_SStructAppendNonzeroBox(run_box, run_boxa);
                  k += span;
               }
               if (rows[r + 1] > m + volume) { break; }
               r += 2;
            }
            hypre_SStructUnionNonzeroBoxes(run_boxa);
            hypre_AppendBoxArray(run_boxa, convert_boxa[part][var]);
            m += volume;
         }
      } /* loop over vars */
   } /* loop over parts */
   hypre_BoxDestroy(ghost_box);
   hypre_BoxDestroy(run_box);
   hypre_BoxArrayDestroy(run_boxa);
   hypre_TFree(rows, HYPRE_MEMORY_HOST);
   HYPRE_ANNOTATE_REGION_END("%s", "Find boxes");

   return convert_boxa;
}

static void
hypre_SStructFreeSupport( hypre_SStructGrid  *grid,
                          hypre_BoxArray   ***boxes )
{
   HYPRE_Int  nparts = hypre_SStructGridNParts(grid);
   HYPRE_Int  nvars;
   HYPRE_Int  part, var;

   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(hypre_SStructGridPGrid(grid, part));
      for (var = 0; var < nvars; var++)
      {
         hypre_BoxArrayDestroy(boxes[part][var]);
      }
      hypre_TFree(boxes[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(boxes, HYPRE_MEMORY_HOST);
}



/*--------------------------------------------------------------------------
 * Returns the boxes of the rows of B with entries (and also the rows given by
 * its columns when include_columns is set). If row_support is not NULL, the
 * boxes of the rows only are also returned in it.
 *--------------------------------------------------------------------------*/

static hypre_BoxArray ***
hypre_SStructCollectSupport( hypre_SStructGrid   *grid,
                             hypre_ParCSRMatrix  *B,
                             HYPRE_Int            include_columns,
                             hypre_BoxArray    ****row_support )
{
   hypre_CSRMatrix       *diag     = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix       *offd     = hypre_ParCSRMatrixOffd(B);
   HYPRE_Int             *di       = hypre_CSRMatrixI(diag);
   HYPRE_Int             *oi       = hypre_CSRMatrixI(offd);
   HYPRE_Int              nrows    = hypre_CSRMatrixNumRows(diag);
   HYPRE_MemoryLocation   location = hypre_ParCSRMatrixMemoryLocation(B);

   hypre_BoxArray      ***boxes;
   HYPRE_Int             *marker;
   HYPRE_Int              i;

   marker = hypre_TAlloc(HYPRE_Int, nrows, location);

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      dim3 block  = hypre_GetDefaultDeviceBlockDimension();
      dim3 launch = hypre_GetDefaultDeviceGridDimension(nrows, "thread", block);

      HYPRE_GPU_LAUNCH(hypre_GPUKernelMarkNonzeroRows, launch, block, nrows, di, oi, marker);
   }
   else
#endif
   {
      for (i = 0; i < nrows; i++)
      {
         marker[i] = (di && di[i + 1] > di[i]) || (oi && oi[i + 1] > oi[i]);
      }
   }

   if (row_support)
   {
      *row_support = hypre_SStructMarkerBoxes(grid, marker, nrows, location);
   }
   if (include_columns)
   {
      hypre_SStructMarkColumns(B, marker);
   }
   boxes = hypre_SStructMarkerBoxes(grid, marker, nrows, location);

   hypre_TFree(marker, location);

   return boxes;
}

/*--------------------------------------------------------------------------
 * Moves the entry of U at element k of segment s into its coefficient in S
 * (adding to it when action > 0) and leaves a zero in U. For action < 0, only
 * clears the coefficient in S.
 *--------------------------------------------------------------------------*/

static inline
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__host__ __device__
#endif
void
hypre_SStructCompressEntry( hypre_SStructMatrixCopySegment *s,
                            HYPRE_Int                       k,
                            HYPRE_BigInt                    first_row,
                            HYPRE_BigInt                    first_col,
                            HYPRE_BigInt                    last_col,
                            HYPRE_Int                      *di,
                            HYPRE_Int                      *dj,
                            HYPRE_Complex                  *da,
                            HYPRE_Int                      *oi,
                            HYPRE_Int                      *oj,
                            HYPRE_Complex                  *oa,
                            HYPRE_BigInt                   *col_map,
                            HYPRE_Int                       num_cols_offd,
                            HYPRE_Int                       action )
{
   HYPRE_Int       x = k % s->size[0];
   HYPRE_Int       y = (k / s->size[0]) % s->size[1];
   HYPRE_Int       z = k / (s->size[0] * s->size[1]);
   HYPRE_Int       row;
   HYPRE_BigInt    col;
   HYPRE_Complex  *u, *coef;

   row = (HYPRE_Int) (s->row_base - first_row + (HYPRE_BigInt) x * s->row_stride[0] +
                      (HYPRE_BigInt) y * s->row_stride[1] + (HYPRE_BigInt) z * s->row_stride[2]);
   col = s->col_base + (HYPRE_BigInt) x * s->col_stride[0] +
         (HYPRE_BigInt) y * s->col_stride[1] + (HYPRE_BigInt) z * s->col_stride[2];

   coef = s->data + s->data_start + x * s->data_stride[0] + y * s->data_stride[1] +
          z * s->data_stride[2];
   if (action < 0)
   {
      *coef = 0.0;
      return;
   }

   u = hypre_SStructUMatrixFindEntry(row, col, first_col, last_col, di, dj, da, oi, oj, oa,
                                     col_map, num_cols_offd);
   if (u)
   {
      *coef = action ? *coef + *u : *u;
      *u = 0.0;
   }
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
__global__ void
hypre_GPUKernelSStructCompressUToS( hypre_DeviceItem               &item,
                                    hypre_SStructMatrixCopySegment *segments,
                                    HYPRE_Int                       nsegments,
                                    HYPRE_Int                       nnz,
                                    HYPRE_BigInt                    first_row,
                                    HYPRE_BigInt                    first_col,
                                    HYPRE_BigInt                    last_col,
                                    HYPRE_Int                      *di,
                                    HYPRE_Int                      *dj,
                                    HYPRE_Complex                  *da,
                                    HYPRE_Int                      *oi,
                                    HYPRE_Int                      *oj,
                                    HYPRE_Complex                  *oa,
                                    HYPRE_BigInt                   *col_map,
                                    HYPRE_Int                       num_cols_offd,
                                    HYPRE_Int                       action )
{
   HYPRE_Int  i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   HYPRE_Int  lo, hi, mid;

   if (i < nnz)
   {
      /* Find the segment of this entry by binary search */
      lo = 0;
      hi = nsegments;
      while (lo + 1 < hi)
      {
         mid = lo + (hi - lo) / 2;
         if (segments[mid].offset <= i)
         {
            lo = mid;
         }
         else
         {
            hi = mid;
         }
      }

      hypre_SStructCompressEntry(segments + lo, i - segments[lo].offset, first_row,
                                 first_col, last_col, di, dj, da, oi, oj, oa, col_map,
                                 num_cols_offd, action);
   }
}
#endif

/*--------------------------------------------------------------------------
 * Adds to (action > 0) or sets (action = 0) the coefficients of S with the
 * entries of U at the same stencil positions within a part, on host or device,
 * and zeros these entries in U. Couplings to neighbor parts stay in U and their
 * coefficients in S are cleared. Missing U entries retain their S values.
 * Supports square matrices with unit strides and one variable per part;
 * returns zero (locally) otherwise.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_SStructMatrixCompressUToSBox( hypre_SStructMatrix *A,
                                    HYPRE_Int            action )
{
   hypre_SStructGraph              *graph         = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid               *grid          = hypre_SStructGraphGrid(graph);
   hypre_ParCSRMatrix              *U             = hypre_SStructMatrixParCSRMatrix(A);
   hypre_CSRMatrix                 *ud            = hypre_ParCSRMatrixDiag(U);
   hypre_CSRMatrix                 *uo            = hypre_ParCSRMatrixOffd(U);
   HYPRE_Int                       *di            = hypre_CSRMatrixI(ud);
   HYPRE_Int                       *dj            = hypre_CSRMatrixJ(ud);
   HYPRE_Complex                   *da            = hypre_CSRMatrixData(ud);
   HYPRE_Int                       *oi            = hypre_CSRMatrixI(uo);
   HYPRE_Int                       *oj            = hypre_CSRMatrixJ(uo);
   HYPRE_Complex                   *oa            = hypre_CSRMatrixData(uo);
   HYPRE_BigInt                    *col_map       = hypre_ParCSRMatrixColMapOffd(U);
   HYPRE_Int                        num_cols_offd = hypre_CSRMatrixNumCols(uo);
   HYPRE_BigInt                     first_row     = hypre_ParCSRMatrixFirstRowIndex(U);
   HYPRE_BigInt                     first_col     = hypre_ParCSRMatrixFirstColDiag(U);
   HYPRE_BigInt                     last_col      = hypre_ParCSRMatrixLastColDiag(U);
#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation             location      = hypre_ParCSRMatrixMemoryLocation(U);
#endif

   hypre_SStructMatrixCopySegment  *segments[2]   = {NULL, NULL};
   hypre_BoxArray                ***boxes;
   HYPRE_Int                        count[2], nnz[2];
   HYPRE_Int                        valid, m, e, i, m_action;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_SStructMatrixCopySegment  *d_segments;
   dim3                             block, launch;
#endif

   if (!hypre_CSRMatrixNumNonzeros(uo))
   {
      num_cols_offd = 0;
   }

   if (!hypre_SStructMatrixCopyLayout(A, grid, 1))
   {
      return 0;
   }
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      return 0;
   }
#endif

   /* Segments of the rows of U with entries: couplings within a part (moved
      from U to S) and couplings to neighbor parts (cleared in S). Both share
      the indexing, periodic-image filtering and overflow checks. */
   boxes = hypre_SStructCollectSupport(grid, U, 0, NULL);
   valid = hypre_SStructMatrixBoxSegments(A, grid, boxes, 0, &segments[0], &count[0], &nnz[0]);
   valid = hypre_SStructMatrixBoxSegments(A, grid, boxes, 1, &segments[1], &count[1], &nnz[1]) &&
           valid;
   hypre_SStructFreeSupport(grid, boxes);
   if (!valid)
   {
      hypre_TFree(segments[0], HYPRE_MEMORY_HOST);
      hypre_TFree(segments[1], HYPRE_MEMORY_HOST);

      return 0;
   }

   for (m = 0; m < 2; m++)
   {
      m_action = m ? -1 : action;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
      {
         if (nnz[m])
         {
            d_segments = hypre_TAlloc(hypre_SStructMatrixCopySegment, count[m], location);
            block      = hypre_GetDefaultDeviceBlockDimension();
            launch     = hypre_GetDefaultDeviceGridDimension(nnz[m], "thread", block);

            if (num_cols_offd)
            {
               hypre_ParCSRMatrixCopyColMapOffdToDevice(U);
               col_map = hypre_ParCSRMatrixDeviceColMapOffd(U);
            }
            hypre_TMemcpy(d_segments, segments[m], hypre_SStructMatrixCopySegment, count[m],
                          location, HYPRE_MEMORY_HOST);
            HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructCompressUToS, launch, block, d_segments,
                             count[m], nnz[m], first_row, first_col, last_col,
                             di, dj, da, oi, oj, oa, col_map, num_cols_offd, m_action);
            hypre_TFree(d_segments, location);
         }
      }
      else
#endif
      {
         for (e = 0; e < count[m]; e++)
         {
            for (i = 0; i < segments[m][e].volume; i++)
            {
               hypre_SStructCompressEntry(segments[m] + e, i, first_row, first_col, last_col,
                                          di, dj, da, oi, oj, oa, col_map, num_cols_offd,
                                          m_action);
            }
         }
      }
      hypre_TFree(segments[m], HYPRE_MEMORY_HOST);
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * Converts sA only on the row/column endpoints of B. The row space of A must
 * match both spaces of B, so these rows suffice for sA^T*B*sA.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixHaloToUMatrix( hypre_SStructMatrix  *A,
                                  hypre_SStructMatrix  *B,
                                  hypre_IJMatrix      **result )
{
   hypre_SStructGrid   *grid  = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(A));
   hypre_SStructGrid   *bgrid = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(B));
   hypre_ParCSRMatrix  *aU    = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParCSRMatrix  *U     = hypre_SStructMatrixParCSRMatrix(B);
   hypre_BoxArray    ***boxes;
   HYPRE_Int            valid, all_valid;

   /* SSAMG passes the same grid object; other grids must have the same boxes */
   valid = (grid == bgrid || hypre_SStructBoundaryGridMatches(grid, bgrid)) &&
           hypre_SStructGraphDomGrid(hypre_SStructMatrixGraph(B)) == bgrid &&
           hypre_ParCSRMatrixNumRows(U) == hypre_ParCSRMatrixNumRows(aU) &&
           hypre_ParCSRMatrixFirstRowIndex(U) == hypre_ParCSRMatrixFirstRowIndex(aU) &&
           hypre_ParCSRMatrixGlobalNumRows(U) == hypre_ParCSRMatrixGlobalNumCols(U) &&
           hypre_ParCSRMatrixFirstRowIndex(U) == hypre_ParCSRMatrixFirstColDiag(U);
   hypre_MPI_Allreduce(&valid, &all_valid, 1, hypre_MPI_INT, hypre_MPI_MIN,
                       hypre_SStructMatrixComm(A));

   if (!all_valid)
   {
      /* U's local numbering cannot be decoded with A's ghost boxes. Convert
         all of sA with its own grid rather than interpreting U's row indices. */
      *result = hypre_SStructMatrixToUMatrix(A, 0);

      return hypre_error_flag;
   }

   boxes = hypre_SStructCollectSupport(grid, U, 1, NULL);
   hypre_SStructMatrixBoxesToUMatrix(A, grid, result, boxes);
   hypre_SStructFreeSupport(grid, boxes);

   return hypre_error_flag;
}


static hypre_ParCSRMatrix *
hypre_SStructRestrictColumns( hypre_ParCSRMatrix *P,
                              HYPRE_Int          *marker )
{
   HYPRE_MemoryLocation  location = hypre_ParCSRMatrixMemoryLocation(P);
   HYPRE_Int             ncols    = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(P));

   hypre_ParCSRMatrix   *left;
   hypre_ParVector      *diagonal;
   hypre_CSRMatrix      *csr;
   hypre_CSRMatrix      *trimmed;
   HYPRE_Complex        *data;
   HYPRE_Int             i, k;

   /* Scale the columns of P by the marker */
   diagonal = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(P),
                                    hypre_ParCSRMatrixGlobalNumCols(P),
                                    hypre_ParCSRMatrixColStarts(P));
   hypre_ParVectorInitialize_v2(diagonal, location);
   data = hypre_VectorData(hypre_ParVectorLocalVector(diagonal));

#if defined(HYPRE_USING_GPU)
   if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
   {
      dim3 block  = hypre_GetDefaultDeviceBlockDimension();
      dim3 launch = hypre_GetDefaultDeviceGridDimension(ncols, "thread", block);

      HYPRE_GPU_LAUNCH(hypre_GPUKernelSStructMarkerValues, launch, block, ncols, marker, data);
   }
   else
#endif
   {
      for (i = 0; i < ncols; i++)
      {
         data[i] = (HYPRE_Complex) marker[i];
      }
   }

   left = hypre_ParCSRMatrixClone_v2(P, 1, location);
   hypre_ParCSRMatrixDiagScale(left, NULL, diagonal);

   /* Remove the zeros created by the scaling */
   for (k = 0; k < 2; k++)
   {
      csr = k ? hypre_ParCSRMatrixOffd(left) : hypre_ParCSRMatrixDiag(left);

#if defined(HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(location) == HYPRE_EXEC_DEVICE)
      {
         trimmed = hypre_CSRMatrixDeleteZerosDevice(csr, 0.0);
      }
      else
#endif
      {
         trimmed = hypre_CSRMatrixDeleteZeros(csr, 0.0);
      }

      if (trimmed)
      {
         hypre_CSRMatrixDestroy(csr);
         if (k)
         {
            hypre_ParCSRMatrixOffd(left) = trimmed;
         }
         else
         {
            hypre_ParCSRMatrixDiag(left) = trimmed;
         }
         hypre_CSRMatrixSetRownnz(trimmed);
      }
   }
   hypre_ParCSRMatrixSetNumNonzeros(left);
   hypre_ParVectorDestroy(diagonal);

   return left;
}

/* Appends the boxes of src shifted by minus each stencil offset, that is, the
 * rows coupled by the stencil to the indices in src. */
static void
hypre_SStructAppendStencilPreimage( hypre_BoxArray       *src,
                                    hypre_SStructStencil *stencil,
                                    hypre_BoxArray       *dst )
{
   hypre_Box  *box;
   HYPRE_Int   b, e;

   box = hypre_BoxCreate(hypre_BoxArrayNDim(src));
   for (b = 0; b < hypre_BoxArraySize(src); b++)
   {
      for (e = 0; e < hypre_SStructStencilSize(stencil); e++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(src, b), box);
         hypre_BoxShiftNeg(box, hypre_SStructStencilShape(stencil)[e]);
         hypre_AppendBox(box, dst);
      }
   }
   hypre_BoxDestroy(box);
}

static HYPRE_Int
hypre_SStructBoundaryGridMatches( hypre_SStructGrid *a,
                                  hypre_SStructGrid *b )
{
   HYPRE_Int           ndim   = hypre_SStructGridNDim(a);
   HYPRE_Int           nparts = hypre_SStructGridNParts(a);

   hypre_SStructPGrid *apgrid;
   hypre_SStructPGrid *bpgrid;
   hypre_StructGrid   *asgrid;
   hypre_StructGrid   *bsgrid;
   hypre_BoxArray     *aboxes;
   hypre_BoxArray     *bboxes;
   hypre_Box          *abox;
   hypre_Box          *bbox;
   HYPRE_Int           part, d;

   if (nparts != hypre_SStructGridNParts(b) || ndim != hypre_SStructGridNDim(b))
   {
      return 0;
   }

   for (part = 0; part < nparts; part++)
   {
      apgrid = hypre_SStructGridPGrid(a, part);
      bpgrid = hypre_SStructGridPGrid(b, part);
      if (hypre_SStructPGridNVars(apgrid) != 1 ||
          hypre_SStructPGridNVars(bpgrid) != 1 ||
          hypre_SStructPGridVarType(apgrid, 0) != HYPRE_SSTRUCT_VARIABLE_CELL ||
          hypre_SStructPGridVarType(bpgrid, 0) != HYPRE_SSTRUCT_VARIABLE_CELL)
      {
         return 0;
      }

      asgrid = hypre_SStructPGridSGrid(apgrid, 0);
      bsgrid = hypre_SStructPGridSGrid(bpgrid, 0);
      aboxes = hypre_StructGridBoxes(asgrid);
      bboxes = hypre_StructGridBoxes(bsgrid);
      if (hypre_BoxArraySize(aboxes) > 1 ||
          hypre_BoxArraySize(aboxes) != hypre_BoxArraySize(bboxes))
      {
         return 0;
      }

      abox = hypre_BoxArraySize(aboxes) ? hypre_BoxArrayBox(aboxes, 0) : NULL;
      bbox = hypre_BoxArraySize(bboxes) ? hypre_BoxArrayBox(bboxes, 0) : NULL;
      for (d = 0; d < ndim; d++)
      {
         if (hypre_StructGridPeriodic(asgrid)[d] ||
             hypre_StructGridPeriodic(bsgrid)[d] ||
             hypre_StructGridNumGhost(asgrid)[2 * d] != hypre_StructGridNumGhost(bsgrid)[2 * d] ||
             hypre_StructGridNumGhost(asgrid)[2 * d + 1] != hypre_StructGridNumGhost(bsgrid)[2 * d + 1])
         {
            return 0;
         }

         if (abox &&
             (hypre_BoxIMinD(abox, d) != hypre_BoxIMinD(bbox, d) ||
              hypre_BoxIMaxD(abox, d) != hypre_BoxIMaxD(bbox, d)))
         {
            return 0;
         }
      }
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * Converts the structured component of A on the rows in boxes and adds the
 * unstructured component of A (U) to it.
 *--------------------------------------------------------------------------*/

static void
hypre_SStructBoxesWithU( hypre_SStructMatrix  *A,
                         hypre_BoxArray     ***boxes,
                         hypre_IJMatrix      **result )
{
   hypre_SStructGraph  *graph = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid   *grid  = hypre_SStructGraphGrid(graph);
   hypre_ParCSRMatrix  *U     = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParCSRMatrix  *structured, *full;

   if (!hypre_SStructMatrixCopyBoxes(A, grid, result, boxes, U))
   {
      hypre_SStructMatrixBoxesToUMatrixGeneral(A, grid, result, boxes);
      HYPRE_IJMatrixGetObject(*result, (void **) &structured);
      hypre_ParCSRMatrixAdd(1.0, structured, 1.0, U, &full);
      hypre_ParCSRMatrixDestroy(structured);
      hypre_IJMatrixObject(*result) = full;
   }
}

/*--------------------------------------------------------------------------
 * Computes the unstructured component of a PtAP product when both uA and uP
 * are nonzero, where sM = sP^T*sA*sP has already been computed natively.
 * Only the rows J of M that differ from sM are formed: complete rows of
 * P^T*A*P from the rows of A and P that reach J. They become uM, and the
 * caller must then set the stencil entries of sM from them with
 * hypre_SStructMatrixCompressUToS(M, -1). Support boxes are exchanged between
 * neighboring ranks only.
 *
 * Returns zero on all ranks when the layout is not supported.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatrixPtAPBoundary( hypre_SStructMatrix  *A,
                                 hypre_SStructMatrix  *P,
                                 hypre_SStructMatrix  *M,
                                 hypre_ParCSRMatrix  **result )
{
   MPI_Comm                       comm   = hypre_SStructMatrixComm(A);
   HYPRE_Int                      nparts = hypre_SStructMatrixNParts(A);
   HYPRE_Int                      ndim   = hypre_SStructMatrixNDim(A);
   hypre_SStructGrid             *fine   = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(A));
   hypre_SStructGrid             *pgrid  = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(P));
   hypre_SStructGrid             *coarse = hypre_SStructGraphDomGrid(hypre_SStructMatrixGraph(P));
   hypre_SStructGrid             *mgrid  = hypre_SStructGraphGrid(hypre_SStructMatrixGraph(M));
   hypre_ParCSRMatrix            *uA     = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParCSRMatrix            *uP     = hypre_SStructMatrixParCSRMatrix(P);
   HYPRE_MemoryLocation           location = hypre_ParCSRMatrixMemoryLocation(uP);

   /* Per-part matrix data */
   hypre_SStructPMatrix          *pa, *pp, *pm;
   hypre_SStructStencil          *astencil, *pstencil, *mstencil;
   hypre_StructMatrix            *sm;
   hypre_StructGrid              *sgrid;
   hypre_IndexRef                 stride;
   hypre_BoxArray                *fboxes, *cboxes, *refined;
   hypre_Box                     *box;
   HYPRE_Int                     *msplit;

   /* reach[0:6] is the stencil reach of A, reach[6:12] the distance from the
      local fine boxes up to which support boxes can extend */
   HYPRE_Int                      reach[12] = {0};
   HYPRE_Int                      global_reach[12];
   HYPRE_Int                     *global_halo = global_reach;
   HYPRE_Int                     *nbr_halo    = global_reach + 6;
   HYPRE_Int                      valid, all_valid, known, preach;

   /* Support boxes and intermediate matrices */
   hypre_SStructGridRemoteBoxes  *rboxes;
   hypre_BoxArray              ***pu_rows, ***boxes, ***right, ***left, ***rows_J;
   hypre_IJMatrix                *ij_s = NULL, *ij_a = NULL, *ij_p = NULL;
   hypre_ParCSRMatrix            *a, *p, *restricted, *rap;
   HYPRE_Int                     *marker;
   HYPRE_Int                      ncols;
   HYPRE_Int                      part, d, e, b;

   if (A == P || ndim < 1 || ndim > 3 ||
       nparts != hypre_SStructMatrixNParts(P) ||
       ndim != hypre_SStructMatrixNDim(P) ||
       hypre_SStructGraphDomGrid(hypre_SStructMatrixGraph(A)) != fine)
   {
      return 0;
   }
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP)
   if (hypre_GetExecPolicy1(hypre_SStructMatrixMemoryLocation(A)) == HYPRE_EXEC_DEVICE)
   {
      return 0;
   }
#endif

   /*------------------------------------------------------
    * Check the layout of A, P and M
    *------------------------------------------------------*/

   box   = hypre_BoxCreate(ndim);
   valid = hypre_SStructBoundaryGridMatches(fine, pgrid) &&
           hypre_SStructBoundaryGridMatches(coarse, mgrid);
   if (hypre_ParCSRMatrixMemoryLocation(uA) != location ||
       hypre_ParCSRMatrixNumRows(uA) != hypre_ParCSRMatrixNumRows(uP) ||
       hypre_ParCSRMatrixFirstRowIndex(uA) != hypre_ParCSRMatrixFirstRowIndex(uP) ||
       hypre_IJMatrixRowPartitioning(hypre_SStructMatrixIJMatrix(M))[0] !=
       hypre_ParCSRMatrixFirstColDiag(uP))
   {
      valid = 0;
   }

   for (part = 0; valid && part < nparts; part++)
   {
      pa = hypre_SStructMatrixPMatrix(A, part);
      pp = hypre_SStructMatrixPMatrix(P, part);
      pm = hypre_SStructMatrixPMatrix(M, part);
      if (hypre_SStructPMatrixNVars(pa) != 1 ||
          hypre_SStructPMatrixNVars(pp) != 1 ||
          hypre_SStructPMatrixNVars(pm) != 1)
      {
         valid = 0;
         break;
      }

      stride   = hypre_SStructPMatrixDomainStride(pp);
      sm       = hypre_SStructPMatrixSMatrix(pm, 0, 0);
      astencil = hypre_SStructPMatrixStencil(pa, 0);
      pstencil = hypre_SStructPMatrixStencil(pp, 0);
      mstencil = hypre_SStructPMatrixStencil(pm, 0);
      fboxes   = hypre_StructGridBoxes(hypre_SStructPGridSGrid(hypre_SStructGridPGrid(fine, part), 0));
      cboxes   = hypre_StructGridBoxes(hypre_SStructPGridSGrid(hypre_SStructGridPGrid(coarse, part), 0));
      msplit   = hypre_SStructMatrixSplit(M, part, 0);

      /* Strides */
      for (d = 0; d < ndim; d++)
      {
         if (hypre_SStructPMatrixDomainStride(pa)[d] != 1 ||
             hypre_SStructPMatrixRangeStride(pa)[d] != 1 ||
             hypre_SStructPMatrixRangeStride(pp)[d] != 1 || stride[d] < 1 ||
             hypre_SStructPMatrixDomainStride(pm)[d] != 1 ||
             hypre_SStructPMatrixRangeStride(pm)[d] != 1 ||
             hypre_StructMatrixDataStride(sm)[d] != 1)
         {
            valid = 0;
         }
      }

      /* Stencil reach of A and P */
      for (e = 0; e < hypre_SStructStencilSize(astencil); e++)
      {
         for (d = 0; d < ndim; d++)
         {
            reach[2 * d] = hypre_max(reach[2 * d],
                                     hypre_abs(hypre_SStructStencilShape(astencil)[e][d]));
         }
      }
      for (d = 0; d < ndim; d++)
      {
         preach = 0;
         for (e = 0; e < hypre_SStructStencilSize(pstencil); e++)
         {
            preach = hypre_max(preach, hypre_abs(hypre_SStructStencilShape(pstencil)[e][d]));
         }
         reach[2 * d + 1] = reach[2 * d];
         reach[6 + 2 * d] = reach[6 + 2 * d + 1] =
                               hypre_max(reach[6 + 2 * d], stride[d] - 1 + preach + reach[2 * d]);
      }

      /* Rows J of sM are overwritten by hypre_SStructMatrixCompressUToS */
      for (e = 0; e < hypre_StructStencilSize(hypre_StructMatrixStencil(sm)); e++)
      {
         if (hypre_StructMatrixConstEntry(sm, e) || hypre_StructMatrixSymmEntries(sm)[e] >= 0)
         {
            valid = 0;
         }
      }
      for (e = 0; e < hypre_SStructStencilSize(mstencil); e++)
      {
         if (msplit[e] < 0 || hypre_SStructStencilVars(mstencil)[e])
         {
            valid = 0;
         }
      }

      /* The fine rows of P with columns in the local coarse box must be near the local fine box */
      if (hypre_BoxArraySize(cboxes))
      {
         hypre_CopyBox(hypre_BoxArrayBox(cboxes, 0), box);
         hypre_RefineBox(box, NULL, stride);
         for (d = 0; valid && d < ndim; d++)
         {
            if (!hypre_BoxArraySize(fboxes) ||
                hypre_BoxIMinD(box, d) < hypre_BoxIMinD(hypre_BoxArrayBox(fboxes, 0), d) - stride[d] + 1 ||
                hypre_BoxIMaxD(box, d) > hypre_BoxIMaxD(hypre_BoxArrayBox(fboxes, 0), d) + stride[d] - 1)
            {
               valid = 0;
            }
         }
      }
   }
   hypre_MPI_Allreduce(reach, global_reach, 12, hypre_MPI_INT, hypre_MPI_MAX, comm);

   /* The box managers must know the boxes within nbr_halo of the local boxes.
      They hold the remote boxes known to the box managers of the struct grids. */
   for (part = 0; part < nparts; part++)
   {
      sgrid = hypre_SStructPGridSGrid(hypre_SStructGridPGrid(fine, part), 0);
      hypre_BoxManGetAllGlobalKnown(hypre_StructGridBoxMan(sgrid), &known);
      for (d = 0; !known && d < ndim; d++)
      {
         if (nbr_halo[2 * d] > hypre_StructGridMaxDistance(sgrid)[d])
         {
            valid = 0;
         }
      }
   }
   hypre_MPI_Allreduce(&valid, &all_valid, 1, hypre_MPI_INT, hypre_MPI_MIN, comm);
   if (!all_valid)
   {
      hypre_BoxDestroy(box);

      return 0;
   }

   /*------------------------------------------------------
    * Find the rows J of M that differ from sM
    *
    * uM has nonzero rows J: the columns of uP, and the columns of sP over the
    * rows of uA and over the rows coupled by sA to the rows of uP
    *------------------------------------------------------*/

   rboxes = hypre_SStructGridRemoteBoxesCreate(fine, nbr_halo);
   ncols  = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(uP));
   marker = hypre_CTAlloc(HYPRE_Int, ncols, location);

   /* Mark the rows of uA once, retaining both row-only and row/column support */
   pu_rows = hypre_SStructCollectSupport(pgrid, uP, 0, NULL);
   right   = hypre_SStructCollectSupport(fine, uA, 1, &boxes);
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructAppendStencilPreimage(pu_rows[part][0],
                                         hypre_SStructPMatrixStencil(hypre_SStructMatrixPMatrix(A, part), 0),
                                         boxes[part][0]);
   }
   hypre_SStructGridRemoteBoxesShare(fine, rboxes, boxes);
   hypre_SStructMatrixBoxesToUMatrix(P, pgrid, &ij_s, boxes);
   hypre_SStructFreeSupport(fine, boxes);
   hypre_SStructMarkColumns((hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij_s), marker);
   hypre_SStructMarkColumns(uP, marker);
   HYPRE_IJMatrixDestroy(ij_s);
   rows_J = hypre_SStructMarkerBoxes(coarse, marker, ncols, location);

   /*------------------------------------------------------
    * Rows of P with columns in J (left), and rows of P reached from them
    * by A (right)
    *------------------------------------------------------*/

   left = hypre_TAlloc(hypre_BoxArray **, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pp      = hypre_SStructMatrixPMatrix(P, part);
      refined = hypre_BoxArrayClone(rows_J[part][0]);
      for (b = 0; b < hypre_BoxArraySize(refined); b++)
      {
         hypre_RefineBox(hypre_BoxArrayBox(refined, b), NULL,
                         hypre_SStructPMatrixDomainStride(pp));
      }
      left[part]    = hypre_TAlloc(hypre_BoxArray *, 1, HYPRE_MEMORY_HOST);
      left[part][0] = hypre_BoxArrayCreate(0, ndim);
      hypre_SStructAppendStencilPreimage(refined, hypre_SStructPMatrixStencil(pp, 0),
                                         left[part][0]);
      hypre_AppendBoxArray(pu_rows[part][0], left[part][0]);
      hypre_BoxArrayDestroy(refined);
   }
   hypre_SStructFreeSupport(fine, pu_rows);
   hypre_SStructGridRemoteBoxesShare(fine, rboxes, left);

   for (part = 0; part < nparts; part++)
   {
      for (b = 0; b < hypre_BoxArraySize(left[part][0]); b++)
      {
         hypre_CopyBox(hypre_BoxArrayBox(left[part][0], b), box);
         hypre_BoxGrowByArray(box, global_halo);
         hypre_AppendBox(box, right[part][0]);
      }
   }
   hypre_SStructGridRemoteBoxesShare(fine, rboxes, right);

   /*------------------------------------------------------
    * Compute the complete rows J of P^T*A*P
    *------------------------------------------------------*/

   hypre_SStructBoxesWithU(A, left, &ij_a);
   hypre_SStructBoxesWithU(P, right, &ij_p);
   hypre_SStructFreeSupport(fine, left);
   hypre_SStructFreeSupport(fine, right);

   a = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij_a);
   p = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(ij_p);
   restricted = hypre_SStructRestrictColumns(p, marker);
   if (!hypre_ParCSRMatrixCommPkg(p))
   {
      hypre_MatvecCommPkgCreate(p);
   }
   rap = hypre_ParCSRMatrixRAPKT(restricted, a, p, 1, 0);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(restricted);
   HYPRE_IJMatrixDestroy(ij_a);
   HYPRE_IJMatrixDestroy(ij_p);
   hypre_TFree(marker, location);
   hypre_SStructFreeSupport(coarse, rows_J);
   hypre_SStructGridRemoteBoxesDestroy(rboxes);
   hypre_BoxDestroy(box);

   *result = rap;

   return 1;
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
