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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatrixCreate( MPI_Comm               comm,
                            hypre_SStructPGrid    *pgrid,
                            hypre_SStructStencil **stencils,
                            hypre_SStructPMatrix **pmatrix_ptr )
{
   hypre_SStructPMatrix  *pmatrix;
   HYPRE_Int              nvars;
   HYPRE_Int            **smaps;
   hypre_StructStencil ***sstencils;
   hypre_StructMatrix  ***smatrices;
   HYPRE_Int            **symmetric;

   hypre_StructStencil   *sstencil;
   HYPRE_Int             *vars;
   hypre_Index           *sstencil_shape;
   HYPRE_Int              sstencil_size;
   HYPRE_Int              new_dim;
   HYPRE_Int             *new_sizes;
   hypre_Index          **new_shapes;
   HYPRE_Int              size;
   hypre_StructGrid      *sgrid;

   HYPRE_Int              vi, vj;
   HYPRE_Int              i, j, k;

   pmatrix = hypre_TAlloc(hypre_SStructPMatrix,  1, HYPRE_MEMORY_HOST);

   hypre_SStructPMatrixComm(pmatrix)     = comm;
   hypre_SStructPMatrixPGrid(pmatrix)    = pgrid;
   hypre_SStructPMatrixStencils(pmatrix) = stencils;
   nvars = hypre_SStructPGridNVars(pgrid);
   hypre_SStructPMatrixNVars(pmatrix) = nvars;

   /* create sstencils */
   smaps     = hypre_TAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);
   sstencils = hypre_TAlloc(hypre_StructStencil **,  nvars, HYPRE_MEMORY_HOST);
   new_sizes  = hypre_TAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);
   new_shapes = hypre_TAlloc(hypre_Index *,  nvars, HYPRE_MEMORY_HOST);
   size = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      sstencils[vi] = hypre_TAlloc(hypre_StructStencil *,  nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sstencils[vi][vj] = NULL;
         new_sizes[vj] = 0;
      }

      sstencil       = hypre_SStructStencilSStencil(stencils[vi]);
      vars           = hypre_SStructStencilVars(stencils[vi]);
      sstencil_shape = hypre_StructStencilShape(sstencil);
      sstencil_size  = hypre_StructStencilSize(sstencil);

      smaps[vi] = hypre_TAlloc(HYPRE_Int,  sstencil_size, HYPRE_MEMORY_HOST);
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         new_sizes[j]++;
      }
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            new_shapes[vj] = hypre_TAlloc(hypre_Index,  new_sizes[vj], HYPRE_MEMORY_HOST);
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
      new_dim = hypre_StructStencilNDim(sstencil);
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            sstencils[vi][vj] =
               hypre_StructStencilCreate(new_dim, new_sizes[vj], new_shapes[vj]);
         }
         size = hypre_max(size, new_sizes[vj]);
      }
   }
   hypre_SStructPMatrixSMaps(pmatrix)     = smaps;
   hypre_SStructPMatrixSStencils(pmatrix) = sstencils;
   hypre_TFree(new_sizes, HYPRE_MEMORY_HOST);
   hypre_TFree(new_shapes, HYPRE_MEMORY_HOST);

   /* create smatrices */
   smatrices = hypre_TAlloc(hypre_StructMatrix **,  nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatrices[vi] = hypre_TAlloc(hypre_StructMatrix *,  nvars, HYPRE_MEMORY_HOST);
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

   /* create symmetric */
   symmetric = hypre_TAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      symmetric[vi] = hypre_TAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         symmetric[vi][vj] = 0;
      }
   }
   hypre_SStructPMatrixSymmetric(pmatrix) = symmetric;

   hypre_SStructPMatrixSEntriesSize(pmatrix) = size;
   hypre_SStructPMatrixSEntries(pmatrix) = hypre_TAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);

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
   HYPRE_Int               vi, vj;

   if (pmatrix)
   {
      hypre_SStructPMatrixRefCount(pmatrix) --;
      if (hypre_SStructPMatrixRefCount(pmatrix) == 0)
      {
         stencils  = hypre_SStructPMatrixStencils(pmatrix);
         nvars     = hypre_SStructPMatrixNVars(pmatrix);
         smaps     = hypre_SStructPMatrixSMaps(pmatrix);
         sstencils = hypre_SStructPMatrixSStencils(pmatrix);
         smatrices = hypre_SStructPMatrixSMatrices(pmatrix);
         symmetric = hypre_SStructPMatrixSymmetric(pmatrix);
         for (vi = 0; vi < nvars; vi++)
         {
            HYPRE_SStructStencilDestroy(stencils[vi]);
            hypre_TFree(smaps[vi], HYPRE_MEMORY_HOST);
            for (vj = 0; vj < nvars; vj++)
            {
               hypre_StructStencilDestroy(sstencils[vi][vj]);
               hypre_StructMatrixDestroy(smatrices[vi][vj]);
            }
            hypre_TFree(sstencils[vi], HYPRE_MEMORY_HOST);
            hypre_TFree(smatrices[vi], HYPRE_MEMORY_HOST);
            hypre_TFree(symmetric[vi], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(stencils, HYPRE_MEMORY_HOST);
         hypre_TFree(smaps, HYPRE_MEMORY_HOST);
         hypre_TFree(sstencils, HYPRE_MEMORY_HOST);
         hypre_TFree(smatrices, HYPRE_MEMORY_HOST);
         hypre_TFree(symmetric, HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_SStructPMatrixSEntries(pmatrix), HYPRE_MEMORY_HOST);
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
   hypre_StructMatrix   *smatrix;
   HYPRE_Int             vi, vj;
   /* HYPRE_Int             num_ghost[2*HYPRE_MAXDIM]; */
   /* HYPRE_Int             vi, vj, d, ndim; */

#if 0
   ndim = hypre_SStructPMatrixNDim(pmatrix);
   /* RDF: Why are the ghosts being reset to one? Maybe it needs to be at least
    * one to set shared coefficients correctly, but not exactly one? */
   for (d = 0; d < ndim; d++)
   {
      num_ghost[2 * d] = num_ghost[2 * d + 1] = 1;
   }
#endif
   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            HYPRE_StructMatrixSetSymmetric(smatrix, symmetric[vi][vj]);
            /* hypre_StructMatrixSetNumGhost(smatrix, num_ghost); */
            hypre_StructMatrixInitialize(smatrix);
            /* needed to get AddTo accumulation correct between processors */
            hypre_StructMatrixClearGhostValues(smatrix);
         }
      }
   }

   hypre_SStructPMatrixAccumulated(pmatrix) = 0;

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
   hypre_SyncCudaDevice(hypre_handle());
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

   /* if values already accumulated, just return */
   if (hypre_SStructPMatrixAccumulated(pmatrix))
   {
      return hypre_error_flag;
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
               num_ghost[2 * d]   = num_ghost[2 * d + 1] = hypre_IndexD(varoffset, d);
            }

            /* accumulate values from AddTo */
            hypre_CreateCommInfoFromNumGhost(sgrid, num_ghost, &comm_info);
            hypre_CommPkgCreate(comm_info,
                                hypre_StructMatrixDataSpace(smatrix),
                                hypre_StructMatrixDataSpace(smatrix),
                                hypre_StructMatrixNumValues(smatrix), NULL, 1,
                                hypre_StructMatrixComm(smatrix),
                                &comm_pkg);
            hypre_InitializeCommunication(comm_pkg,
                                          hypre_StructMatrixData(smatrix),
                                          hypre_StructMatrixData(smatrix),
                                          1, 0, &comm_handle);
            hypre_FinalizeCommunication(comm_handle);

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
 *--------------------------------------------------------------------------*/

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
            hypre_sprintf(new_filename, "%s.%02d.%02d", filename, vi, vj);
            hypre_StructMatrixPrint(new_filename, smatrix, all);
         }
      }
   }

   return hypre_error_flag;
}

/*==========================================================================
 * SStructUMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixInitialize( hypre_SStructMatrix *matrix )
{
   HYPRE_Int               ndim        = hypre_SStructMatrixNDim(matrix);
   HYPRE_IJMatrix          ijmatrix    = hypre_SStructMatrixIJMatrix(matrix);
   HYPRE_Int               matrix_type = hypre_SStructMatrixObjectType(matrix);
   hypre_SStructGraph     *graph       = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid      *grid        = hypre_SStructGraphGrid(graph);
   HYPRE_Int               nparts      = hypre_SStructGraphNParts(graph);
   hypre_SStructPGrid    **pgrids      = hypre_SStructGraphPGrids(graph);
   hypre_SStructStencil ***stencils    = hypre_SStructGraphStencils(graph);
   HYPRE_Int               nUventries  = hypre_SStructGraphNUVEntries(graph);
   HYPRE_Int              *iUventries  = hypre_SStructGraphIUVEntries(graph);
   hypre_SStructUVEntry  **Uventries   = hypre_SStructGraphUVEntries(graph);
   HYPRE_Int             **nvneighbors = hypre_SStructGridNVNeighbors(grid);
   hypre_StructGrid       *sgrid;
   hypre_SStructStencil   *stencil;
   HYPRE_Int              *split;
   HYPRE_Int               nvars;
   HYPRE_Int               nrows, rowstart, nnzs ;
   HYPRE_Int               part, var, entry, b, m, mi;
   HYPRE_Int              *row_sizes;
   HYPRE_Int               max_row_size;

   hypre_BoxArray         *boxes;
   hypre_Box              *box;
   hypre_Box              *ghost_box;
   hypre_IndexRef          start;
   hypre_Index             loop_size, stride;

   HYPRE_IJMatrixSetObjectType(ijmatrix, HYPRE_PARCSR);

#ifdef HYPRE_USING_OPENMP
   HYPRE_IJMatrixSetOMPFlag(ijmatrix, 1); /* Use OpenMP */
#endif

   if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
   {
      rowstart = hypre_SStructGridGhstartRank(grid);
      nrows = hypre_SStructGridGhlocalSize(grid) ;
   }
   else /* matrix_type == HYPRE_PARCSR */
   {
      rowstart = hypre_SStructGridStartRank(grid);
      nrows = hypre_SStructGridLocalSize(grid);
   }

   /* set row sizes */
   m = 0;
   max_row_size = 0;
   ghost_box = hypre_BoxCreate(ndim);
   row_sizes = hypre_CTAlloc(HYPRE_Int, nrows, HYPRE_MEMORY_HOST);
   hypre_SetIndex(stride, 1);
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[part]);
      for (var = 0; var < nvars; var++)
      {
         sgrid = hypre_SStructPGridSGrid(pgrids[part], var);

         stencil = stencils[part][var];
         split = hypre_SStructMatrixSplit(matrix, part, var);
         nnzs = 0;
         for (entry = 0; entry < hypre_SStructStencilSize(stencil); entry++)
         {
            if (split[entry] == -1)
            {
               nnzs++;
            }
         }
#if 0
         /* TODO: For now, assume stencil is full/complete */
         if (hypre_SStructMatrixSymmetric(matrix))
         {
            nnzs = 2 * nnzs - 1;
         }
#endif
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
            zypre_BoxLoop1Begin(hypre_SStructMatrixNDim(matrix), loop_size,
                                ghost_box, start, stride, mi);
            {
               row_sizes[m + mi] = nnzs;
            }
            zypre_BoxLoop1End(mi);

            m += hypre_BoxVolume(ghost_box);
         }

         max_row_size = hypre_max(max_row_size, nnzs);
         if (nvneighbors[part][var])
         {
            max_row_size =
               hypre_max(max_row_size, hypre_SStructStencilSize(stencil));
         }
      }
   }
   hypre_BoxDestroy(ghost_box);

   /* GEC0902 essentially for each UVentry we figure out how many extra columns
    * we need to add to the rowsizes                                   */

   /* RDF: THREAD? */
   for (entry = 0; entry < nUventries; entry++)
   {
      mi = iUventries[entry];
      m = hypre_SStructUVEntryRank(Uventries[mi]) - rowstart;
      if ((m > -1) && (m < nrows))
      {
         row_sizes[m] += hypre_SStructUVEntryNUEntries(Uventries[mi]);
         max_row_size = hypre_max(max_row_size, row_sizes[m]);
      }
   }

   /* ZTODO: Update row_sizes based on neighbor off-part couplings */
   HYPRE_IJMatrixSetRowSizes (ijmatrix, (const HYPRE_Int *) row_sizes);

   hypre_TFree(row_sizes, HYPRE_MEMORY_HOST);

   hypre_SStructMatrixTmpSize(matrix) = max_row_size;
   hypre_SStructMatrixTmpRowCoords(matrix) = hypre_CTAlloc(HYPRE_BigInt, max_row_size,
                                                           HYPRE_MEMORY_HOST);
   hypre_SStructMatrixTmpColCoords(matrix) = hypre_CTAlloc(HYPRE_BigInt, max_row_size,
                                                           HYPRE_MEMORY_HOST);
   hypre_SStructMatrixTmpCoeffs(matrix)    = hypre_CTAlloc(HYPRE_Complex, max_row_size,
                                                           HYPRE_MEMORY_HOST);

   HYPRE_IJMatrixInitialize(ijmatrix);
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
   HYPRE_Int                ndim     = hypre_SStructMatrixNDim(matrix);
   HYPRE_IJMatrix           ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph      *graph    = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid       *grid     = hypre_SStructGraphGrid(graph);
   hypre_SStructGrid       *dom_grid = hypre_SStructGraphDomainGrid(graph);
   hypre_SStructStencil    *stencil  = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int               *vars     = hypre_SStructStencilVars(stencil);
   hypre_Index             *shape    = hypre_SStructStencilShape(stencil);
   HYPRE_Int                size     = hypre_SStructStencilSize(stencil);
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
   HYPRE_Int                matrix_type = hypre_SStructMatrixObjectType(matrix);
   HYPRE_Complex           *h_values;
   HYPRE_MemoryLocation     memory_location = hypre_IJMatrixMemoryLocation(ijmatrix);

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
      return hypre_error_flag;
   }
   else
   {
      hypre_BoxManEntryGetInfo(boxman_entry, (void **) &entry_info);
   }

   hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index,
                                         &row_coord, matrix_type);

   col_coords = hypre_SStructMatrixTmpColCoords(matrix);
   coeffs = hypre_SStructMatrixTmpCoeffs(matrix);

   if ( hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE )
   {
      h_values = hypre_TAlloc(HYPRE_Complex, nentries, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_values, values, HYPRE_Complex, nentries, HYPRE_MEMORY_HOST, memory_location);
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
            col_coords[ncoeffs] = hypre_SStructUVEntryToRank(Uventry, entry);
            coeffs[ncoeffs] = h_values[i];
            ncoeffs++;
         }
      }
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
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

      hypre_TMemcpy(hypre_SStructMatrixTmpColCoordsDevice(matrix), col_coords, HYPRE_BigInt, ncoeffs,
                    memory_location, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_SStructMatrixTmpCoeffsDevice(matrix), coeffs, HYPRE_Complex, ncoeffs,
                    memory_location, HYPRE_MEMORY_HOST);

      if (action > 0)
      {
         HYPRE_IJMatrixAddToValues(ijmatrix, ncoeffs, NULL, hypre_SStructMatrixTmpRowCoordsDevice(matrix),
                                   (const HYPRE_BigInt *) hypre_SStructMatrixTmpColCoordsDevice(matrix),
                                   (const HYPRE_Complex *) hypre_SStructMatrixTmpCoeffsDevice(matrix));
      }
      else if (action > -1)
      {
         HYPRE_IJMatrixSetValues(ijmatrix, ncoeffs, NULL, hypre_SStructMatrixTmpRowCoordsDevice(matrix),
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Note: Entries must all be of type stencil or non-stencil, but not both.
 *
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
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
   HYPRE_Int             ndim     = hypre_SStructMatrixNDim(matrix);
   HYPRE_IJMatrix        ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph   *graph    = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid     = hypre_SStructGraphGrid(graph);
   hypre_SStructGrid    *dom_grid = hypre_SStructGraphDomainGrid(graph);
   hypre_SStructStencil *stencil  = hypre_SStructGraphStencil(graph, part, var);
   HYPRE_Int            *vars     = hypre_SStructStencilVars(stencil);
   hypre_Index          *shape    = hypre_SStructStencilShape(stencil);
   HYPRE_Int             size     = hypre_SStructStencilSize(stencil);
   hypre_IndexRef        offset;
   hypre_BoxManEntry   **boxman_entries;
   HYPRE_Int             nboxman_entries;
   hypre_BoxManEntry   **boxman_to_entries;
   HYPRE_Int             nboxman_to_entries;
   HYPRE_Int             nrows;
   HYPRE_Int            *ncols, *row_indexes;;
   HYPRE_BigInt         *rows, *cols;
   HYPRE_Complex        *ijvalues;
   hypre_Box            *box = hypre_BoxCreate(ndim);
   hypre_Box            *to_box;
   hypre_Box            *map_box;
   hypre_Box            *int_box;
   hypre_Index           index, stride, loop_size;
   hypre_IndexRef        start;
   hypre_Index           rs, cs;
   HYPRE_BigInt          row_base, col_base;
   HYPRE_Int             ei, entry, ii, jj;
   HYPRE_Int             matrix_type = hypre_SStructMatrixObjectType(matrix);
   HYPRE_MemoryLocation  memory_location = hypre_IJMatrixMemoryLocation(ijmatrix);

   /*------------------------------------------
    * all stencil entries
    *------------------------------------------*/

   if (entries[0] < size)
   {
      to_box  = hypre_BoxCreate(ndim);
      map_box = hypre_BoxCreate(ndim);
      int_box = hypre_BoxCreate(ndim);

      nrows       = hypre_BoxVolume(set_box);
      ncols       = hypre_CTAlloc(HYPRE_Int,     nrows,            memory_location);
      rows        = hypre_CTAlloc(HYPRE_BigInt,  nrows,            memory_location);
      row_indexes = hypre_CTAlloc(HYPRE_Int,     nrows,            memory_location);
      cols        = hypre_CTAlloc(HYPRE_BigInt,  nrows * nentries, memory_location);
      ijvalues    = hypre_CTAlloc(HYPRE_Complex, nrows * nentries, memory_location);

      hypre_SetIndex(stride, 1);

      hypre_SStructGridIntersect(grid, part, var, set_box, -1,
                                 &boxman_entries, &nboxman_entries);

      for (ii = 0; ii < nboxman_entries; ii++)
      {
         hypre_SStructBoxManEntryGetStrides(boxman_entries[ii], rs, matrix_type);

         hypre_CopyBox(set_box, box);
         hypre_BoxManEntryGetExtents(boxman_entries[ii],
                                     hypre_BoxIMin(map_box), hypre_BoxIMax(map_box));
         hypre_IntersectBoxes(box, map_box, int_box);
         hypre_CopyBox(int_box, box);

         /* For each index in 'box', compute a row of length <= nentries and
          * insert it into an nentries-length segment of 'cols' and 'ijvalues'.
          * This may result in gaps, but IJSetValues2() is designed for that. */

         nrows = hypre_BoxVolume(box);

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(ncols,row_indexes)
         hypre_LoopBegin(nrows, i)
         {
            ncols[i] = 0;
            row_indexes[i] = i * nentries;
         }
         hypre_LoopEnd()
#undef DEVICE_VAR
#define DEVICE_VAR

         for (ei = 0; ei < nentries; ei++)
         {
            entry = entries[ei];

            hypre_CopyBox(box, to_box);

            offset = shape[entry];
            hypre_BoxShiftPos(to_box, offset);

            hypre_SStructGridIntersect(dom_grid, part, vars[entry], to_box, -1,
                                       &boxman_to_entries, &nboxman_to_entries);

            for (jj = 0; jj < nboxman_to_entries; jj++)
            {
               hypre_SStructBoxManEntryGetStrides(boxman_to_entries[jj], cs, matrix_type);

               hypre_BoxManEntryGetExtents(boxman_to_entries[jj],
                                           hypre_BoxIMin(map_box), hypre_BoxIMax(map_box));
               hypre_IntersectBoxes(to_box, map_box, int_box);

               hypre_CopyIndex(hypre_BoxIMin(int_box), index);
               hypre_SStructBoxManEntryGetGlobalRank(boxman_to_entries[jj],
                                                     index, &col_base, matrix_type);

               hypre_BoxShiftNeg(int_box, offset);

               hypre_CopyIndex(hypre_BoxIMin(int_box), index);
               hypre_SStructBoxManEntryGetGlobalRank(boxman_entries[ii],
                                                     index, &row_base, matrix_type);

               start = hypre_BoxIMin(int_box);
               hypre_BoxGetSize(int_box, loop_size);

#if defined(HYPRE_USING_GPU)
               {
                  hypre_assert(ndim <= 3);

                  HYPRE_Int rs_0, rs_1, rs_2;
                  HYPRE_Int cs_0, cs_1, cs_2;

                  if (ndim > 0)
                  {
                     rs_0 = rs[0];
                     cs_0 = cs[0];
                  }

                  if (ndim > 1)
                  {
                     rs_1 = rs[1];
                     cs_1 = cs[1];
                  }

                  if (ndim > 2)
                  {
                     rs_2 = rs[2];
                     cs_2 = cs[2];
                  }

#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(ncols,rows,cols,ijvalues,values)
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      box,       start, stride, mi,
                                      value_box, start, stride, vi);
                  {
                     hypre_Index index;
                     HYPRE_Int   ci;

                     hypre_BoxLoopGetIndex(index);

                     ci = mi * nentries + ncols[mi];
                     rows[mi] = row_base;
                     cols[ci] = col_base;

                     if (ndim > 0)
                     {
                        rows[mi] += index[0] * rs_0;
                        cols[ci] += index[0] * cs_0;
                     }

                     if (ndim > 1)
                     {
                        rows[mi] += index[1] * rs_1;
                        cols[ci] += index[1] * cs_1;
                     }

                     if (ndim > 2)
                     {
                        rows[mi] += index[2] * rs_2;
                        cols[ci] += index[2] * cs_2;
                     }

                     ijvalues[ci] = values[ei + vi * nentries];
                     ncols[mi]++;
                  }
                  hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
               }
#else
               {
                  hypre_BoxLoop2Begin(ndim, loop_size,
                                      box,       start, stride, mi,
                                      value_box, start, stride, vi);
                  {
                     hypre_Index index;
                     HYPRE_Int   ci;

                     hypre_BoxLoopGetIndex(index);

                     ci = mi * nentries + ncols[mi];
                     rows[mi] = row_base;
                     cols[ci] = col_base;

                     HYPRE_Int d;
                     for (d = 0; d < ndim; d++)
                     {
                        rows[mi] += index[d] * rs[d];
                        cols[ci] += index[d] * cs[d];
                     }

                     ijvalues[ci] = values[ei + vi * nentries];
                     ncols[mi]++;
                  }
                  hypre_BoxLoop2End(mi, vi);
               }
#endif
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
            HYPRE_IJMatrixGetValues(ijmatrix, nrows, ncols, rows, cols, values);
         }

      } /* end loop through boxman entries */

      hypre_TFree(boxman_entries, HYPRE_MEMORY_HOST);

      hypre_TFree(ncols, memory_location);
      hypre_TFree(rows, memory_location);
      hypre_TFree(row_indexes, memory_location);
      hypre_TFree(cols, memory_location);
      hypre_TFree(ijvalues, memory_location);

      hypre_BoxDestroy(to_box);
      hypre_BoxDestroy(map_box);
      hypre_BoxDestroy(int_box);
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

   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructUMatrixAssemble( hypre_SStructMatrix *matrix )
{
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);

   HYPRE_IJMatrixAssemble(ijmatrix);

   return hypre_error_flag;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

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
   HYPRE_Int                ndim  = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph      *graph = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid       *grid  = hypre_SStructGraphGrid(graph);
   hypre_SStructPMatrix    *pmatrix;
   hypre_SStructPGrid      *pgrid;

   hypre_SStructStencil    *stencil;
   hypre_Index             *shape;
   HYPRE_Int               *smap;
   HYPRE_Int               *vars, frvartype, tovartype;
   hypre_StructMatrix      *smatrix;
   hypre_Box               *box, *ibox0, *ibox1, *tobox, *frbox;
   hypre_Index              stride, loop_size;
   hypre_IndexRef           offset, start;
   hypre_BoxManEntry      **frentries, **toentries;
   hypre_SStructBoxManInfo *frinfo, *toinfo;
   HYPRE_Complex           *tvalues = NULL;
   HYPRE_Int                tvalues_size = 0;
   HYPRE_Int                nfrentries, ntoentries, frpart, topart;
   HYPRE_Int                entry, sentry, ei, fri, toi;
   HYPRE_MemoryLocation     memory_location = hypre_IJMatrixMemoryLocation(hypre_SStructMatrixIJMatrix(
                                                                              matrix));

   pmatrix = hypre_SStructMatrixPMatrix(matrix, part);

   pgrid = hypre_SStructPMatrixPGrid(pmatrix);
   frvartype = hypre_SStructPGridVarType(pgrid, var);

   box   = hypre_BoxCreate(ndim);
   ibox0 = hypre_BoxCreate(ndim);
   ibox1 = hypre_BoxCreate(ndim);
   tobox = hypre_BoxCreate(ndim);
   frbox = hypre_BoxCreate(ndim);

   stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   shape   = hypre_SStructStencilShape(stencil);
   vars    = hypre_SStructStencilVars(stencil);

   hypre_SetIndex(stride, 1);

   for (ei = 0; ei < nentries; ei++)
   {
      entry  = entries[ei];
      sentry = smap[entry];
      offset = shape[entry];
      smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entry]);
      tovartype = hypre_SStructPGridVarType(pgrid, vars[entry]);

      /* shift box in the stencil offset direction */
      hypre_CopyBox(set_box, box);

      hypre_AddIndexes(hypre_BoxIMin(box), offset, ndim, hypre_BoxIMin(box));
      hypre_AddIndexes(hypre_BoxIMax(box), offset, ndim, hypre_BoxIMax(box));

      /* get "to" entries */
      hypre_SStructGridIntersect(grid, part, vars[entry], box, -1,
                                 &toentries, &ntoentries);

      for (toi = 0; toi < ntoentries; toi++)
      {
         hypre_BoxManEntryGetExtents(
            toentries[toi], hypre_BoxIMin(tobox), hypre_BoxIMax(tobox));
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

               hypre_BoxManEntryGetExtents(
                  frentries[fri], hypre_BoxIMin(frbox), hypre_BoxIMax(frbox));
               hypre_IntersectBoxes(ibox0, frbox, ibox1);
               if (hypre_BoxVolume(ibox1))
               {
                  HYPRE_Int tvalues_new_size = hypre_BoxVolume(ibox1);
                  tvalues = hypre_TReAlloc_v2(tvalues, HYPRE_Complex, tvalues_size, HYPRE_Complex, tvalues_new_size,
                                              memory_location);
                  tvalues_size = tvalues_new_size;

                  if (action >= 0)
                  {
                     /* set or add */

                     /* copy values into tvalues */
                     start = hypre_BoxIMin(ibox1);
                     hypre_BoxGetSize(ibox1, loop_size);
#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(tvalues,values)
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         ibox1, start, stride, mi,
                                         value_box, start, stride, vi);
                     {
                        tvalues[mi] = values[ei + vi * nentries];
                     }
                     hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
                     /* put values into UMatrix */
                     hypre_SStructUMatrixSetBoxValues(
                        matrix, part, ibox1, var, 1, &entry, ibox1, tvalues, action);
                     /* zero out values in PMatrix (possibly in ghost) */
                     hypre_StructMatrixClearBoxValues(
                        smatrix, ibox1, 1, &sentry, -1, 1);
                  }
                  else
                  {
                     /* get */

                     /* get values from UMatrix */
                     hypre_SStructUMatrixSetBoxValues(
                        matrix, part, ibox1, var, 1, &entry, ibox1, tvalues, action);

                     /* copy tvalues into values */
                     start = hypre_BoxIMin(ibox1);
                     hypre_BoxGetSize(ibox1, loop_size);
#undef DEVICE_VAR
#define DEVICE_VAR is_device_ptr(tvalues,values)
                     hypre_BoxLoop2Begin(ndim, loop_size,
                                         ibox1, start, stride, mi,
                                         value_box, start, stride, vi);
                     {
                        values[ei + vi * nentries] = tvalues[mi];
                     }
                     hypre_BoxLoop2End(mi, vi);
#undef DEVICE_VAR
#define DEVICE_VAR
                  } /* end if action */
               } /* end if nonzero ibox1 */
            } /* end of "from" boxman entries loop */
            hypre_TFree(frentries, HYPRE_MEMORY_HOST);
         } /* end if nonzero ibox0 */
      } /* end of "to" boxman entries loop */
      hypre_TFree(toentries, HYPRE_MEMORY_HOST);
   } /* end of entries loop */

   hypre_BoxDestroy(box);
   hypre_BoxDestroy(ibox0);
   hypre_BoxDestroy(ibox1);
   hypre_BoxDestroy(tobox);
   hypre_BoxDestroy(frbox);
   hypre_TFree(tvalues, memory_location);

   return hypre_error_flag;
}

HYPRE_MemoryLocation
hypre_SStructMatrixMemoryLocation(hypre_SStructMatrix *matrix)
{
   HYPRE_Int type = hypre_SStructMatrixObjectType(matrix);

   if (type == HYPRE_SSTRUCT)
   {
      return hypre_ParCSRMatrixMemoryLocation(hypre_SStructMatrixParCSRMatrix(matrix));
   }

   void *object;
   HYPRE_SStructMatrixGetObject(matrix, &object);

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
