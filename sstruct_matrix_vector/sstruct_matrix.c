/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_SStructPMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

/*==========================================================================
 * SStructPMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixCreate
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatrixCreate( MPI_Comm               comm,
                            hypre_SStructPGrid    *pgrid,
                            hypre_SStructStencil **stencils,
                            hypre_SStructPMatrix **pmatrix_ptr )
{
   int ierr = 0;

   hypre_SStructPMatrix  *pmatrix;
   int                    nvars;
   int                  **smaps;
   hypre_StructStencil ***sstencils;
   hypre_StructMatrix  ***smatrices;
   int                   *sentries;
   int                    num_ghost[6] = {1, 1, 1, 1, 1, 1};

   hypre_StructStencil   *sstencil;
   int                   *vars;
   hypre_Index           *sstencil_shape;
   int                    sstencil_size;
   int                    new_dim;
   int                   *new_sizes;
   hypre_Index          **new_shapes;
   int                    size;
   hypre_StructGrid      *sgrid;

   int                    vi, vj;
   int                    i, j, k;

   pmatrix = hypre_TAlloc(hypre_SStructPMatrix, 1);

   hypre_SStructPMatrixComm(pmatrix)     = comm;
   hypre_SStructPMatrixPGrid(pmatrix)    = pgrid;
   hypre_SStructPMatrixStencils(pmatrix) = stencils;
   nvars = hypre_SStructPGridNVars(pgrid);
   hypre_SStructPMatrixNVars(pmatrix) = nvars;

   /* create sstencils */
   smaps     = hypre_TAlloc(int *, nvars);
   sstencils = hypre_TAlloc(hypre_StructStencil **, nvars);
   new_sizes  = hypre_TAlloc(int, nvars);
   new_shapes = hypre_TAlloc(hypre_Index *, nvars);
   size = 0;
   for (vi = 0; vi < nvars; vi++)
   {
      sstencils[vi] = hypre_TAlloc(hypre_StructStencil *, nvars);
      for (vj = 0; vj < nvars; vj++)
      {
         sstencils[vi][vj] = NULL;
         new_sizes[vj] = 0;
      }

      sstencil       = hypre_SStructStencilSStencil(stencils[vi]);
      vars           = hypre_SStructStencilVars(stencils[vi]);
      sstencil_shape = hypre_StructStencilShape(sstencil);
      sstencil_size  = hypre_StructStencilSize(sstencil);

      smaps[vi] = hypre_TAlloc(int, sstencil_size);
      for (i = 0; i < sstencil_size; i++)
      {
         j = vars[i];
         new_sizes[j]++;
      }
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            new_shapes[vj] = hypre_TAlloc(hypre_Index, new_sizes[vj]);
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
      new_dim = hypre_StructStencilDim(sstencil);
      for (vj = 0; vj < nvars; vj++)
      {
         if (new_sizes[vj])
         {
            sstencils[vi][vj] = hypre_StructStencilCreate(new_dim,
                                                          new_sizes[vj],
                                                          new_shapes[vj]);
         }
         size = hypre_max(size, new_sizes[vj]);
      }
   }
   hypre_SStructPMatrixSMaps(pmatrix)     = smaps;
   hypre_SStructPMatrixSStencils(pmatrix) = sstencils;
   hypre_TFree(new_sizes);
   hypre_TFree(new_shapes);

   /* create smatrices */
   smatrices = hypre_TAlloc(hypre_StructMatrix **, nvars);
   for (vi = 0; vi < nvars; vi++)
   {
      smatrices[vi] = hypre_TAlloc(hypre_StructMatrix *, nvars);
      for (vj = 0; vj < nvars; vj++)
      {
         smatrices[vi][vj] = NULL;
         if (sstencils[vi][vj] != NULL)
         {
            sgrid = hypre_SStructPGridSGrid(pgrid, vi);
            smatrices[vi][vj] =
               hypre_StructMatrixCreate(comm, sgrid, sstencils[vi][vj]);
            /* TODO hypre_StructMatrixSetSymmetric(smatrices[vi][vj], 1); */
            hypre_StructMatrixSetNumGhost(smatrices[vi][vj], num_ghost);
         }
      }
   }
   hypre_SStructPMatrixSMatrices(pmatrix) = smatrices;

   hypre_SStructPMatrixSEntries(pmatrix) = hypre_TAlloc(int, size);

   *pmatrix_ptr = pmatrix;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixDestroy( hypre_SStructPMatrix *pmatrix )
{
   int ierr = 0;

   hypre_SStructStencil  **stencils;
   int                     nvars;
   int                   **smaps;
   hypre_StructStencil  ***sstencils;
   hypre_StructMatrix   ***smatrices;
   int                    *sentries;
   int                     vi, vj;

   if (pmatrix)
   {
      stencils  = hypre_SStructPMatrixStencils(pmatrix);
      nvars     = hypre_SStructPMatrixNVars(pmatrix);
      smaps     = hypre_SStructPMatrixSMaps(pmatrix);
      sstencils = hypre_SStructPMatrixSStencils(pmatrix);
      smatrices = hypre_SStructPMatrixSMatrices(pmatrix);
      for (vi = 0; vi < nvars; vi++)
      {
         HYPRE_SStructStencilDestroy(stencils[vi]);
         hypre_TFree(smaps[vi]);
         for (vj = 0; vj < nvars; vj++)
         {
            hypre_StructStencilDestroy(sstencils[vi][vj]);
            hypre_StructMatrixDestroy(smatrices[vi][vj]);
         }
         hypre_TFree(sstencils[vi]);
         hypre_TFree(smatrices[vi]);
      }
      hypre_TFree(stencils);
      hypre_TFree(smaps);
      hypre_TFree(sstencils);
      hypre_TFree(smatrices);
      hypre_TFree(hypre_SStructPMatrixSEntries(pmatrix));
      hypre_TFree(pmatrix);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixInitialize
 *--------------------------------------------------------------------------*/
int 
hypre_SStructPMatrixInitialize( hypre_SStructPMatrix *pmatrix )
{
   int ierr = 0;
   int                 nvars = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix *smatrix;
   int                 vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            hypre_StructMatrixInitialize(smatrix);
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixSetValues( hypre_SStructPMatrix *pmatrix,
                               hypre_Index           index,
                               int                   var,
                               int                   nentries,
                               int                  *entries,
                               double               *values,
                               int                   add_to )
{
   int ierr = 0;
   hypre_SStructStencil *stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   int                  *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_StructMatrix   *smatrix;
   int                  *sentries;
   int                   d, i;

   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   sentries = hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   ierr = hypre_StructMatrixSetValues(smatrix, index, nentries, sentries,
                                      values, add_to);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixSetBoxValues( hypre_SStructPMatrix *pmatrix,
                                  hypre_Index           ilower,
                                  hypre_Index           iupper,
                                  int                   var,
                                  int                   nentries,
                                  int                  *entries,
                                  double               *values,
                                  int                   add_to )
{
   int ierr = 0;

   hypre_SStructStencil *stencil = hypre_SStructPMatrixStencil(pmatrix, var);
   int                  *smap    = hypre_SStructPMatrixSMap(pmatrix, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_StructMatrix   *smatrix;
   hypre_Box            *box;
   int                  *sentries;
   int                   d, i;

   smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, vars[entries[0]]);

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));

   sentries = hypre_SStructPMatrixSEntries(pmatrix);
   for (i = 0; i < nentries; i++)
   {
      sentries[i] = smap[entries[i]];
   }

   ierr = hypre_StructMatrixSetBoxValues(smatrix, box, nentries, sentries,
                                         values, add_to);

   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_SStructPMatrixAssemble( hypre_SStructPMatrix *pmatrix )
{
   int ierr = 0;
   int                 nvars = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix *smatrix;
   int                 vi, vj;

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            hypre_StructMatrixAssemble(smatrix);
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructPMatrixPrint
 *--------------------------------------------------------------------------*/

int
hypre_SStructPMatrixPrint( char                 *filename,
                           hypre_SStructPMatrix *pmatrix,
                           int                   all )
{
   int ierr = 0;
   int                 nvars = hypre_SStructPMatrixNVars(pmatrix);
   hypre_StructMatrix *smatrix;
   int                 vi, vj;
   char                new_filename[255];

   for (vi = 0; vi < nvars; vi++)
   {
      for (vj = 0; vj < nvars; vj++)
      {
         smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
         if (smatrix != NULL)
         {
            sprintf(new_filename, "%s.%02d.%02d", filename, vi, vj);
            hypre_StructMatrixPrint(new_filename, smatrix, all);
         }
      }
   }

   return ierr;
}

/*==========================================================================
 * SStructUMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixInitialize( hypre_SStructMatrix *matrix )
{
   int ierr = 0;

   HYPRE_IJMatrix          ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph     *graph    = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid      *grid     = hypre_SStructGraphGrid(graph);
   int                     nparts   = hypre_SStructGraphNParts(graph);
   hypre_SStructPGrid    **pgrids   = hypre_SStructGraphPGrids(graph);
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_StructGrid       *sgrid;
   hypre_StructStencil    *sstencil;
   int                     nvars;
   int                     nrows, ncols, nnzs;
   int                     part, var, i, j;
   int                    *row_sizes;

   ierr = HYPRE_IJMatrixSetLocalStorageType(ijmatrix, HYPRE_PARCSR);

   nrows = hypre_SStructGridLocalSize(grid); 
   ncols = hypre_SStructGridLocalSize(grid); 
   ierr += HYPRE_IJMatrixSetLocalSize(ijmatrix, nrows, ncols);

   /* set row sizes */
   i = 0;
   row_sizes = hypre_CTAlloc(int, nrows);
   for (part = 0; part < nparts; part++)
   {
      nvars = hypre_SStructPGridNVars(pgrids[part]);
      for (var = 0; var < nvars; var++)
      {
         sgrid    = hypre_SStructPGridSGrid(pgrids[part], var);
         nrows    = hypre_StructGridLocalSize(sgrid);
         sstencil = hypre_SStructStencilSStencil(stencils[part][var]);
         nnzs     = hypre_StructStencilSize(sstencil);
         if (hypre_SStructMatrixSymmetric(matrix))
         {
            nnzs = 2*nnzs - 1;
         }
         for (j = 0; j < nrows; j++)
         {
            row_sizes[i++] = nnzs;
         }
      }
   }
   ierr += HYPRE_IJMatrixSetRowSizes (ijmatrix, (const int *) row_sizes);
   hypre_TFree(row_sizes);

   ierr += HYPRE_IJMatrixInitialize(ijmatrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixSetValues( hypre_SStructMatrix *matrix,
                               int                  part,
                               hypre_Index          index,
                               int                  var,
                               int                  nentries,
                               int                 *entries,
                               double              *values,
                               int                  add_to )
{
   int ierr = 0;

   HYPRE_IJMatrix        ijmatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_SStructGraph   *graph   = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid    = hypre_SStructGraphGrid(graph);
   hypre_SStructStencil *stencil = hypre_SStructGraphStencil(graph, part, var);
   int                  *vars    = hypre_SStructStencilVars(stencil);
   hypre_Index          *shape   = hypre_SStructStencilShape(stencil);
   int                   size    = hypre_SStructStencilSize(stencil);
   hypre_IndexRef        offset;
   hypre_Index           to_index;
   hypre_SStructUVEntry *Uventry;
   int                   row_coord;
   int                  *col_coords;
   int                   ncoeffs;
   double               *coeffs;
   int                   i, box, entry;

   hypre_SStructGridIndexToBox(grid, part, index, var, &box);
   if (box == -1)
   {
      printf("Warning: Attempt to set coeffs for point not in grid\n");
      printf("hypre_SStructUMatrixSetValues call aborted for grid point\n");
      printf("    part=%d, var=%d, index=(%d, %d, %d)\n", part, var,
             hypre_IndexD(index,0),
             hypre_IndexD(index,1),
             hypre_IndexD(index,2) );
      return(0);
   }
   hypre_SStructGridSVarIndexToRank(grid, box, part, index, var, &row_coord);

   col_coords = hypre_CTAlloc(int,    nentries);
   coeffs     = hypre_CTAlloc(double, nentries);
   ncoeffs = 0;
   for (i = 0; i < nentries; i++)
   {
      entry = entries[i];

      if (entry < size)
      {
         /* stencil entries */
         offset = shape[entry];
         hypre_IndexX(to_index) = hypre_IndexX(index) + hypre_IndexX(offset);
         hypre_IndexY(to_index) = hypre_IndexY(index) + hypre_IndexY(offset);
         hypre_IndexZ(to_index) = hypre_IndexZ(index) + hypre_IndexZ(offset);
         
         hypre_SStructGridIndexToBox(grid, part, to_index, vars[entry], &box);
         
         if (box > -1)
         {
            hypre_SStructGridSVarIndexToRank(grid, box,
                                             part, to_index, vars[entry],
                                             &col_coords[ncoeffs]);
            coeffs[ncoeffs] = values[i];
            ncoeffs++;
         }
      }
      else
      {
         /* non-stencil entries */
         entry -= size;
         hypre_SStructGraphFindUVEntry(graph, part, index, var, &Uventry);
         col_coords[ncoeffs] = hypre_SStructUVEntryRank(Uventry, entry);
         coeffs[ncoeffs] = values[i];
         ncoeffs++;
      }
   }

   if (add_to)
   {
      ierr = HYPRE_IJMatrixAddToValues(ijmatrix, ncoeffs, row_coord,
                                       (const int *) col_coords,
                                       (const double *) values);
   }
   else
   {
      ierr = HYPRE_IJMatrixSetValues(ijmatrix, ncoeffs, row_coord,
                                     (const int *) col_coords,
                                     (const double *) values);
   }

   hypre_TFree(col_coords);
   hypre_TFree(coeffs);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixSetBoxValues TODO (optimize)
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixSetBoxValues( hypre_SStructMatrix *matrix,
                                  int                  part,
                                  hypre_Index          ilower,
                                  hypre_Index          iupper,
                                  int                  var,
                                  int                  nentries,
                                  int                 *entries,
                                  double              *values,
                                  int                  add_to )
{
   int ierr = 0;

   hypre_SStructGraph   *graph = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid = hypre_SStructGraphGrid(graph);
   int                   ndim = hypre_SStructGridNDim(grid);
   hypre_Box            *box;
   hypre_Index           index;
   int                   d, i, j, k;

   hypre_StructStencil *stencil;

   box = hypre_BoxCreate();
   hypre_CopyIndex(ilower, hypre_BoxIMin(box));
   hypre_CopyIndex(iupper, hypre_BoxIMax(box));

   for (k = hypre_BoxIMinZ(box); k <= hypre_BoxIMaxZ(box); k++)
   {
      for (j = hypre_BoxIMinY(box); j <= hypre_BoxIMaxY(box); j++)
      {
         for (i = hypre_BoxIMinX(box); i <= hypre_BoxIMaxX(box); i++)
         {
            hypre_SetIndex(index, i, j, k);
            ierr += hypre_SStructUMatrixSetValues(matrix, part, index, var,
                                                  nentries, entries, values,
                                                  add_to);
            values += nentries;
         }
      }
   }

   hypre_BoxDestroy(box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SStructUMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_SStructUMatrixAssemble( hypre_SStructMatrix *matrix )
{
   int ierr = 0;
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);

   ierr = HYPRE_IJMatrixAssemble(ijmatrix);
   hypre_SStructMatrixParCSRMatrix(matrix) =
      (hypre_ParCSRMatrix *) HYPRE_IJMatrixGetLocalStorage(ijmatrix);

   return ierr;
}

/*==========================================================================
 * SStructMatrix routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixRef
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatrixRef( hypre_SStructMatrix  *matrix,
                        hypre_SStructMatrix **matrix_ref )
{
   hypre_SStructMatrixRefCount(matrix) ++;
   *matrix_ref = matrix;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_SStructMatrixSplitEntries
 *--------------------------------------------------------------------------*/

int
hypre_SStructMatrixSplitEntries( hypre_SStructMatrix *matrix,
                                 int                  part,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 int                 *nSentries_ptr,
                                 int                **Sentries_ptr,
                                 int                 *nUentries_ptr,
                                 int                **Uentries_ptr )
{
   int ierr = 0;
   hypre_SStructGraph   *graph   = hypre_SStructMatrixGraph(matrix);
   int                  *split   = hypre_SStructMatrixSplit(matrix, part, var);
   hypre_SStructStencil *stencil = hypre_SStructGraphStencil(graph, part, var);
   int                   entry;
   int                   i;

   int                   nSentries = 0;
   int                  *Sentries  = hypre_SStructMatrixSEntries(matrix);
   int                   nUentries = 0;
   int                  *Uentries  = hypre_SStructMatrixUEntries(matrix);

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

   return ierr;
}

