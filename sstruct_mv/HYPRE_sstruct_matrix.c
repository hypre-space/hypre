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
 * HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixCreate( MPI_Comm              comm,
                           HYPRE_SStructGraph    graph,
                           HYPRE_SStructMatrix  *matrix_ptr )
{
   int ierr = 0;

   hypre_SStructGrid      *grid     = hypre_SStructGraphGrid(graph);
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_SStructMatrix    *matrix;
   int                  ***splits;
   int                     nparts;
   hypre_SStructPMatrix  **pmatrices;

   hypre_SStructPGrid     *pgrid;
   int                     nvars;

   int                     stencil_size;
   int                    *stencil_vars;
   int                     pstencil_size;

   HYPRE_SStructVariable   vitype, vjtype;
   int                     part, vi, vj, i;
   int                     size;

   matrix = hypre_TAlloc(hypre_SStructMatrix, 1);

   hypre_SStructMatrixComm(matrix)  = comm;
   hypre_SStructMatrixNDim(matrix)  = hypre_SStructGraphNDim(graph);
   hypre_SStructGraphRef(graph, &hypre_SStructMatrixGraph(matrix));

   /* compute S/U-matrix split */
   nparts = hypre_SStructGraphNParts(graph);
   hypre_SStructMatrixNParts(matrix) = nparts;
   splits = hypre_TAlloc(int **, nparts);
   hypre_SStructMatrixSplits(matrix) = splits;
   pmatrices = hypre_TAlloc(hypre_SStructPMatrix *, nparts);
   hypre_SStructMatrixPMatrices(matrix) = pmatrices;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      splits[part] = hypre_TAlloc(int *, nvars);
      for (vi = 0; vi < nvars; vi++)
      {
         stencil_size  = hypre_SStructStencilSize(stencils[part][vi]);
         stencil_vars  = hypre_SStructStencilVars(stencils[part][vi]);
         pstencil_size = 0;
         splits[part][vi] = hypre_TAlloc(int, stencil_size);
         for (i = 0; i < stencil_size; i++)
         {
            vj = stencil_vars[i];
            vitype = hypre_SStructPGridVarType(pgrid, vi);
            vjtype = hypre_SStructPGridVarType(pgrid, vj);
            if (vjtype == vitype)
            {
               splits[part][vi][i] = pstencil_size;
               pstencil_size++;
            }
            else
            {
               splits[part][vi][i] = -1;
            }
         }
      }
   }

   HYPRE_IJMatrixCreate(comm, &hypre_SStructMatrixIJMatrix(matrix),
                        hypre_SStructGridGlobalSize(grid),
                        hypre_SStructGridGlobalSize(grid));
   hypre_SStructMatrixParCSRMatrix(matrix) = NULL;

   size = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (vi = 0; vi < nvars; vi++)
      {
         size = hypre_max(size, hypre_SStructStencilSize(stencils[part][vi]));
      }
   }
   hypre_SStructMatrixSEntries(matrix) = hypre_TAlloc(int, size);
   hypre_SStructMatrixUEntries(matrix) = hypre_TAlloc(int, size);
   hypre_SStructMatrixTmpColCoords(matrix) = NULL;
   hypre_SStructMatrixTmpCoeffs(matrix)    = NULL;

   hypre_SStructMatrixSymmetric(matrix)  = 0;
   hypre_SStructMatrixGlobalSize(matrix) = 0;
   hypre_SStructMatrixRefCount(matrix)   = 1;

   *matrix_ptr = matrix;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixDestroy( HYPRE_SStructMatrix matrix )
{
   int ierr = 0;

   hypre_SStructGraph     *graph;
   int                  ***splits;
   int                     nparts;
   hypre_SStructPMatrix  **pmatrices;
   hypre_SStructPGrid     *pgrid;
   int                     nvars;
   int                     part, var;

   if (matrix)
   {
      hypre_SStructMatrixRefCount(matrix) --;
      if (hypre_SStructMatrixRefCount(matrix) == 0)
      {
         graph        = hypre_SStructMatrixGraph(matrix);
         splits       = hypre_SStructMatrixSplits(matrix);
         nparts       = hypre_SStructMatrixNParts(matrix);
         pmatrices    = hypre_SStructMatrixPMatrices(matrix);
         for (part = 0; part < nparts; part++)
         {
            pgrid = hypre_SStructGraphPGrid(graph, part);
            nvars = hypre_SStructPGridNVars(pgrid);
            for (var = 0; var < nvars; var++)
            {
               hypre_TFree(splits[part][var]);
            }
            hypre_TFree(splits[part]);
            hypre_SStructPMatrixDestroy(pmatrices[part]);
         }
         HYPRE_SStructGraphDestroy(graph);
         hypre_TFree(splits);
         hypre_TFree(pmatrices);
         HYPRE_IJMatrixDestroy(hypre_SStructMatrixIJMatrix(matrix));
         hypre_TFree(hypre_SStructMatrixSEntries(matrix));
         hypre_TFree(hypre_SStructMatrixUEntries(matrix));
         hypre_TFree(hypre_SStructMatrixTmpColCoords(matrix));
         hypre_TFree(hypre_SStructMatrixTmpCoeffs(matrix));
         hypre_TFree(matrix);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixInitialize( HYPRE_SStructMatrix matrix )
{
   int ierr = 0;

   int                     nparts    = hypre_SStructMatrixNParts(matrix);
   hypre_SStructGraph     *graph     = hypre_SStructMatrixGraph(matrix);
   hypre_SStructPMatrix  **pmatrices = hypre_SStructMatrixPMatrices(matrix);
   hypre_SStructStencil ***stencils  = hypre_SStructGraphStencils(graph);
   int                    *split;

   MPI_Comm                pcomm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **pstencils;
   int                     nvars;

   int                     stencil_size;
   hypre_Index            *stencil_shape;
   int                    *stencil_vars;
   int                     pstencil_ndim;
   int                     pstencil_size;

   int                     part, var, i;

   /* S-matrix */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      pstencils = hypre_TAlloc(hypre_SStructStencil *, nvars);
      for (var = 0; var < nvars; var++)
      {
         split = hypre_SStructMatrixSplit(matrix, part, var);
         stencil_size  = hypre_SStructStencilSize(stencils[part][var]);
         stencil_shape = hypre_SStructStencilShape(stencils[part][var]);
         stencil_vars  = hypre_SStructStencilVars(stencils[part][var]);
         pstencil_ndim = hypre_SStructStencilNDim(stencils[part][var]);
         pstencil_size = 0;
         for (i = 0; i < stencil_size; i++)
         {
            if (split[i] > -1)
            {
               pstencil_size++;
            }
         }
         HYPRE_SStructStencilCreate(pstencil_ndim, pstencil_size,
                                    &pstencils[var]);
         for (i = 0; i < stencil_size; i++)
         {
            if (split[i] > -1)
            {
               HYPRE_SStructStencilSetEntry(pstencils[var], split[i],
                                            stencil_shape[i],
                                            stencil_vars[i]);
            }
         }
      }
      pcomm = hypre_SStructPGridComm(pgrid);
      hypre_SStructPMatrixCreate(pcomm, pgrid, pstencils, &pmatrices[part]);
      hypre_SStructPMatrixInitialize(pmatrices[part]);
   }

   /* U-matrix */
   hypre_SStructUMatrixInitialize(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixSetValues( HYPRE_SStructMatrix  matrix,
                              int                  part,
                              int                 *index,
                              int                  var,
                              int                  nentries,
                              int                 *entries,
                              double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
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
                                    nSentries, Sentries, values, 0);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, 0);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixSetBoxValues( HYPRE_SStructMatrix  matrix,
                                 int                  part,
                                 int                 *ilower,
                                 int                 *iupper,
                                 int                  var,
                                 int                  nentries,
                                 int                 *entries,
                                 double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetBoxValues(pmatrix, cilower, ciupper, var,
                                       nSentries, Sentries, values, 0);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetBoxValues(matrix, part, cilower, ciupper, var,
                                       nUentries, Uentries, values, 0);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixAddToValues( HYPRE_SStructMatrix  matrix,
                                int                  part,
                                int                 *index,
                                int                  var,
                                int                  nentries,
                                int                 *entries,
                                double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
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
                                    nSentries, Sentries, values, 1);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetValues(matrix, part, cindex, var,
                                    nUentries, Uentries, values, 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixAddToBoxValues( HYPRE_SStructMatrix  matrix,
                                   int                  part,
                                   int                 *ilower,
                                   int                 *iupper,
                                   int                  var,
                                   int                  nentries,
                                   int                 *entries,
                                   double              *values )
{
   int ierr = 0;
   int                   ndim  = hypre_SStructMatrixNDim(matrix);
   int                  *Sentries;
   int                  *Uentries;
   int                   nSentries;
   int                   nUentries;
   hypre_SStructPMatrix *pmatrix;
   hypre_Index           cilower;
   hypre_Index           ciupper;

   hypre_SStructMatrixSplitEntries(matrix, part, var, nentries, entries,
                                   &nSentries, &Sentries,
                                   &nUentries, &Uentries);

   hypre_CopyToCleanIndex(ilower, ndim, cilower);
   hypre_CopyToCleanIndex(iupper, ndim, ciupper);

   /* S-matrix */
   if (nSentries > 0)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      hypre_SStructPMatrixSetBoxValues(pmatrix, cilower, ciupper, var,
                                       nSentries, Sentries, values, 1);
   }
   /* U-matrix */
   if (nUentries > 0)
   {
      hypre_SStructUMatrixSetBoxValues(matrix, part, cilower, ciupper, var,
                                       nUentries, Uentries, values, 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructMatrixAssemble( HYPRE_SStructMatrix matrix )
{
   int ierr = 0;
   int nparts = hypre_SStructMatrixNParts(matrix);
   int part;

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatrixAssemble(hypre_SStructMatrixPMatrix(matrix, part));
   }

   /* U-matrix */
   hypre_SStructUMatrixAssemble(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * TODO
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SStructMatrixSetSymmetric( HYPRE_SStructMatrix  matrix,
                                 int                  symmetric )
{
   int ierr = 0;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixSetObjectType( HYPRE_SStructMatrix  matrix,
                                  int                  type )
{
   int ierr = 0;

   hypre_SStructGraph     *graph    = hypre_SStructMatrixGraph(matrix);
   int                  ***splits   = hypre_SStructMatrixSplits(matrix);
   int                     nparts   = hypre_SStructMatrixNParts(matrix);
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_SStructPGrid     *pgrid;
   int                     nvars;
   int                     stencil_size;
   int                     part, var, i;

   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (var = 0; var < nvars; var++)
      {
         stencil_size = hypre_SStructStencilSize(stencils[part][var]);
         for (i = 0; i < stencil_size; i++)
         {
            splits[part][var][i] = -1;
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixGetObject( HYPRE_SStructMatrix   matrix,
                              void                **object )
{
   int ierr = 0;
   HYPRE_IJMatrix ijmatrix = hypre_SStructMatrixIJMatrix(matrix);

   *object = HYPRE_IJMatrixGetLocalStorage(ijmatrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructMatrixPrint( char                *filename,
                          HYPRE_SStructMatrix  matrix,
                          int                  all )
{
   int ierr = 0;
   int  nparts = hypre_SStructMatrixNParts(matrix);
   int  part;
   char new_filename[255];

   for (part = 0; part < nparts; part++)
   {
      sprintf(new_filename, "%s.%02d", filename, part);
      hypre_SStructPMatrixPrint(new_filename,
                                hypre_SStructMatrixPMatrix(matrix, part),
                                all);
   }

   /* U-matrix */
   sprintf(new_filename, "%s.UMatrix", filename);
   hypre_ParCSRMatrixPrintIJ(hypre_SStructMatrixParCSRMatrix(matrix),
                             new_filename);

   return ierr;
}

