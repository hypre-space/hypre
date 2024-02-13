/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixCreate( MPI_Comm              comm,
                           HYPRE_SStructGraph    graph,
                           HYPRE_SStructMatrix  *matrix_ptr )
{
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_SStructMatrix    *matrix;
   HYPRE_Int            ***splits;
   HYPRE_Int               nparts;
   hypre_SStructPMatrix  **pmatrices;
   HYPRE_Int            ***symmetric;

   hypre_SStructPGrid     *pgrid;
   HYPRE_Int               nvars;

   HYPRE_Int               stencil_size;
   HYPRE_Int              *stencil_vars;
   HYPRE_Int               pstencil_size;

   HYPRE_SStructVariable   vitype, vjtype;
   HYPRE_Int               part, vi, vj, i;
   HYPRE_Int               size, rectangular;

   matrix = hypre_TAlloc(hypre_SStructMatrix,  1, HYPRE_MEMORY_HOST);

   hypre_SStructMatrixComm(matrix)  = comm;
   hypre_SStructMatrixNDim(matrix)  = hypre_SStructGraphNDim(graph);
   hypre_SStructGraphRef(graph, &hypre_SStructMatrixGraph(matrix));

   /* compute S/U-matrix split */
   nparts = hypre_SStructGraphNParts(graph);
   hypre_SStructMatrixNParts(matrix) = nparts;
   splits = hypre_TAlloc(HYPRE_Int **,  nparts, HYPRE_MEMORY_HOST);
   hypre_SStructMatrixSplits(matrix) = splits;
   pmatrices = hypre_TAlloc(hypre_SStructPMatrix *,  nparts, HYPRE_MEMORY_HOST);
   hypre_SStructMatrixPMatrices(matrix) = pmatrices;
   symmetric = hypre_TAlloc(HYPRE_Int **,  nparts, HYPRE_MEMORY_HOST);
   hypre_SStructMatrixSymmetric(matrix) = symmetric;
   /* is this a rectangular matrix? */
   rectangular = 0;
   if (hypre_SStructGraphGrid(graph) != hypre_SStructGraphDomainGrid(graph))
   {
      rectangular = 1;
   }
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      splits[part] = hypre_TAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);
      symmetric[part] = hypre_TAlloc(HYPRE_Int *,  nvars, HYPRE_MEMORY_HOST);
      for (vi = 0; vi < nvars; vi++)
      {
         stencil_size  = hypre_SStructStencilSize(stencils[part][vi]);
         stencil_vars  = hypre_SStructStencilVars(stencils[part][vi]);
         pstencil_size = 0;
         splits[part][vi] = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
         symmetric[part][vi] = hypre_TAlloc(HYPRE_Int,  nvars, HYPRE_MEMORY_HOST);
         for (i = 0; i < stencil_size; i++)
         {
            /* for rectangular matrices, put all coefficients in U-matrix */
            if (rectangular)
            {
               splits[part][vi][i] = -1;
            }
            else
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
         for (vj = 0; vj < nvars; vj++)
         {
            symmetric[part][vi][vj] = 0;
         }
      }
   }

   /* GEC0902 move the IJ creation to the initialization phase
    * ilower = hypre_SStructGridGhstartRank(grid);
    * iupper = ilower + hypre_SStructGridGhlocalSize(grid) - 1;
    * HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper,
    *                    &hypre_SStructMatrixIJMatrix(matrix)); */

   hypre_SStructMatrixIJMatrix(matrix)     = NULL;
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
   hypre_SStructMatrixSEntries(matrix) = hypre_TAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   size += hypre_SStructGraphUEMaxSize(graph);
   hypre_SStructMatrixUEntries(matrix) = hypre_TAlloc(HYPRE_Int,  size, HYPRE_MEMORY_HOST);
   hypre_SStructMatrixEntriesSize(matrix) = size;
   hypre_SStructMatrixTmpRowCoords(matrix) = NULL;
   hypre_SStructMatrixTmpColCoords(matrix) = NULL;
   hypre_SStructMatrixTmpCoeffs(matrix)    = NULL;
   hypre_SStructMatrixTmpRowCoordsDevice(matrix) = NULL;
   hypre_SStructMatrixTmpColCoordsDevice(matrix) = NULL;
   hypre_SStructMatrixTmpCoeffsDevice(matrix)    = NULL;

   hypre_SStructMatrixNSSymmetric(matrix) = 0;
   hypre_SStructMatrixGlobalSize(matrix)  = 0;
   hypre_SStructMatrixRefCount(matrix)    = 1;

   /* GEC0902 setting the default of the object_type to HYPRE_SSTRUCT */

   hypre_SStructMatrixObjectType(matrix) = HYPRE_SSTRUCT;

   *matrix_ptr = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixDestroy( HYPRE_SStructMatrix matrix )
{
   hypre_SStructGraph     *graph;
   HYPRE_Int            ***splits;
   HYPRE_Int               nparts;
   hypre_SStructPMatrix  **pmatrices;
   HYPRE_Int            ***symmetric;
   hypre_SStructPGrid     *pgrid;
   HYPRE_Int               nvars;
   HYPRE_Int               part, var;
   HYPRE_MemoryLocation    memory_location;

   if (matrix)
   {
      memory_location = hypre_SStructMatrixMemoryLocation(matrix);

      hypre_SStructMatrixRefCount(matrix) --;
      if (hypre_SStructMatrixRefCount(matrix) == 0)
      {
         graph        = hypre_SStructMatrixGraph(matrix);
         splits       = hypre_SStructMatrixSplits(matrix);
         nparts       = hypre_SStructMatrixNParts(matrix);
         pmatrices    = hypre_SStructMatrixPMatrices(matrix);
         symmetric    = hypre_SStructMatrixSymmetric(matrix);
         for (part = 0; part < nparts; part++)
         {
            pgrid = hypre_SStructGraphPGrid(graph, part);
            nvars = hypre_SStructPGridNVars(pgrid);
            for (var = 0; var < nvars; var++)
            {
               hypre_TFree(splits[part][var], HYPRE_MEMORY_HOST);
               hypre_TFree(symmetric[part][var], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(splits[part], HYPRE_MEMORY_HOST);
            hypre_TFree(symmetric[part], HYPRE_MEMORY_HOST);
            hypre_SStructPMatrixDestroy(pmatrices[part]);
         }
         HYPRE_SStructGraphDestroy(graph);
         hypre_TFree(splits, HYPRE_MEMORY_HOST);
         hypre_TFree(pmatrices, HYPRE_MEMORY_HOST);
         hypre_TFree(symmetric, HYPRE_MEMORY_HOST);
         HYPRE_IJMatrixDestroy(hypre_SStructMatrixIJMatrix(matrix));
         hypre_TFree(hypre_SStructMatrixSEntries(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_SStructMatrixUEntries(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_SStructMatrixTmpRowCoords(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_SStructMatrixTmpColCoords(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_SStructMatrixTmpCoeffs(matrix), HYPRE_MEMORY_HOST);
         hypre_TFree(hypre_SStructMatrixTmpRowCoordsDevice(matrix), memory_location);
         hypre_TFree(hypre_SStructMatrixTmpColCoordsDevice(matrix), memory_location);
         hypre_TFree(hypre_SStructMatrixTmpCoeffsDevice(matrix), memory_location);
         hypre_TFree(matrix, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixInitialize( HYPRE_SStructMatrix matrix )
{
   HYPRE_Int               nparts    = hypre_SStructMatrixNParts(matrix);
   hypre_SStructGraph     *graph     = hypre_SStructMatrixGraph(matrix);
   hypre_SStructPMatrix  **pmatrices = hypre_SStructMatrixPMatrices(matrix);
   HYPRE_Int            ***symmetric = hypre_SStructMatrixSymmetric(matrix);
   hypre_SStructStencil ***stencils  = hypre_SStructGraphStencils(graph);
   HYPRE_Int              *split;

   MPI_Comm                pcomm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **pstencils;
   HYPRE_Int               nvars;

   HYPRE_Int               stencil_size;
   hypre_Index            *stencil_shape;
   HYPRE_Int              *stencil_vars;
   HYPRE_Int               pstencil_ndim;
   HYPRE_Int               pstencil_size;

   HYPRE_Int               part, var, i;

   /* GEC0902 addition of variables for ilower and iupper   */
   MPI_Comm                comm;
   hypre_SStructGrid      *grid, *domain_grid;
   HYPRE_BigInt            ilower, iupper, jlower, jupper;
   HYPRE_Int               matrix_type = hypre_SStructMatrixObjectType(matrix);

   /* S-matrix */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      pstencils = hypre_TAlloc(hypre_SStructStencil *,  nvars, HYPRE_MEMORY_HOST);
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
      for (var = 0; var < nvars; var++)
      {
         for (i = 0; i < nvars; i++)
         {
            hypre_SStructPMatrixSetSymmetric(pmatrices[part], var, i,
                                             symmetric[part][var][i]);
         }
      }
      hypre_SStructPMatrixInitialize(pmatrices[part]);
   }

   /* U-matrix */

   /* GEC0902  knowing the kind of matrix we can create the IJMATRIX with the
    *  the right dimension (HYPRE_PARCSR without ghosts) */

   grid = hypre_SStructGraphGrid(graph);
   domain_grid = hypre_SStructGraphDomainGrid(graph);
   comm =  hypre_SStructMatrixComm(matrix);

   if (matrix_type == HYPRE_PARCSR)
   {
      ilower = hypre_SStructGridStartRank(grid);
      iupper = ilower + hypre_SStructGridLocalSize(grid) - 1;
      jlower = hypre_SStructGridStartRank(domain_grid);
      jupper = jlower + hypre_SStructGridLocalSize(domain_grid) - 1;
   }
   else if (matrix_type == HYPRE_SSTRUCT || matrix_type == HYPRE_STRUCT)
   {
      ilower = hypre_SStructGridGhstartRank(grid);
      iupper = ilower + hypre_SStructGridGhlocalSize(grid) - 1;
      jlower = hypre_SStructGridGhstartRank(domain_grid);
      jupper = jlower + hypre_SStructGridGhlocalSize(domain_grid) - 1;
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Invalid matrix type!\n");
      return hypre_error_flag;
   }

   HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper,
                        &hypre_SStructMatrixIJMatrix(matrix));

   hypre_SStructUMatrixInitialize(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixSetValues( HYPRE_SStructMatrix  matrix,
                              HYPRE_Int            part,
                              HYPRE_Int           *index,
                              HYPRE_Int            var,
                              HYPRE_Int            nentries,
                              HYPRE_Int           *entries,
                              HYPRE_Complex       *values )
{
   hypre_SStructMatrixSetValues(matrix, part, index, var,
                                nentries, entries, values, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixAddToValues( HYPRE_SStructMatrix  matrix,
                                HYPRE_Int            part,
                                HYPRE_Int           *index,
                                HYPRE_Int            var,
                                HYPRE_Int            nentries,
                                HYPRE_Int           *entries,
                                HYPRE_Complex       *values )
{
   hypre_SStructMatrixSetValues(matrix, part, index, var,
                                nentries, entries, values, 1);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D - RDF: Why? */

HYPRE_Int
HYPRE_SStructMatrixAddFEMValues( HYPRE_SStructMatrix  matrix,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *index,
                                 HYPRE_Complex       *values )
{
   HYPRE_Int           ndim         = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph *graph        = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid  *grid         = hypre_SStructGraphGrid(graph);
   HYPRE_Int           fem_nsparse  = hypre_SStructGraphFEMPNSparse(graph, part);
   HYPRE_Int          *fem_sparse_i = hypre_SStructGraphFEMPSparseI(graph, part);
   HYPRE_Int          *fem_entries  = hypre_SStructGraphFEMPEntries(graph, part);
   HYPRE_Int          *fem_vars     = hypre_SStructGridFEMPVars(grid, part);
   hypre_Index        *fem_offsets  = hypre_SStructGridFEMPOffsets(grid, part);
   HYPRE_Int           s, i, d, vindex[HYPRE_MAXDIM];

   /* Set one coefficient at a time */
   for (s = 0; s < fem_nsparse; s++)
   {
      i = fem_sparse_i[s];
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + hypre_IndexD(fem_offsets[i], d);
      }
      HYPRE_SStructMatrixAddToValues(
         matrix, part, vindex, fem_vars[i], 1, &fem_entries[s], &values[s]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixGetValues( HYPRE_SStructMatrix  matrix,
                              HYPRE_Int            part,
                              HYPRE_Int           *index,
                              HYPRE_Int            var,
                              HYPRE_Int            nentries,
                              HYPRE_Int           *entries,
                              HYPRE_Complex       *values )
{
   hypre_SStructMatrixSetValues(matrix, part, index, var,
                                nentries, entries, values, -1);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D - RDF: Why? */

HYPRE_Int
HYPRE_SStructMatrixGetFEMValues( HYPRE_SStructMatrix  matrix,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *index,
                                 HYPRE_Complex       *values )
{
   HYPRE_Int           ndim         = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph *graph        = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid  *grid         = hypre_SStructGraphGrid(graph);
   HYPRE_Int           fem_nsparse  = hypre_SStructGraphFEMPNSparse(graph, part);
   HYPRE_Int          *fem_sparse_i = hypre_SStructGraphFEMPSparseI(graph, part);
   HYPRE_Int          *fem_entries  = hypre_SStructGraphFEMPEntries(graph, part);
   HYPRE_Int          *fem_vars     = hypre_SStructGridFEMPVars(grid, part);
   hypre_Index        *fem_offsets  = hypre_SStructGridFEMPOffsets(grid, part);
   HYPRE_Int           s, i, d, vindex[HYPRE_MAXDIM];

   for (s = 0; s < fem_nsparse; s++)
   {
      i = fem_sparse_i[s];
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vindex[d] = index[d] + hypre_IndexD(fem_offsets[i], d);
      }
      hypre_SStructMatrixSetValues(
         matrix, part, vindex, fem_vars[i], 1, &fem_entries[s], &values[s], -1);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixSetBoxValues( HYPRE_SStructMatrix  matrix,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *ilower,
                                 HYPRE_Int           *iupper,
                                 HYPRE_Int            var,
                                 HYPRE_Int            nentries,
                                 HYPRE_Int           *entries,
                                 HYPRE_Complex       *values )
{
   HYPRE_SStructMatrixSetBoxValues2(matrix, part, ilower, iupper, var, nentries, entries,
                                    ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixAddToBoxValues( HYPRE_SStructMatrix  matrix,
                                   HYPRE_Int            part,
                                   HYPRE_Int           *ilower,
                                   HYPRE_Int           *iupper,
                                   HYPRE_Int            var,
                                   HYPRE_Int            nentries,
                                   HYPRE_Int           *entries,
                                   HYPRE_Complex       *values )
{
   HYPRE_SStructMatrixAddToBoxValues2(matrix, part, ilower, iupper, var, nentries, entries,
                                      ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixGetBoxValues( HYPRE_SStructMatrix  matrix,
                                 HYPRE_Int            part,
                                 HYPRE_Int           *ilower,
                                 HYPRE_Int           *iupper,
                                 HYPRE_Int            var,
                                 HYPRE_Int            nentries,
                                 HYPRE_Int           *entries,
                                 HYPRE_Complex       *values )
{
   HYPRE_SStructMatrixGetBoxValues2(matrix, part, ilower, iupper, var, nentries, entries,
                                    ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixSetBoxValues2( HYPRE_SStructMatrix  matrix,
                                  HYPRE_Int            part,
                                  HYPRE_Int           *ilower,
                                  HYPRE_Int           *iupper,
                                  HYPRE_Int            var,
                                  HYPRE_Int            nentries,
                                  HYPRE_Int           *entries,
                                  HYPRE_Int           *vilower,
                                  HYPRE_Int           *viupper,
                                  HYPRE_Complex       *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d, ndim = hypre_SStructMatrixNDim(matrix);

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(ndim);
   value_box = hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_SStructMatrixSetBoxValues(matrix, part, set_box, var, nentries, entries,
                                   value_box, values, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixAddToBoxValues2( HYPRE_SStructMatrix  matrix,
                                    HYPRE_Int            part,
                                    HYPRE_Int           *ilower,
                                    HYPRE_Int           *iupper,
                                    HYPRE_Int            var,
                                    HYPRE_Int            nentries,
                                    HYPRE_Int           *entries,
                                    HYPRE_Int           *vilower,
                                    HYPRE_Int           *viupper,
                                    HYPRE_Complex       *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d, ndim = hypre_SStructMatrixNDim(matrix);

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(ndim);
   value_box = hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_SStructMatrixSetBoxValues(matrix, part, set_box, var, nentries, entries,
                                   value_box, values, 1);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixGetBoxValues2( HYPRE_SStructMatrix  matrix,
                                  HYPRE_Int            part,
                                  HYPRE_Int           *ilower,
                                  HYPRE_Int           *iupper,
                                  HYPRE_Int            var,
                                  HYPRE_Int            nentries,
                                  HYPRE_Int           *entries,
                                  HYPRE_Int           *vilower,
                                  HYPRE_Int           *viupper,
                                  HYPRE_Complex       *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d, ndim = hypre_SStructMatrixNDim(matrix);

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(ndim);
   value_box = hypre_BoxCreate(ndim);

   for (d = 0; d < ndim; d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_SStructMatrixSetBoxValues(matrix, part, set_box, var, nentries, entries,
                                   value_box, values, -1);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixAddFEMBoxValues(HYPRE_SStructMatrix  matrix,
                                   HYPRE_Int            part,
                                   HYPRE_Int           *ilower,
                                   HYPRE_Int           *iupper,
                                   HYPRE_Complex       *values)
{
   HYPRE_Int             ndim            = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph   *graph           = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid    *grid            = hypre_SStructGraphGrid(graph);
   HYPRE_MemoryLocation  memory_location = hypre_SStructMatrixMemoryLocation(matrix);

   HYPRE_Int             fem_nsparse     = hypre_SStructGraphFEMPNSparse(graph, part);
   HYPRE_Int            *fem_sparse_i    = hypre_SStructGraphFEMPSparseI(graph, part);
   HYPRE_Int            *fem_entries     = hypre_SStructGraphFEMPEntries(graph, part);
   HYPRE_Int            *fem_vars        = hypre_SStructGridFEMPVars(grid, part);
   hypre_Index          *fem_offsets     = hypre_SStructGridFEMPOffsets(grid, part);

   HYPRE_Complex        *tvalues;
   hypre_Box            *box;

   HYPRE_Int             s, i, d, vilower[HYPRE_MAXDIM], viupper[HYPRE_MAXDIM];
   HYPRE_Int             ei, vi, nelts;

   /* Set one coefficient at a time */
   box = hypre_BoxCreate(ndim);
   hypre_BoxSetExtents(box, ilower, iupper);
   nelts = hypre_BoxVolume(box);
   tvalues = hypre_TAlloc(HYPRE_Complex, nelts, memory_location);

   for (s = 0; s < fem_nsparse; s++)
   {
      i = fem_sparse_i[s];
      for (d = 0; d < ndim; d++)
      {
         /* note: these offsets are different from what the user passes in */
         vilower[d] = ilower[d] + hypre_IndexD(fem_offsets[i], d);
         viupper[d] = iupper[d] + hypre_IndexD(fem_offsets[i], d);
      }

#if defined(HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
      {
         hypreDevice_ComplexStridedCopy(nelts, fem_nsparse, values + s, tvalues);
      }
      else
#endif
      {
         for (ei = 0, vi = s; ei < nelts; ei ++, vi += fem_nsparse)
         {
            tvalues[ei] = values[vi];
         }
      }

      HYPRE_SStructMatrixAddToBoxValues(matrix, part, vilower, viupper,
                                        fem_vars[i], 1, &fem_entries[s],
                                        tvalues);
   }

   /* Free memory */
   hypre_TFree(tvalues, memory_location);
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixAssemble( HYPRE_SStructMatrix matrix )
{
   HYPRE_Int               ndim           = hypre_SStructMatrixNDim(matrix);
   hypre_SStructGraph     *graph          = hypre_SStructMatrixGraph(matrix);
   HYPRE_Int               nparts         = hypre_SStructMatrixNParts(matrix);
   hypre_SStructPMatrix  **pmatrices      = hypre_SStructMatrixPMatrices(matrix);
   hypre_SStructGrid      *grid           = hypre_SStructGraphGrid(graph);
   hypre_SStructCommInfo **vnbor_comm_info = hypre_SStructGridVNborCommInfo(grid);
   HYPRE_Int               vnbor_ncomms    = hypre_SStructGridVNborNComms(grid);

   HYPRE_Int               part;

   hypre_CommInfo         *comm_info;
   HYPRE_Int               send_part,    recv_part;
   HYPRE_Int               send_var,     recv_var;
   hypre_StructMatrix     *send_matrix, *recv_matrix;
   hypre_CommPkg          *comm_pkg;
   hypre_CommHandle       *comm_handle;
   HYPRE_Int               ci;


   /*------------------------------------------------------
    * NOTE: Inter-part couplings were taken care of earlier.
    *------------------------------------------------------*/

   /*------------------------------------------------------
    * Communicate and accumulate within parts
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatrixAccumulate(pmatrices[part]);
   }

   /*------------------------------------------------------
    * Communicate and accumulate between parts
    *------------------------------------------------------*/

   for (ci = 0; ci < vnbor_ncomms; ci++)
   {
      comm_info = hypre_SStructCommInfoCommInfo(vnbor_comm_info[ci]);
      send_part = hypre_SStructCommInfoSendPart(vnbor_comm_info[ci]);
      recv_part = hypre_SStructCommInfoRecvPart(vnbor_comm_info[ci]);
      send_var  = hypre_SStructCommInfoSendVar(vnbor_comm_info[ci]);
      recv_var  = hypre_SStructCommInfoRecvVar(vnbor_comm_info[ci]);

      send_matrix = hypre_SStructPMatrixSMatrix(
                       hypre_SStructMatrixPMatrix(matrix, send_part), send_var, send_var);
      recv_matrix = hypre_SStructPMatrixSMatrix(
                       hypre_SStructMatrixPMatrix(matrix, recv_part), recv_var, recv_var);

      if ((send_matrix != NULL) && (recv_matrix != NULL))
      {
         hypre_StructStencil *send_stencil = hypre_StructMatrixStencil(send_matrix);
         hypre_StructStencil *recv_stencil = hypre_StructMatrixStencil(recv_matrix);
         HYPRE_Int            num_values, stencil_size, num_transforms;
         HYPRE_Int           *symm;
         HYPRE_Int           *v_to_s, *s_to_v;
         hypre_Index         *coords, *dirs;
         HYPRE_Int          **orders, *order;
         hypre_IndexRef       sentry0;
         hypre_Index          sentry1;
         HYPRE_Int            ti, si, i, j;

         /* to compute 'orders', remember that we are doing reverse communication */
         num_values = hypre_StructMatrixNumValues(recv_matrix);
         symm = hypre_StructMatrixSymmElements(recv_matrix);
         stencil_size = hypre_StructStencilSize(recv_stencil);
         v_to_s = hypre_TAlloc(HYPRE_Int,  num_values, HYPRE_MEMORY_HOST);
         s_to_v = hypre_TAlloc(HYPRE_Int,  stencil_size, HYPRE_MEMORY_HOST);
         for (si = 0, i = 0; si < stencil_size; si++)
         {
            s_to_v[si] = -1;
            if (symm[si] < 0)  /* this is a stored coefficient */
            {
               v_to_s[i] = si;
               s_to_v[si] = i;
               i++;
            }
         }
         hypre_CommInfoGetTransforms(comm_info, &num_transforms, &coords, &dirs);
         orders = hypre_TAlloc(HYPRE_Int *,  num_transforms, HYPRE_MEMORY_HOST);
         order = hypre_TAlloc(HYPRE_Int,  num_values, HYPRE_MEMORY_HOST);
         for (ti = 0; ti < num_transforms; ti++)
         {
            for (i = 0; i < num_values; i++)
            {
               si = v_to_s[i];
               sentry0 = hypre_StructStencilElement(recv_stencil, si);
               for (j = 0; j < ndim; j++)
               {
                  hypre_IndexD(sentry1, hypre_IndexD(coords[ti], j)) =
                     hypre_IndexD(sentry0, j) * hypre_IndexD(dirs[ti], j);
               }
               order[i] = hypre_StructStencilElementRank(send_stencil, sentry1);
               /* currently, both send and recv transforms are parsed */
               if (order[i] > -1)
               {
                  order[i] = s_to_v[order[i]];
               }
            }
            /* want order to indicate the natural order on the remote process */
            orders[ti] = hypre_TAlloc(HYPRE_Int,  num_values, HYPRE_MEMORY_HOST);
            for (i = 0; i < num_values; i++)
            {
               orders[ti][i] = -1;
            }
            for (i = 0; i < num_values; i++)
            {
               if (order[i] > -1)
               {
                  orders[ti][order[i]] = i;
               }
            }
         }
         hypre_TFree(v_to_s, HYPRE_MEMORY_HOST);
         hypre_TFree(s_to_v, HYPRE_MEMORY_HOST);
         hypre_TFree(order, HYPRE_MEMORY_HOST);

         /* want to communicate and add ghost data to real data */
         hypre_CommPkgCreate(comm_info,
                             hypre_StructMatrixDataSpace(send_matrix),
                             hypre_StructMatrixDataSpace(recv_matrix),
                             num_values, orders, 1,
                             hypre_StructMatrixComm(send_matrix), &comm_pkg);
         /* note reversal of send/recv data here */
         hypre_InitializeCommunication(comm_pkg,
                                       hypre_StructMatrixData(recv_matrix),
                                       hypre_StructMatrixData(send_matrix),
                                       1, 0, &comm_handle);
         hypre_FinalizeCommunication(comm_handle);
         hypre_CommPkgDestroy(comm_pkg);

         for (ti = 0; ti < num_transforms; ti++)
         {
            hypre_TFree(orders[ti], HYPRE_MEMORY_HOST);
         }
         hypre_TFree(orders, HYPRE_MEMORY_HOST);
      }
   }

   /*------------------------------------------------------
    * Assemble P and U matrices
    *------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatrixAssemble(pmatrices[part]);
   }

   /* U-matrix */
   hypre_SStructUMatrixAssemble(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Should set things up so that this information can be passed
 * immediately to the PMatrix.  Unfortunately, the PMatrix is
 * currently not created until the SStructMatrix is initialized.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixSetSymmetric( HYPRE_SStructMatrix matrix,
                                 HYPRE_Int           part,
                                 HYPRE_Int           var,
                                 HYPRE_Int           to_var,
                                 HYPRE_Int           symmetric )
{
   HYPRE_Int          ***msymmetric = hypre_SStructMatrixSymmetric(matrix);
   hypre_SStructGraph   *graph      = hypre_SStructMatrixGraph(matrix);
   hypre_SStructPGrid   *pgrid;

   HYPRE_Int pstart = part;
   HYPRE_Int psize  = 1;
   HYPRE_Int vstart = var;
   HYPRE_Int vsize  = 1;
   HYPRE_Int tstart = to_var;
   HYPRE_Int tsize  = 1;
   HYPRE_Int p, v, t;

   if (part == -1)
   {
      pstart = 0;
      psize  = hypre_SStructMatrixNParts(matrix);
   }

   for (p = pstart; p < psize; p++)
   {
      pgrid = hypre_SStructGraphPGrid(graph, p);
      if (var == -1)
      {
         vstart = 0;
         vsize  = hypre_SStructPGridNVars(pgrid);
      }
      if (to_var == -1)
      {
         tstart = 0;
         tsize  = hypre_SStructPGridNVars(pgrid);
      }

      for (v = vstart; v < vsize; v++)
      {
         for (t = tstart; t < tsize; t++)
         {
            msymmetric[p][v][t] = symmetric;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixSetNSSymmetric( HYPRE_SStructMatrix matrix,
                                   HYPRE_Int           symmetric )
{
   hypre_SStructMatrixNSSymmetric(matrix) = symmetric;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixSetObjectType( HYPRE_SStructMatrix  matrix,
                                  HYPRE_Int            type )
{
   hypre_SStructGraph     *graph    = hypre_SStructMatrixGraph(matrix);
   HYPRE_Int            ***splits   = hypre_SStructMatrixSplits(matrix);
   HYPRE_Int               nparts   = hypre_SStructMatrixNParts(matrix);
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);

   hypre_SStructPGrid     *pgrid;
   HYPRE_Int               nvars;
   HYPRE_Int               stencil_size;
   HYPRE_Int               part, var, i;

   hypre_SStructMatrixObjectType(matrix) = type ;

   /* RDF: This and all other modifications to 'split' really belong
    * in the Initialize routine */
   if (type != HYPRE_SSTRUCT && type != HYPRE_STRUCT)
   {
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
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixGetObject( HYPRE_SStructMatrix   matrix,
                              void                **object )
{
   HYPRE_Int             type     = hypre_SStructMatrixObjectType(matrix);
   hypre_SStructPMatrix *pmatrix;
   hypre_StructMatrix   *smatrix;
   HYPRE_Int             part, var;

   if (type == HYPRE_SSTRUCT)
   {
      *object = matrix;
   }
   else if (type == HYPRE_PARCSR)
   {
      *object = hypre_SStructMatrixParCSRMatrix(matrix);
   }
   else if (type == HYPRE_STRUCT)
   {
      /* only one part & one variable */
      part = 0;
      var = 0;
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      smatrix = hypre_SStructPMatrixSMatrix(pmatrix, var, var);
      *object = smatrix;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMatrixPrint
 *
 * This function prints a SStructMatrix to file. Assumptions:
 *
 *   1) All StructMatrices have the same number of ghost layers.
 *   2) Range and domain num_ghosts are equal.
 *
 * TODO: Add GPU support
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixPrint( const char          *filename,
                          HYPRE_SStructMatrix  matrix,
                          HYPRE_Int            all )
{
   /* Matrix variables */
   MPI_Comm                comm = hypre_SStructMatrixComm(matrix);
   HYPRE_Int               nparts = hypre_SStructMatrixNParts(matrix);
   hypre_SStructGraph     *graph = hypre_SStructMatrixGraph(matrix);
   hypre_SStructGrid      *grid = hypre_SStructGraphGrid(graph);
   hypre_SStructStencil ***stencils = hypre_SStructGraphStencils(graph);
   hypre_SStructPMatrix   *pmatrix;
   hypre_StructMatrix     *smatrix;
   HYPRE_Int               data_size;

   /* Local variables */
   FILE                   *file;
   HYPRE_Int               myid;
   HYPRE_Int               part;
   HYPRE_Int               var, vi, vj, nvars;
   HYPRE_Int               num_symm_calls;
   char                    new_filename[255];

   /* Sanity check */
   hypre_assert(nparts > 0);

   /* Print auxiliary info */
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.SMatrix.%05d", filename, myid);
   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_printf("Error: can't open output file %s\n", new_filename);
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   /* Print grid info */
   hypre_fprintf(file, "SStructMatrix\n");
   hypre_SStructGridPrint(file, grid);

   /* Print stencil info */
   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      nvars = hypre_SStructPMatrixNVars(pmatrix);

      for (var = 0; var < nvars; var++)
      {
         hypre_fprintf(file, "\nStencil - (Part %d, Var %d):\n", part, var);
         HYPRE_SStructStencilPrint(file, stencils[part][var]);
      }
   }
   hypre_fprintf(file, "\n");

   /* Print graph info */
   HYPRE_SStructGraphPrint(file, graph);

   /* Print symmetric info */
   num_symm_calls = 0;
   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      nvars = hypre_SStructPMatrixNVars(pmatrix);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            if (smatrix)
            {
               num_symm_calls++;
            }
         }
      }
   }
   hypre_fprintf(file, "\nMatrixNumSetSymmetric: %d", num_symm_calls);
   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      nvars = hypre_SStructPMatrixNVars(pmatrix);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            if (smatrix)
            {
               hypre_fprintf(file, "\nMatrixSetSymmetric: %d %d %d %d",
                             part, vi, vj, hypre_StructMatrixSymmetric(smatrix));
            }
         }
      }
   }
   hypre_fprintf(file, "\n");

   /* Print data */
   for (part = 0; part < nparts; part++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
      nvars = hypre_SStructPMatrixNVars(pmatrix);

      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            data_size = (smatrix) ? hypre_StructMatrixDataSize(smatrix) : 0;

            hypre_fprintf(file, "\nData - (Part %d, Vi %d, Vj %d): %d\n",
                          part, vi, vj, data_size);
            if (smatrix)
            {
               hypre_StructMatrixPrintData(file, smatrix, all);
            }
         }
      }
   }
   fclose(file);

   /* Print unstructured matrix (U-Matrix) */
   hypre_sprintf(new_filename, "%s.UMatrix", filename);
   HYPRE_IJMatrixPrint(hypre_SStructMatrixIJMatrix(matrix), new_filename);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructMatrixRead
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixRead( MPI_Comm              comm,
                         const char           *filename,
                         HYPRE_SStructMatrix  *matrix_ptr )
{
   /* Matrix variables */
   HYPRE_SStructMatrix     matrix;
   hypre_SStructPMatrix   *pmatrix;
   hypre_StructMatrix     *smatrix;
   HYPRE_SStructGrid       grid;
   hypre_SStructPGrid     *pgrid;
   HYPRE_SStructGraph      graph;
   HYPRE_SStructStencil  **stencils;
   HYPRE_Int               nparts;
   HYPRE_Int               nvars;
   HYPRE_Int               data_size;
   HYPRE_IJMatrix          umatrix;
   HYPRE_IJMatrix          h_umatrix;
   hypre_ParCSRMatrix     *h_parmatrix;
   hypre_ParCSRMatrix     *parmatrix = NULL;

   /* Local variables */
   FILE                   *file;
   HYPRE_Int               myid;
   HYPRE_Int               part, var;
   HYPRE_Int               p, v, i, j, vi, vj;
   HYPRE_Int               symmetric;
   HYPRE_Int               num_symm_calls;
   char                    new_filename[255];

   HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());

   hypre_MPI_Comm_rank(comm, &myid);

   /*-----------------------------------------------------------
    * Read S-Matrix
    *-----------------------------------------------------------*/

   hypre_sprintf(new_filename, "%s.SMatrix.%05d", filename, myid);
   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_printf("Error: can't open input file %s\n", new_filename);
      hypre_error_in_arg(2);

      return hypre_error_flag;
   }

   /* Read grid info */
   hypre_fscanf(file, "SStructMatrix\n");
   hypre_SStructGridRead(comm, file, &grid);
   nparts = hypre_SStructGridNParts(grid);

   /* Read stencil info */
   stencils = hypre_TAlloc(HYPRE_SStructStencil *, nparts, HYPRE_MEMORY_HOST);
   for (p = 0; p < nparts; p++)
   {
      pgrid = hypre_SStructGridPGrid(grid, p);
      nvars = hypre_SStructPGridNVars(pgrid);

      stencils[p] = hypre_TAlloc(HYPRE_SStructStencil, nvars, HYPRE_MEMORY_HOST);
      for (v = 0; v < nvars; v++)
      {
         hypre_fscanf(file, "\nStencil - (Part %d, Var %d):\n", &part, &var);
         HYPRE_SStructStencilRead(file, &stencils[part][var]);
      }
   }
   hypre_fscanf(file, "\n");

   /* Read graph info */
   HYPRE_SStructGraphRead(file, grid, stencils, &graph);

   /* Free memory */
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);

      for (var = 0; var < nvars; var++)
      {
         HYPRE_SStructStencilDestroy(stencils[part][var]);
      }
      hypre_TFree(stencils[part], HYPRE_MEMORY_HOST);
   }
   hypre_TFree(stencils, HYPRE_MEMORY_HOST);

   /* Assemble graph */
   HYPRE_SStructGraphAssemble(graph);

   /* Create matrix */
   HYPRE_SStructMatrixCreate(comm, graph, &matrix);

   /* Read symmetric info */
   hypre_fscanf(file, "\nMatrixNumSetSymmetric: %d", &num_symm_calls);
   for (i = 0; i < num_symm_calls; i++)
   {
      hypre_fscanf(file, "\nMatrixSetSymmetric: %d %d %d %d",
                   &part, &vi, &vj, &symmetric);
      HYPRE_SStructMatrixSetSymmetric(matrix, part, vi, vj, symmetric);
   }
   hypre_fscanf(file, "\n");

   /* Initialize matrix */
   HYPRE_SStructMatrixInitialize(matrix);

   /* Read data */
   for (p = 0; p < nparts; p++)
   {
      pmatrix = hypre_SStructMatrixPMatrix(matrix, p);
      nvars = hypre_SStructPMatrixNVars(pmatrix);

      for (i = 0; i < nvars; i++)
      {
         for (j = 0; j < nvars; j++)
         {
            hypre_fscanf(file, "\nData - (Part %d, Vi %d, Vj %d): %d\n",
                         &part, &vi, &vj, &data_size);

            pmatrix = hypre_SStructMatrixPMatrix(matrix, part);
            smatrix = hypre_SStructPMatrixSMatrix(pmatrix, vi, vj);
            if (data_size > 0)
            {
               hypre_StructMatrixReadData(file, smatrix);
            }
         }
      }
   }
   fclose(file);

   /*-----------------------------------------------------------
    * Read U-Matrix
    *-----------------------------------------------------------*/

   /* Read unstructured matrix from file using host memory */
   hypre_sprintf(new_filename, "%s.UMatrix", filename);
   HYPRE_IJMatrixRead(new_filename, comm, HYPRE_PARCSR, &h_umatrix);
   h_parmatrix = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(h_umatrix);

   /* Move ParCSRMatrix to device memory if necessary */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      parmatrix = hypre_ParCSRMatrixClone_v2(h_parmatrix, 1, memory_location);
   }
   else
   {
      parmatrix = h_parmatrix;
      hypre_IJMatrixObject(h_umatrix) = NULL;
   }

   /* Free memory */
   HYPRE_IJMatrixDestroy(h_umatrix);

   /* Update the umatrix with contents read from file,
      which now live on the correct memory location */
   umatrix = hypre_SStructMatrixIJMatrix(matrix);
   hypre_IJMatrixDestroyParCSR(umatrix);
   hypre_IJMatrixObject(umatrix) = (void*) parmatrix;
   hypre_SStructMatrixParCSRMatrix(matrix) = (hypre_ParCSRMatrix*) parmatrix;
   hypre_IJMatrixAssembleFlag(umatrix) = 1;

   /* Assemble SStructMatrix */
   HYPRE_SStructMatrixAssemble(matrix);

   /* Decrease ref counters */
   HYPRE_SStructGraphDestroy(graph);
   HYPRE_SStructGridDestroy(grid);

   *matrix_ptr = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructMatrixMatvec( HYPRE_Complex       alpha,
                           HYPRE_SStructMatrix A,
                           HYPRE_SStructVector x,
                           HYPRE_Complex       beta,
                           HYPRE_SStructVector y     )
{
   hypre_SStructMatvec(alpha, A, x, beta, y);

   return hypre_error_flag;
}
