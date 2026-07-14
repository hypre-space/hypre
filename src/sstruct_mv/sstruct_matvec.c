/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * SStruct matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#define TEST_INDEXESTOGLOBALRANKS

/*==========================================================================
 * PMatvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructPMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructPMatvecData_struct
{
   HYPRE_Int    nvars;
   HYPRE_Int    transpose;
   void      ***smatvec_data;
} hypre_SStructPMatvecData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecCreate( void **pmatvec_vdata_ptr )
{
   hypre_SStructPMatvecData *pmatvec_data;

   pmatvec_data = hypre_CTAlloc(hypre_SStructPMatvecData, 1, HYPRE_MEMORY_HOST);
   (pmatvec_data -> nvars)     = 0;
   (pmatvec_data -> transpose) = 0;

   *pmatvec_vdata_ptr = (void *) pmatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecSetTranspose( void      *pmatvec_vdata,
                                  HYPRE_Int  transpose )
{
   hypre_SStructPMatvecData  *pmatvec_data = (hypre_SStructPMatvecData *) pmatvec_vdata;

   (pmatvec_data -> transpose) = transpose;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecDestroy( void *pmatvec_vdata )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
   HYPRE_Int                   nvars;
   void                     ***smatvec_data;
   HYPRE_Int                   vi, vj;

   if (pmatvec_data)
   {
      nvars        = (pmatvec_data -> nvars);
      smatvec_data = (pmatvec_data -> smatvec_data);
      for (vi = 0; vi < nvars; vi++)
      {
         for (vj = 0; vj < nvars; vj++)
         {
            if (smatvec_data[vi][vj] != NULL)
            {
               hypre_StructMatvecDestroy(smatvec_data[vi][vj]);
            }
         }
         hypre_TFree(smatvec_data[vi], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(smatvec_data, HYPRE_MEMORY_HOST);
      hypre_TFree(pmatvec_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecSetup( void                 *pmatvec_vdata,
                           hypre_SStructPMatrix *pA,
                           hypre_SStructPVector *px )
{
   hypre_SStructPMatvecData    *pmatvec_data = (hypre_SStructPMatvecData *) pmatvec_vdata;
   HYPRE_Int                    transpose = (pmatvec_data -> transpose);
   HYPRE_Int                    nvars;
   void                      ***smatvec_data;
   hypre_StructMatrix          *sA;
   hypre_StructVector          *sx;
   HYPRE_Int                    vi, vj;

   nvars = hypre_SStructPMatrixNVars(pA);
   smatvec_data = hypre_TAlloc(void **, nvars, HYPRE_MEMORY_HOST);
   for (vi = 0; vi < nvars; vi++)
   {
      smatvec_data[vi] = hypre_TAlloc(void *, nvars, HYPRE_MEMORY_HOST);
      for (vj = 0; vj < nvars; vj++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
         sx = hypre_SStructPVectorSVector(px, vj);
         smatvec_data[vi][vj] = NULL;
         if (sA != NULL)
         {
            smatvec_data[vi][vj] = hypre_StructMatvecCreate();

            hypre_StructMatvecSetTranspose(smatvec_data[vi][vj], transpose);
            hypre_StructMatvecSetup(smatvec_data[vi][vj], sA, sx);
         }
      }
   }
   (pmatvec_data -> nvars)        = nvars;
   (pmatvec_data -> smatvec_data) = smatvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvecCompute( void                 *pmatvec_vdata,
                             HYPRE_Complex         alpha,
                             hypre_SStructPMatrix *pA,
                             hypre_SStructPVector *px,
                             HYPRE_Complex         beta,
                             hypre_SStructPVector *pb,
                             hypre_SStructPVector *py )
{
   hypre_SStructPMatvecData   *pmatvec_data = (hypre_SStructPMatvecData   *)pmatvec_vdata;
   HYPRE_Int                   nvars        = (pmatvec_data -> nvars);
   void                     ***smatvec_data = (pmatvec_data -> smatvec_data);

   hypre_SStructPGrid         *pgrid;
   hypre_StructMatrix         *sA;
   hypre_StructVector         *sx, *sb, *sy;
   HYPRE_Int                   vi, vj, active;
   void                       *sdata;

   for (vi = 0; vi < nvars; vi++)
   {
      pgrid  = hypre_SStructPMatrixPGrid(pA);
      active = hypre_SStructPGridActive(pgrid, vi);
      sb     = hypre_SStructPVectorSVector(pb, vi);
      sy     = hypre_SStructPVectorSVector(py, vi);

      if (active)
      {
         /* diagonal block computation */
         if (smatvec_data[vi][vi] != NULL)
         {
            sdata = smatvec_data[vi][vi];
            sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
            sx = hypre_SStructPVectorSVector(px, vi);
            hypre_StructMatvecCompute(sdata, alpha, sA, sx, beta, sb, sy);
         }
         else
         {
            hypre_StructVectorAxpy(0.0, sb, beta, sb, sy);
         }

         /* off-diagonal block computation */
         for (vj = 0; vj < nvars; vj++)
         {
            if ((smatvec_data[vi][vj] != NULL) && (vj != vi))
            {
               sdata = smatvec_data[vi][vj];
               sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
               sx = hypre_SStructPVectorSVector(px, vj);
               hypre_StructMatvecCompute(sdata, alpha, sA, sx, 1.0, sy, sy);
            }
         }
      }
      else
      {
         hypre_StructVectorAxpy(0.0, sb, beta, sb, sy);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: This does not seem to be used anywhere.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructPMatvec( HYPRE_Complex         alpha,
                      hypre_SStructPMatrix *pA,
                      hypre_SStructPVector *px,
                      HYPRE_Complex         beta,
                      hypre_SStructPVector *py )
{
   void *pmatvec_data;

   hypre_SStructPMatvecCreate(&pmatvec_data);
   hypre_SStructPMatvecSetup(pmatvec_data, pA, px);
   hypre_SStructPMatvecCompute(pmatvec_data, alpha, pA, px, beta, py, py);
   hypre_SStructPMatvecDestroy(pmatvec_data);

   return hypre_error_flag;
}

/*==========================================================================
 * Matvec routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 * hypre_SStructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructMatvecData_struct
{
   HYPRE_Int    nparts;
   HYPRE_Int    transpose;
   void       **pmatvec_data;
} hypre_SStructMatvecData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecCreate( void **matvec_vdata_ptr )
{
   hypre_SStructMatvecData *matvec_data;

   matvec_data = hypre_CTAlloc(hypre_SStructMatvecData, 1, HYPRE_MEMORY_HOST);
   (matvec_data -> nparts)    = 0;
   (matvec_data -> transpose) = 0;

   *matvec_vdata_ptr = (void *) matvec_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecSetTranspose( void      *matvec_vdata,
                                 HYPRE_Int  transpose )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData *) matvec_vdata;

   (matvec_data -> transpose) = transpose;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SStructMatvecCopyToParCSR( hypre_SStructMatrix *A,
                                           HYPRE_Int transpose,
                                           hypre_SStructVector *x )
{
   HYPRE_Int                 ndim        = hypre_SStructMatrixNDim(A);
   HYPRE_Int                 nparts      = hypre_SStructMatrixNParts(A);

   hypre_SStructGraph       *graph       = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid        *grid        = hypre_SStructGraphGrid(graph);
   hypre_SStructPGrid       *pgrid;
   hypre_SStructPVector     *pvector;
   hypre_StructVector       *svector;

   hypre_ParVector          *tmp;
   HYPRE_Complex            *tmp_data;
   HYPRE_Int                *copy_ranks;
   HYPRE_Int                *copy_ranks_part_var_starts;
   HYPRE_Int             ****copy_indexes;

   HYPRE_Int                 nvars, npartvars, num_ranks;
   HYPRE_Int                 part, var, i, j, d;
   HYPRE_Complex             val;
   HYPRE_Int                 index[HYPRE_MAXDIM];

   /* Get tmp vector and copy info in either the domain or range depending on transpose */
   if (transpose)
   {
      tmp = hypre_SStructMatrixRanTmp(A);
      copy_ranks = hypre_SStructMatrixRanCopyRanks(A);
      copy_ranks_part_var_starts = hypre_SStructMatrixRanCopyRanksPartVarStarts(A);
      copy_indexes = hypre_SStructMatrixRanCopyIndexes(A);
   }
   else
   {
      tmp = hypre_SStructMatrixDomTmp(A);
      copy_ranks = hypre_SStructMatrixDomCopyRanks(A);
      copy_ranks_part_var_starts = hypre_SStructMatrixDomCopyRanksPartVarStarts(A);
      copy_indexes = hypre_SStructMatrixDomCopyIndexes(A);
   }
   tmp_data = hypre_ParVectorLocalData(tmp);

   /* Loop over part/vars and copy from sstruct vector, x, into tmp par vector */
   npartvars = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      pvector = hypre_SStructVectorPVector(x, part);
      for (var = 0; var < nvars; var++)
      {
         /* WM: todo - GPU */
         svector = hypre_SStructPVectorSVector(pvector, var);
         num_ranks = copy_ranks_part_var_starts[npartvars + 1] - copy_ranks_part_var_starts[npartvars];
         for (i = 0, j = copy_ranks_part_var_starts[npartvars]; i < num_ranks; i++, j++)
         {
            for (d = 0; d < ndim; d++)
            {
               index[d] = copy_indexes[part][var][d][i];
            }
            hypre_StructVectorSetValues(svector, index, &val, -1, -1, 0);

            tmp_data[ copy_ranks[j] ] = val;
         }
         npartvars++;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecAddToSStruct( hypre_SStructMatrix *A,
                                 HYPRE_Int transpose,
                                 hypre_SStructVector *y )
{
   HYPRE_Int                 ndim        = hypre_SStructMatrixNDim(A);
   HYPRE_Int                 nparts      = hypre_SStructMatrixNParts(A);

   hypre_SStructGraph       *graph       = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid        *grid        = hypre_SStructGraphGrid(graph);
   hypre_SStructPGrid       *pgrid;

   hypre_ParVector          *tmp;
   HYPRE_Complex            *tmp_data;
   HYPRE_Int                *copy_ranks;
   HYPRE_Int                *copy_ranks_part_var_starts;
   HYPRE_Int             ****copy_indexes;

   HYPRE_Int                 nvars, npartvars, num_ranks;
   HYPRE_Int                 part, var, i, j, d;
   HYPRE_Complex             val;
   HYPRE_Int                 index[HYPRE_MAXDIM];

   /* Get tmp vector and copy info in either the domain or range depending on transpose */
   if (transpose)
   {
      tmp = hypre_SStructMatrixDomTmp(A);
      copy_ranks = hypre_SStructMatrixDomCopyRanks(A);
      copy_ranks_part_var_starts = hypre_SStructMatrixDomCopyRanksPartVarStarts(A);
      copy_indexes = hypre_SStructMatrixDomCopyIndexes(A);
   }
   else
   {
      tmp = hypre_SStructMatrixRanTmp(A);
      copy_ranks = hypre_SStructMatrixRanCopyRanks(A);
      copy_ranks_part_var_starts = hypre_SStructMatrixRanCopyRanksPartVarStarts(A);
      copy_indexes = hypre_SStructMatrixRanCopyIndexes(A);
   }
   tmp_data = hypre_ParVectorLocalData(tmp);

   /* Loop over part/vars and add values from tmp par vector to sstruct vector, y */
   npartvars = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = hypre_SStructGridPGrid(grid, part);
      nvars = hypre_SStructPGridNVars(pgrid);
      for (var = 0; var < nvars; var++)
      {
         /* WM: todo - GPU */
         num_ranks = copy_ranks_part_var_starts[npartvars + 1] - copy_ranks_part_var_starts[npartvars];
         for (i = 0, j = copy_ranks_part_var_starts[npartvars]; i < num_ranks; i++, j++)
         {
            for (d = 0; d < ndim; d++)
            {
               index[d] = copy_indexes[part][var][d][i];
            }
            val = tmp_data[ copy_ranks[j] ];
            HYPRE_SStructVectorAddToValues(y, part, index, var, &val);
         }
         npartvars++;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecSetup( void                *matvec_vdata,
                          hypre_SStructMatrix *A,
                          hypre_SStructVector *x )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData   *)matvec_vdata;
   HYPRE_Int                 transpose   = (matvec_data -> transpose);

   HYPRE_Int                 nparts;
   HYPRE_Int                 part;
   void                    **pmatvec_data;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px;

#if defined(HYPRE_SSTRUCT_MATVEC_COPY)
   hypre_SStructGraph       *graph       = hypre_SStructMatrixGraph(A);
   hypre_SStructGrid        *grid        = hypre_SStructGraphGrid(graph);
   /* WM: todo - There is no explicitly stored range grid?
    *            Is the range grid always assumed to be the same as grid above? */
   hypre_SStructGrid        *dom_grid    = hypre_SStructGraphDomGrid(graph);
   hypre_SStructPGrid       *pgrid;

   HYPRE_Int                 object_type     = hypre_SStructMatrixObjectType(A);
   hypre_ParCSRMatrix       *parcsr_A        = hypre_SStructMatrixParCSRMatrix(A);
   HYPRE_MemoryLocation      memory_location = hypre_ParCSRMatrixMemoryLocation(parcsr_A);
   hypre_ParCSRCommPkg      *comm_pkg;

   HYPRE_Int                 nvars, npartvars, num_ranks;
   HYPRE_Int                 var, i;

   HYPRE_Int                 num_col_ind, num_send_map_elmts;
   HYPRE_Int                 diag_num_rownnz, offd_num_rownnz;

   HYPRE_Int                 dom_copy_ranks_size, ran_copy_ranks_size;
   HYPRE_Int                *dom_copy_ranks;
   HYPRE_Int                *ran_copy_ranks;
   HYPRE_Int                *dom_copy_ranks_part_var_starts;
   HYPRE_Int                *ran_copy_ranks_part_var_starts;
   HYPRE_Int                *dom_copy_global_ranks;
   HYPRE_Int                *ran_copy_global_ranks;
   HYPRE_Int             ****dom_copy_indexes;
   HYPRE_Int             ****ran_copy_indexes;
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   nparts = hypre_SStructMatrixNParts(A);
   pmatvec_data = hypre_TAlloc(void *, nparts, HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      hypre_SStructPMatvecCreate(&pmatvec_data[part]);
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);

      hypre_SStructPMatvecSetTranspose(pmatvec_data[part], transpose);
      hypre_SStructPMatvecSetup(pmatvec_data[part], pA, px);
   }
   (matvec_data -> nparts)       = nparts;
   (matvec_data -> pmatvec_data) = pmatvec_data;


#if defined(HYPRE_SSTRUCT_MATVEC_COPY)
   /* If necessary, setup the temporary vectors and copy info
    * needed for parcsr component of the matvec*/
   if (object_type == HYPRE_SSTRUCT && !hypre_SStructMatrixDomTmp(A))
   {
      /* Initialize temporary domain and range vectors */
      hypre_SStructMatrixDomTmp(A) = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parcsr_A),
                                                         hypre_ParCSRMatrixGlobalNumCols(parcsr_A),
                                                         hypre_ParCSRMatrixColStarts(parcsr_A));
      hypre_ParVectorInitialize_v2(hypre_SStructMatrixDomTmp(A), memory_location);
      hypre_SStructMatrixRanTmp(A) = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parcsr_A),
                                                         hypre_ParCSRMatrixGlobalNumRows(parcsr_A),
                                                         hypre_ParCSRMatrixRowStarts(parcsr_A));
      hypre_ParVectorInitialize_v2(hypre_SStructMatrixRanTmp(A), memory_location);

      /* Get the dom_copy_ranks (comprised by the sorted and uniqued
       * local column indices and send map elements of parcsr_A) */
      comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(parcsr_A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(parcsr_A);
      }
      num_col_ind = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(parcsr_A));
      num_send_map_elmts = hypre_ParCSRCommPkgSendMapStart(comm_pkg, hypre_ParCSRCommPkgNumSends(comm_pkg));
      dom_copy_ranks_size = num_col_ind + num_send_map_elmts;
      dom_copy_ranks = hypre_TAlloc(HYPRE_Int, dom_copy_ranks_size, memory_location);
      /* WM: todo - use device send map elmts when on GPU? */
      hypre_TMemcpy(dom_copy_ranks, hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(parcsr_A)),
                    HYPRE_Int, num_col_ind, memory_location, memory_location);
      hypre_TMemcpy(dom_copy_ranks + num_col_ind, hypre_ParCSRCommPkgSendMapElmts(comm_pkg),
                    HYPRE_Int, num_send_map_elmts, memory_location, HYPRE_MEMORY_HOST);
      hypre_UniqueIntArrayND(1, &dom_copy_ranks_size, &dom_copy_ranks);
      dom_copy_ranks = hypre_TReAlloc(dom_copy_ranks, HYPRE_Int, dom_copy_ranks_size, memory_location);

      /* Convert local to global ranks, get the part/var starts,
       * then map to part/var/index to obtain dom_copy_indexes */
      dom_copy_global_ranks = hypre_TAlloc(HYPRE_BigInt, dom_copy_ranks_size, memory_location);
      for (i = 0; i < dom_copy_ranks_size; i++)
      {
         dom_copy_global_ranks[i] = hypre_ParCSRMatrixFirstColDiag(parcsr_A) + (HYPRE_BigInt) dom_copy_ranks[i];
      }
      hypre_SStructGridGetGlobalRanksPartVarStarts(dom_grid,
                                                   object_type,
                                                   dom_copy_ranks_size,
                                                   dom_copy_global_ranks,
                                                   &(dom_copy_ranks_part_var_starts));
      npartvars = 0;
      dom_copy_indexes = hypre_CTAlloc(HYPRE_Int***, nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(dom_grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);
         dom_copy_indexes[part] = hypre_CTAlloc(HYPRE_Int**, nvars, HYPRE_MEMORY_HOST);
         for (var = 0; var < nvars; var++)
         {
            num_ranks = dom_copy_ranks_part_var_starts[npartvars + 1] - dom_copy_ranks_part_var_starts[npartvars];
            if (num_ranks)
            {
               hypre_SStructGridGlobalRanksToIndexes(dom_grid, object_type, part, var, num_ranks,
                                                     &(dom_copy_global_ranks[dom_copy_ranks_part_var_starts[npartvars]]),
                                                     &(dom_copy_indexes[part][var]));
#if defined(TEST_INDEXESTOGLOBALRANKS)
               /* WM: debug - check that mapping back to global ranks works */
               HYPRE_BigInt *global_ranks_check = hypre_TAlloc(HYPRE_BigInt, num_ranks, HYPRE_MEMORY_HOST);
               hypre_SStructGridIndexesToGlobalRanks(dom_grid, object_type, part, var,
                                                 num_ranks, dom_copy_indexes[part][var],
                                                 &global_ranks_check);
               for (i = 0; i < num_ranks; i++)
               {
                  /* hypre_printf("WM: debug - global_ranks_check = %b, dom_copy_global_ranks = %b\n", global_ranks_check[i], dom_copy_global_ranks[ dom_copy_ranks_part_var_starts[npartvars] + i ]); */
                  assert( global_ranks_check[i] == dom_copy_global_ranks[ dom_copy_ranks_part_var_starts[npartvars] + i ] );
               }
               hypre_TFree(global_ranks_check, HYPRE_MEMORY_HOST);
#endif
            }
            npartvars++;
         }
      }
      hypre_TFree(dom_copy_global_ranks, memory_location);
      hypre_SStructMatrixDomCopyRanks(A) = dom_copy_ranks;
      hypre_SStructMatrixDomCopyRanksPartVarStarts(A) = dom_copy_ranks_part_var_starts;
      hypre_SStructMatrixDomCopyIndexes(A) = dom_copy_indexes;

      /* Get the ran_copy_ranks (comprised by non-zero rows of parcsr_A) */
      /* WM: todo - do I have to worry about rownnz not being set here? */
      diag_num_rownnz = hypre_CSRMatrixNumRownnz(hypre_ParCSRMatrixDiag(parcsr_A));
      offd_num_rownnz = hypre_CSRMatrixNumRownnz(hypre_ParCSRMatrixOffd(parcsr_A));
      ran_copy_ranks_size = diag_num_rownnz + offd_num_rownnz;
      ran_copy_ranks = hypre_TAlloc(HYPRE_Int, ran_copy_ranks_size, memory_location);
      hypre_TMemcpy(ran_copy_ranks, hypre_CSRMatrixRownnz(hypre_ParCSRMatrixDiag(parcsr_A)),
                    HYPRE_Int, diag_num_rownnz, memory_location, memory_location);
      hypre_TMemcpy(ran_copy_ranks + diag_num_rownnz, hypre_CSRMatrixRownnz(hypre_ParCSRMatrixOffd(parcsr_A)),
                    HYPRE_Int, offd_num_rownnz, memory_location, memory_location);
      hypre_UniqueIntArrayND(1, &ran_copy_ranks_size, &ran_copy_ranks);
      ran_copy_ranks = hypre_TReAlloc(ran_copy_ranks, HYPRE_Int, ran_copy_ranks_size, memory_location);

      /* Convert local to global ranks, get the part/var starts,
       * then map to part/var/index to obtain ran_copy_indexes */
      ran_copy_global_ranks = hypre_TAlloc(HYPRE_BigInt, ran_copy_ranks_size, memory_location);
      for (i = 0; i < ran_copy_ranks_size; i++)
      {
         ran_copy_global_ranks[i] = hypre_ParCSRMatrixFirstRowIndex(parcsr_A) + (HYPRE_BigInt) ran_copy_ranks[i];
      }
      hypre_SStructGridGetGlobalRanksPartVarStarts(grid,
                                                   object_type,
                                                   ran_copy_ranks_size,
                                                   ran_copy_global_ranks,
                                                   &(ran_copy_ranks_part_var_starts));
      npartvars = 0;
      ran_copy_indexes = hypre_CTAlloc(HYPRE_Int***, nparts, HYPRE_MEMORY_HOST);
      for (part = 0; part < nparts; part++)
      {
         pgrid = hypre_SStructGridPGrid(grid, part);
         nvars = hypre_SStructPGridNVars(pgrid);
         ran_copy_indexes[part] = hypre_CTAlloc(HYPRE_Int**, nvars, HYPRE_MEMORY_HOST);
         for (var = 0; var < nvars; var++)
         {
            num_ranks = ran_copy_ranks_part_var_starts[npartvars + 1] - ran_copy_ranks_part_var_starts[npartvars];
            if (num_ranks)
            {
               hypre_SStructGridGlobalRanksToIndexes(grid, object_type, part, var, num_ranks,
                                                     &(ran_copy_global_ranks[ran_copy_ranks_part_var_starts[npartvars]]),
                                                     &(ran_copy_indexes[part][var]));
#if defined(TEST_INDEXESTOGLOBALRANKS)
               /* WM: debug - check that mapping back to global ranks works */
               HYPRE_BigInt *global_ranks_check = hypre_TAlloc(HYPRE_BigInt, num_ranks, HYPRE_MEMORY_HOST);
               hypre_SStructGridIndexesToGlobalRanks(grid, object_type, part, var,
                                                 num_ranks, ran_copy_indexes[part][var],
                                                 &global_ranks_check);
               for (i = 0; i < num_ranks; i++)
               {
                  /* hypre_printf("WM: debug - global_ranks_check = %b, ran_copy_global_ranks = %b\n", global_ranks_check[i], ran_copy_global_ranks[ dom_copy_ranks_part_var_starts[npartvars] + i ]); */
                  assert( global_ranks_check[i] == ran_copy_global_ranks[ ran_copy_ranks_part_var_starts[npartvars] + i ] );
               }
               hypre_TFree(global_ranks_check, HYPRE_MEMORY_HOST);
#endif
            }
            npartvars++;
         }
      }
      hypre_TFree(ran_copy_global_ranks, memory_location);
      hypre_SStructMatrixRanCopyRanks(A) = ran_copy_ranks;
      hypre_SStructMatrixRanCopyRanksPartVarStarts(A) = ran_copy_ranks_part_var_starts;
      hypre_SStructMatrixRanCopyIndexes(A) = ran_copy_indexes;
   }
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute y = alpha*A*x + beta*b
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecCompute( void                *matvec_vdata,
                            HYPRE_Complex        alpha,
                            hypre_SStructMatrix *A,
                            hypre_SStructVector *x,
                            HYPRE_Complex        beta,
                            hypre_SStructVector *b,
                            hypre_SStructVector *y )
{
   hypre_SStructMatvecData  *matvec_data  = (hypre_SStructMatvecData   *)matvec_vdata;
   HYPRE_Int                 nparts       = (matvec_data -> nparts);
   HYPRE_Int                 transpose    = (matvec_data -> transpose);
   void                    **pmatvec_data = (matvec_data -> pmatvec_data);

   void                     *pdata;
   hypre_SStructPMatrix     *pA;
   hypre_SStructPVector     *px, *pb, *py;

   hypre_ParCSRMatrix       *parcsrA = hypre_SStructMatrixParCSRMatrix(A);
   hypre_ParVector          *parx, *parb, *pary;

   HYPRE_Int                 x_object_type = hypre_SStructVectorObjectType(x);
   HYPRE_Int                 A_object_type = hypre_SStructMatrixObjectType(A);
   HYPRE_Int                 part;

   if (x_object_type != A_object_type)
   {
      hypre_error_in_arg(2);
      hypre_error_in_arg(3);

      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("SStructMatvecCompute");

   if ( (x_object_type == HYPRE_SSTRUCT) || (x_object_type == HYPRE_STRUCT) )
   {
      /* do S-matrix computations */
      hypre_GpuProfilingPushRange("Structured");
      for (part = 0; part < nparts; part++)
      {
         pA = hypre_SStructMatrixPMatrix(A, part);
         px = hypre_SStructVectorPVector(x, part);
         pb = hypre_SStructVectorPVector(b, part);
         py = hypre_SStructVectorPVector(y, part);
         pdata = pmatvec_data[part];

         hypre_SStructPMatvecCompute(pdata, alpha, pA, px, beta, pb, py);
      }
      hypre_GpuProfilingPopRange();

      hypre_GpuProfilingPushRange("Unstructured");
      if (x_object_type == HYPRE_SSTRUCT)
      {
         /* do U-matrix computations */

#if defined(HYPRE_SSTRUCT_MATVEC_COPY)
         /* Fill appropriate domain/range tmp par vector with values from x */
         hypre_SStructMatvecCopyToParCSR(A, transpose, x);

         /* Do the parcsr matvec */
         if (transpose)
         {
            hypre_ParCSRMatrixMatvecT(alpha,
                                      parcsrA,
                                      hypre_SStructMatrixRanTmp(A),
                                      0.0,
                                      hypre_SStructMatrixDomTmp(A));
         }
         else
         {
            hypre_ParCSRMatrixMatvec(alpha,
                                     parcsrA,
                                     hypre_SStructMatrixDomTmp(A),
                                     0.0,
                                     hypre_SStructMatrixRanTmp(A));
         }

         /* Add the matvec result to y */
         hypre_SStructMatvecAddToSStruct(A, transpose, y);
#else
         /* GEC1002 the data chunk pointed by the local-parvectors
          *  inside the semistruct vectors x and y is now identical to the
          *  data chunk of the structure vectors x and y. The role of the function
          *  convert is to pass the addresses of the data chunk
          *  to the parx and pary. */

         hypre_SStructVectorConvert(x, &parx);
         hypre_SStructVectorConvert(y, &pary);

         if (transpose)
         {
            hypre_ParCSRMatrixMatvecT(alpha, parcsrA, parx, 1.0, pary);
         }
         else
         {
            hypre_ParCSRMatrixMatvec(alpha, parcsrA, parx, 1.0, pary);
         }

         /* dummy functions since there is nothing to restore  */
         hypre_SStructVectorRestore(x, parx);
         hypre_SStructVectorRestore(y, pary);
#endif

      }
      hypre_GpuProfilingPopRange();
   }
   else if (x_object_type == HYPRE_PARCSR)
   {
      hypre_SStructVectorConvert(x, &parx);
      hypre_SStructVectorConvert(b, &parb);
      hypre_SStructVectorConvert(y, &pary);

      hypre_ParCSRMatrixMatvecOutOfPlace(alpha, parcsrA, parx, beta, parb, pary);

      hypre_SStructVectorRestore(x, parx);
      hypre_SStructVectorRestore(b, parb);
      hypre_SStructVectorRestore(y, pary);
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvecDestroy( void *matvec_vdata )
{
   hypre_SStructMatvecData  *matvec_data = (hypre_SStructMatvecData   *)matvec_vdata;
   HYPRE_Int                 nparts;
   void                    **pmatvec_data;
   HYPRE_Int                 part;

   if (matvec_data)
   {
      nparts       = (matvec_data -> nparts);
      pmatvec_data = (matvec_data -> pmatvec_data);
      for (part = 0; part < nparts; part++)
      {
         hypre_SStructPMatvecDestroy(pmatvec_data[part]);
      }
      hypre_TFree(pmatvec_data, HYPRE_MEMORY_HOST);
      hypre_TFree(matvec_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SStructMatvec( HYPRE_Complex        alpha,
                     hypre_SStructMatrix *A,
                     hypre_SStructVector *x,
                     HYPRE_Complex        beta,
                     hypre_SStructVector *y )
{
   void *matvec_data;

   hypre_SStructMatvecCreate(&matvec_data);
   hypre_SStructMatvecSetup(matvec_data, A, x);
   hypre_SStructMatvecCompute(matvec_data, alpha, A, x, beta, y, y);
   hypre_SStructMatvecDestroy(matvec_data);

   return hypre_error_flag;
}
