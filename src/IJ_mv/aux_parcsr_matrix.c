/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_AuxParCSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "aux_parcsr_matrix.h"

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParCSRMatrixCreate( hypre_AuxParCSRMatrix **aux_matrix,
                             HYPRE_Int               local_num_rows,
                             HYPRE_Int               local_num_cols,
                             HYPRE_Int              *sizes )
{
   hypre_AuxParCSRMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_AuxParCSRMatrix,  1, HYPRE_MEMORY_HOST);

   hypre_AuxParCSRMatrixLocalNumRows(matrix) = local_num_rows;
   hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rows;
   hypre_AuxParCSRMatrixLocalNumCols(matrix) = local_num_cols;

   hypre_AuxParCSRMatrixRowSpace(matrix) = sizes;

   /* set defaults */
   hypre_AuxParCSRMatrixNeedAux(matrix) = 1;
   hypre_AuxParCSRMatrixMaxOffProcElmts(matrix) = 0;
   hypre_AuxParCSRMatrixCurrentOffProcElmts(matrix) = 0;
   hypre_AuxParCSRMatrixOffProcIIndx(matrix) = 0;
   hypre_AuxParCSRMatrixRownnz(matrix) = NULL;
   hypre_AuxParCSRMatrixRowLength(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxJ(matrix) = NULL;
   hypre_AuxParCSRMatrixAuxData(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxDiag(matrix) = NULL;
   hypre_AuxParCSRMatrixIndxOffd(matrix) = NULL;
   hypre_AuxParCSRMatrixDiagSizes(matrix) = NULL;
   hypre_AuxParCSRMatrixOffdSizes(matrix) = NULL;
   /* stash for setting or adding on/off-proc values */
   hypre_AuxParCSRMatrixOffProcI(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcJ(matrix) = NULL;
   hypre_AuxParCSRMatrixOffProcData(matrix) = NULL;
   hypre_AuxParCSRMatrixMemoryLocation(matrix) = HYPRE_MEMORY_HOST;
#if defined(HYPRE_USING_GPU)
   hypre_AuxParCSRMatrixMaxStackElmts(matrix) = 0;
   hypre_AuxParCSRMatrixCurrentStackElmts(matrix) = 0;
   hypre_AuxParCSRMatrixStackI(matrix) = NULL;
   hypre_AuxParCSRMatrixStackJ(matrix) = NULL;
   hypre_AuxParCSRMatrixStackData(matrix) = NULL;
   hypre_AuxParCSRMatrixStackSorA(matrix) = NULL;
   hypre_AuxParCSRMatrixUsrOnProcElmts(matrix) = -1;
   hypre_AuxParCSRMatrixUsrOffProcElmts(matrix) = -1;
   hypre_AuxParCSRMatrixInitAllocFactor(matrix) = 5;
   hypre_AuxParCSRMatrixGrowFactor(matrix) = 2;
#endif

   *aux_matrix = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParCSRMatrixDestroy( hypre_AuxParCSRMatrix *matrix )
{
   HYPRE_Int   num_rownnz;
   HYPRE_Int   num_rows;
   HYPRE_Int  *rownnz;
   HYPRE_Int   i;

   if (matrix)
   {
      rownnz     = hypre_AuxParCSRMatrixRownnz(matrix);
      num_rownnz = hypre_AuxParCSRMatrixLocalNumRownnz(matrix);
      num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);

      if (hypre_AuxParCSRMatrixAuxJ(matrix))
      {
         if (hypre_AuxParCSRMatrixRownnz(matrix))
         {
            for (i = 0; i < num_rownnz; i++)
            {
               hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix)[rownnz[i]], HYPRE_MEMORY_HOST);
            }
         }
         else
         {
            for (i = 0; i < num_rows; i++)
            {
               hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix)[i], HYPRE_MEMORY_HOST);
            }
         }

         hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix), HYPRE_MEMORY_HOST);
      }

      if (hypre_AuxParCSRMatrixAuxData(matrix))
      {
         if (hypre_AuxParCSRMatrixRownnz(matrix))
         {
            for (i = 0; i < num_rownnz; i++)
            {
               hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix)[rownnz[i]], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix), HYPRE_MEMORY_HOST);
         }
         else
         {
            for (i = 0; i < num_rows; i++)
            {
               hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix)[i], HYPRE_MEMORY_HOST);
            }
            hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix), HYPRE_MEMORY_HOST);
         }
      }

      hypre_TFree(hypre_AuxParCSRMatrixRownnz(matrix), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParCSRMatrixRowLength(matrix), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParCSRMatrixRowSpace(matrix), HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_AuxParCSRMatrixIndxDiag(matrix), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParCSRMatrixIndxOffd(matrix), HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_AuxParCSRMatrixDiagSizes(matrix), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParCSRMatrixOffdSizes(matrix), HYPRE_MEMORY_HOST);

      hypre_TFree(hypre_AuxParCSRMatrixOffProcI(matrix),    HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParCSRMatrixOffProcJ(matrix),    HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParCSRMatrixOffProcData(matrix), HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_GPU)
      hypre_TFree(hypre_AuxParCSRMatrixStackI(matrix),    hypre_AuxParCSRMatrixMemoryLocation(matrix));
      hypre_TFree(hypre_AuxParCSRMatrixStackJ(matrix),    hypre_AuxParCSRMatrixMemoryLocation(matrix));
      hypre_TFree(hypre_AuxParCSRMatrixStackData(matrix), hypre_AuxParCSRMatrixMemoryLocation(matrix));
      hypre_TFree(hypre_AuxParCSRMatrixStackSorA(matrix), hypre_AuxParCSRMatrixMemoryLocation(matrix));
#endif

      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixSetRownnz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParCSRMatrixSetRownnz( hypre_AuxParCSRMatrix *matrix )
{
   HYPRE_Int   local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
   HYPRE_Int  *row_space      = hypre_AuxParCSRMatrixRowSpace(matrix);
   HYPRE_Int   num_rownnz_old = hypre_AuxParCSRMatrixLocalNumRownnz(matrix);
   HYPRE_Int  *rownnz_old     = hypre_AuxParCSRMatrixRownnz(matrix);
   HYPRE_Int  *rownnz;

   HYPRE_Int   i, ii, local_num_rownnz;

   /* Count number of nonzero rows */
   local_num_rownnz = 0;
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:local_num_rownnz) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < local_num_rows; i++)
   {
      if (row_space[i] > 0)
      {
         local_num_rownnz++;
      }
   }

   if (local_num_rownnz != local_num_rows)
   {
      rownnz = hypre_CTAlloc(HYPRE_Int, local_num_rownnz, HYPRE_MEMORY_HOST);

      /* Find nonzero rows */
      local_num_rownnz = 0;
      for (i = 0; i < local_num_rows; i++)
      {
         if (row_space[i] > 0)
         {
            rownnz[local_num_rownnz++] = i;
         }
      }

      /* Free memory if necessary */
      if (rownnz_old && rownnz && (local_num_rownnz < num_rownnz_old))
      {
         ii = 0;
         for (i = 0; i < num_rownnz_old; i++)
         {
            if (rownnz_old[i] == rownnz[ii])
            {
               ii++;
            }
            else
            {
               hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix)[rownnz_old[i]], HYPRE_MEMORY_HOST);
               hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix)[rownnz_old[i]], HYPRE_MEMORY_HOST);
            }

            if (ii == local_num_rownnz)
            {
               i = i + 1;
               for (; i < num_rownnz_old; i++)
               {
                  hypre_TFree(hypre_AuxParCSRMatrixAuxJ(matrix)[rownnz_old[i]],
                              HYPRE_MEMORY_HOST);
                  hypre_TFree(hypre_AuxParCSRMatrixAuxData(matrix)[rownnz_old[i]],
                              HYPRE_MEMORY_HOST);
               }
               break;
            }
         }
      }
      hypre_TFree(rownnz_old, HYPRE_MEMORY_HOST);

      hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rownnz;
      hypre_AuxParCSRMatrixRownnz(matrix) = rownnz;
   }
   else
   {
      hypre_TFree(rownnz_old, HYPRE_MEMORY_HOST);
      hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rows;
      hypre_AuxParCSRMatrixRownnz(matrix) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParCSRMatrixInitialize_v2
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_AuxParCSRMatrixInitialize_v2( hypre_AuxParCSRMatrix *matrix,
                                    HYPRE_MemoryLocation memory_location )
{
   HYPRE_Int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(matrix);
   HYPRE_Int max_off_proc_elmts = hypre_AuxParCSRMatrixMaxOffProcElmts(matrix);

   hypre_AuxParCSRMatrixMemoryLocation(matrix) = memory_location;

   if (local_num_rows < 0)
   {
      return -1;
   }

   if (local_num_rows == 0)
   {
      return 0;
   }

#if defined(HYPRE_USING_GPU)
   if (memory_location != HYPRE_MEMORY_HOST)
   {
      /* GPU assembly */
      hypre_AuxParCSRMatrixNeedAux(matrix) = 1;
   }
   else
#endif
   {
      /* CPU assembly */
      /* allocate stash for setting or adding off processor values */
      if (max_off_proc_elmts > 0)
      {
         hypre_AuxParCSRMatrixOffProcI(matrix)    = hypre_CTAlloc(HYPRE_BigInt, 2 * max_off_proc_elmts,
                                                                  HYPRE_MEMORY_HOST);
         hypre_AuxParCSRMatrixOffProcJ(matrix)    = hypre_CTAlloc(HYPRE_BigInt,   max_off_proc_elmts,
                                                                  HYPRE_MEMORY_HOST);
         hypre_AuxParCSRMatrixOffProcData(matrix) = hypre_CTAlloc(HYPRE_Complex,  max_off_proc_elmts,
                                                                  HYPRE_MEMORY_HOST);
      }

      if (hypre_AuxParCSRMatrixNeedAux(matrix))
      {
         HYPRE_Int      *row_space = hypre_AuxParCSRMatrixRowSpace(matrix);
         HYPRE_Int      *rownnz    = hypre_AuxParCSRMatrixRownnz(matrix);
         HYPRE_BigInt  **aux_j     = hypre_CTAlloc(HYPRE_BigInt *,  local_num_rows, HYPRE_MEMORY_HOST);
         HYPRE_Complex **aux_data  = hypre_CTAlloc(HYPRE_Complex *, local_num_rows, HYPRE_MEMORY_HOST);

         HYPRE_Int       local_num_rownnz;
         HYPRE_Int       i, ii;

         if (row_space)
         {
            /* Count number of nonzero rows */
            local_num_rownnz = 0;
            for (i = 0; i < local_num_rows; i++)
            {
               if (row_space[i] > 0)
               {
                  local_num_rownnz++;
               }
            }

            if (local_num_rownnz != local_num_rows)
            {
               rownnz = hypre_CTAlloc(HYPRE_Int, local_num_rownnz, HYPRE_MEMORY_HOST);

               /* Find nonzero rows */
               local_num_rownnz = 0;
               for (i = 0; i < local_num_rows; i++)
               {
                  if (row_space[i] > 0)
                  {
                     rownnz[local_num_rownnz++] = i;
                  }
               }

               hypre_AuxParCSRMatrixLocalNumRownnz(matrix) = local_num_rownnz;
               hypre_AuxParCSRMatrixRownnz(matrix) = rownnz;
            }
         }

         if (!hypre_AuxParCSRMatrixRowLength(matrix))
         {
            hypre_AuxParCSRMatrixRowLength(matrix) = hypre_CTAlloc(HYPRE_Int, local_num_rows,
                                                                   HYPRE_MEMORY_HOST);
         }

         if (row_space)
         {
            if (local_num_rownnz != local_num_rows)
            {
               for (i = 0; i < local_num_rownnz; i++)
               {
                  ii = rownnz[i];
                  aux_j[ii] = hypre_CTAlloc(HYPRE_BigInt, row_space[ii], HYPRE_MEMORY_HOST);
                  aux_data[ii] = hypre_CTAlloc(HYPRE_Complex, row_space[ii], HYPRE_MEMORY_HOST);
               }
            }
            else
            {
               for (i = 0; i < local_num_rows; i++)
               {
                  aux_j[i] = hypre_CTAlloc(HYPRE_BigInt, row_space[i], HYPRE_MEMORY_HOST);
                  aux_data[i] = hypre_CTAlloc(HYPRE_Complex, row_space[i], HYPRE_MEMORY_HOST);
               }
            }
         }
         else
         {
            row_space = hypre_CTAlloc(HYPRE_Int, local_num_rows, HYPRE_MEMORY_HOST);
            for (i = 0; i < local_num_rows; i++)
            {
               row_space[i] = 30;
               aux_j[i] = hypre_CTAlloc(HYPRE_BigInt, 30, HYPRE_MEMORY_HOST);
               aux_data[i] = hypre_CTAlloc(HYPRE_Complex, 30, HYPRE_MEMORY_HOST);
            }
            hypre_AuxParCSRMatrixRowSpace(matrix) = row_space;
         }
         hypre_AuxParCSRMatrixAuxJ(matrix) = aux_j;
         hypre_AuxParCSRMatrixAuxData(matrix) = aux_data;
      }
      else
      {
         hypre_AuxParCSRMatrixIndxDiag(matrix) = hypre_CTAlloc(HYPRE_Int, local_num_rows, HYPRE_MEMORY_HOST);
         hypre_AuxParCSRMatrixIndxOffd(matrix) = hypre_CTAlloc(HYPRE_Int, local_num_rows, HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_AuxParCSRMatrixInitialize(hypre_AuxParCSRMatrix *matrix)
{
   if (matrix)
   {
      return hypre_AuxParCSRMatrixInitialize_v2(matrix, hypre_AuxParCSRMatrixMemoryLocation(matrix));
   }

   return -2;
}
