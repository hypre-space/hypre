/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Overlapping domain decomposition computation for ParCSR matrices
 *
 *****************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * Create and initialize an overlap data structure.
 *--------------------------------------------------------------------------*/

hypre_OverlapData*
hypre_OverlapDataCreate(void)
{
   hypre_OverlapData *overlap_data;

   overlap_data = hypre_CTAlloc(hypre_OverlapData, 1, HYPRE_MEMORY_HOST);

   hypre_OverlapDataOverlapOrder(overlap_data)       = 0;
   hypre_OverlapDataNumLocalRows(overlap_data)       = 0;
   hypre_OverlapDataFirstRowIndex(overlap_data)      = 0;
   hypre_OverlapDataLastRowIndex(overlap_data)       = -1;
   hypre_OverlapDataNumExtendedRows(overlap_data)    = 0;
   hypre_OverlapDataNumOverlapRows(overlap_data)     = 0;
   hypre_OverlapDataExtendedRowIndices(overlap_data) = NULL;
   hypre_OverlapDataGlobalToExtended(overlap_data)   = NULL;
   hypre_OverlapDataExtendedToGlobal(overlap_data)   = NULL;
   hypre_OverlapDataRowIsOwned(overlap_data)         = NULL;
   hypre_OverlapDataOverlapCommPkg(overlap_data)     = NULL;
   hypre_OverlapDataExternalRows(overlap_data)       = NULL;
   hypre_OverlapDataExternalRowMap(overlap_data)     = NULL;

   return overlap_data;
}

/*--------------------------------------------------------------------------
 * Destroy an overlap data structure and free all associated memory.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_OverlapDataDestroy(hypre_OverlapData *overlap_data)
{
   if (overlap_data)
   {
      hypre_TFree(hypre_OverlapDataExtendedRowIndices(overlap_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_OverlapDataGlobalToExtended(overlap_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_OverlapDataExtendedToGlobal(overlap_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_OverlapDataRowIsOwned(overlap_data), HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_OverlapDataExternalRowMap(overlap_data), HYPRE_MEMORY_HOST);

      if (hypre_OverlapDataOverlapCommPkg(overlap_data))
      {
         hypre_MatvecCommPkgDestroy(hypre_OverlapDataOverlapCommPkg(overlap_data));
      }

      if (hypre_OverlapDataExternalRows(overlap_data))
      {
         hypre_CSRMatrixDestroy(hypre_OverlapDataExternalRows(overlap_data));
      }

      hypre_TFree(overlap_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Check if a sorted BigInt array contains a value using binary search.
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_BigIntArrayContains(HYPRE_BigInt *array, HYPRE_Int size, HYPRE_BigInt value)
{
   HYPRE_Int left = 0;
   HYPRE_Int right = size - 1;

   while (left <= right)
   {
      HYPRE_Int mid = (left + right) / 2;
      if (array[mid] == value)
      {
         return 1;
      }
      else if (array[mid] < value)
      {
         left = mid + 1;
      }
      else
      {
         right = mid - 1;
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Compute the union of two sorted BigInt arrays, returning a new sorted array.
 *--------------------------------------------------------------------------*/

static HYPRE_BigInt*
hypre_BigIntArrayUnion(HYPRE_BigInt *arr1, HYPRE_Int size1,
                       HYPRE_BigInt *arr2, HYPRE_Int size2,
                       HYPRE_Int *result_size)
{
   HYPRE_Int      alloc_size = size1 + size2;
   HYPRE_BigInt  *result;
   HYPRE_Int      i = 0, j = 0, k = 0;

   if (alloc_size == 0)
   {
      *result_size = 0;
      return NULL;
   }

   result = hypre_TAlloc(HYPRE_BigInt, alloc_size, HYPRE_MEMORY_HOST);

   while (i < size1 && j < size2)
   {
      if (arr1[i] < arr2[j])
      {
         result[k++] = arr1[i++];
      }
      else if (arr1[i] > arr2[j])
      {
         result[k++] = arr2[j++];
      }
      else
      {
         result[k++] = arr1[i++];
         j++;
      }
   }

   while (i < size1)
   {
      result[k++] = arr1[i++];
   }

   while (j < size2)
   {
      result[k++] = arr2[j++];
   }

   *result_size = k;

   return result;
}

/*--------------------------------------------------------------------------
 * Compute the overlapping domain decomposition for a ParCSR matrix.
 *
 * Given a matrix A and overlap order delta, this function computes
 * the extended set of row indices that includes all rows within
 * delta hops of the locally owned rows.
 *
 * Arguments:
 *   A            - Input ParCSR matrix
 *   overlap_order - Number of overlap levels (delta >= 0)
 *   overlap_data_ptr - Output overlap data structure
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixComputeOverlap(hypre_ParCSRMatrix  *A,
                                 HYPRE_Int            overlap_order,
                                 hypre_OverlapData  **overlap_data_ptr)
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int            num_procs, my_id;
   hypre_OverlapData   *overlap_data;

   /* Matrix data */
   hypre_CSRMatrix     *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix     *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int           *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int           *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Int           *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int           *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_BigInt        *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_Int            num_rows = hypre_CSRMatrixNumRows(diag);
   HYPRE_Int            num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_BigInt         first_row = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt         global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);

   /* Working sets */
   HYPRE_BigInt        *current_set = NULL;
   HYPRE_Int            current_size = 0;
   HYPRE_BigInt        *new_set = NULL;
   HYPRE_Int            new_size = 0;
   HYPRE_BigInt        *external_indices = NULL;
   HYPRE_Int            num_external = 0;

   HYPRE_Int            level, i, j, jj;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Create overlap data structure */
   overlap_data = hypre_OverlapDataCreate();
   hypre_OverlapDataOverlapOrder(overlap_data)  = overlap_order;
   hypre_OverlapDataNumLocalRows(overlap_data)  = num_rows;
   hypre_OverlapDataFirstRowIndex(overlap_data) = first_row;
   hypre_OverlapDataLastRowIndex(overlap_data)  = first_row + num_rows - 1;

   /* Initialize current set with local rows */
   current_size = num_rows;
   if (num_rows > 0)
   {
      current_set = hypre_TAlloc(HYPRE_BigInt, num_rows, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_rows; i++)
      {
         current_set[i] = first_row + (HYPRE_BigInt) i;
      }
   }

   /* If overlap_order is 0, we're done - just local rows */
   if (overlap_order == 0 || num_procs == 1)
   {
      hypre_OverlapDataNumExtendedRows(overlap_data) = num_rows;
      hypre_OverlapDataNumOverlapRows(overlap_data) = 0;
      hypre_OverlapDataExtendedRowIndices(overlap_data) = current_set;

      /* Build ownership array - all owned */
      if (num_rows > 0)
      {
         hypre_OverlapDataRowIsOwned(overlap_data) = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_rows; i++)
         {
            hypre_OverlapDataRowIsOwned(overlap_data)[i] = 1;
         }
      }

      *overlap_data_ptr = overlap_data;
      return hypre_error_flag;
   }

   /* Iteratively expand the set for each level of overlap */
   for (level = 0; level < overlap_order; level++)
   {
      /* Find all neighbors of current set that are not already in current set */
      /* First, collect potential new external indices */
      HYPRE_BigInt *potential_external = NULL;
      HYPRE_Int     num_potential = 0;
      HYPRE_Int     potential_alloc = 0;

      /* Estimate allocation */
      potential_alloc = num_cols_offd;
      if (potential_alloc > 0)
      {
         potential_external = hypre_TAlloc(HYPRE_BigInt, potential_alloc, HYPRE_MEMORY_HOST);
      }

      /* For each row in current set, find its off-processor neighbors */
      for (i = 0; i < current_size; i++)
      {
         HYPRE_BigInt global_row = current_set[i];

         /* Check if this row is local */
         if (global_row >= first_row && global_row < first_row + num_rows)
         {
            HYPRE_Int local_row = (HYPRE_Int)(global_row - first_row);

            /* Check off-diagonal neighbors */
            for (jj = offd_i[local_row]; jj < offd_i[local_row + 1]; jj++)
            {
               HYPRE_Int col = offd_j[jj];
               HYPRE_BigInt global_col = col_map_offd[col];

               /* Check if already in current set */
               if (!hypre_BigIntArrayContains(current_set, current_size, global_col))
               {
                  /* Check if already in potential list */
                  HYPRE_Int found = 0;
                  for (j = 0; j < num_potential; j++)
                  {
                     if (potential_external[j] == global_col)
                     {
                        found = 1;
                        break;
                     }
                  }

                  if (!found)
                  {
                     /* Add to potential list */
                     if (num_potential >= potential_alloc)
                     {
                        potential_alloc = hypre_max(2 * potential_alloc, num_potential + 100);
                        potential_external = hypre_TReAlloc(potential_external, HYPRE_BigInt,
                                                            potential_alloc, HYPRE_MEMORY_HOST);
                     }
                     potential_external[num_potential++] = global_col;
                  }
               }
            }
         }
      }

      /* Sort potential external indices */
      if (num_potential > 1)
      {
         hypre_BigQsort0(potential_external, 0, num_potential - 1);
      }

      /* At this point, for multi-processor case with overlap > 1,
       * we need to communicate to find rows that are neighbors of external rows.
       * For level > 0, we need to get the sparsity pattern of external rows
       * to find their neighbors. */

      if (level > 0)
      {
         hypre_ParCSRCommPkg *ext_comm_pkg = NULL;
         hypre_CSRMatrix     *A_ext = NULL;
         void                *request = NULL;

         /* Build comm pkg for external indices (works with num_external = 0) */
         hypre_ParCSRFindExtendCommPkg(comm, global_num_rows, first_row, num_rows,
                                       hypre_ParCSRMatrixRowStarts(A),
                                       hypre_ParCSRMatrixAssumedPartition(A),
                                       num_external, external_indices, &ext_comm_pkg);

         /* Fetch graph of external connections A_ext (pattern only, no data needed) */
         /* All processes must call this, even with num_external = 0 */
         hypre_ParcsrGetExternalRowsInit(A, num_external, external_indices,
                                         ext_comm_pkg, 0, &request);
         A_ext = hypre_ParcsrGetExternalRowsWait(request);

         /* Find neighbors of external rows (only if this process received rows) */
         if (A_ext)
         {
            HYPRE_Int     *ext_i = hypre_CSRMatrixI(A_ext);
            HYPRE_BigInt  *ext_j = hypre_CSRMatrixBigJ(A_ext);
            HYPRE_Int      ext_num_rows = hypre_CSRMatrixNumRows(A_ext);

            for (i = 0; i < ext_num_rows; i++)
            {
               for (jj = ext_i[i]; jj < ext_i[i + 1]; jj++)
               {
                  HYPRE_BigInt neighbor = ext_j[jj];

                  /* Check if already in current set or potential */
                  if (!hypre_BigIntArrayContains(current_set, current_size, neighbor))
                  {
                     HYPRE_Int found = 0;
                     for (j = 0; j < num_potential; j++)
                     {
                        if (potential_external[j] == neighbor)
                        {
                           found = 1;
                           break;
                        }
                     }

                     if (!found)
                     {
                        if (num_potential >= potential_alloc)
                        {
                           potential_alloc = hypre_max(2 * potential_alloc, num_potential + 100);
                           potential_external = hypre_TReAlloc(potential_external, HYPRE_BigInt,
                                                               potential_alloc, HYPRE_MEMORY_HOST);
                        }
                        potential_external[num_potential++] = neighbor;
                     }
                  }
               }
            }

            hypre_CSRMatrixDestroy(A_ext);
         }

         if (ext_comm_pkg)
         {
            hypre_MatvecCommPkgDestroy(ext_comm_pkg);
         }

         /* Re-sort after adding more */
         if (num_potential > 1)
         {
            hypre_BigQsort0(potential_external, 0, num_potential - 1);

            /* Remove duplicates */
            HYPRE_Int unique_count = 1;
            for (i = 1; i < num_potential; i++)
            {
               if (potential_external[i] != potential_external[unique_count - 1])
               {
                  potential_external[unique_count++] = potential_external[i];
               }
            }
            num_potential = unique_count;
         }
      }

      /* Merge current set with potential external to get new set */
      new_set = hypre_BigIntArrayUnion(current_set, current_size,
                                       potential_external, num_potential,
                                       &new_size);

      /* Update external indices for next level */
      hypre_TFree(external_indices, HYPRE_MEMORY_HOST);
      num_external = num_potential;
      external_indices = potential_external;

      /* Replace current set */
      hypre_TFree(current_set, HYPRE_MEMORY_HOST);
      current_set = new_set;
      current_size = new_size;
   }

   /* Store final extended set */
   hypre_OverlapDataNumExtendedRows(overlap_data) = current_size;
   hypre_OverlapDataNumOverlapRows(overlap_data) = current_size - num_rows;
   hypre_OverlapDataExtendedRowIndices(overlap_data) = current_set;

   /* Build ownership array */
   if (current_size > 0)
   {
      hypre_OverlapDataRowIsOwned(overlap_data) = hypre_TAlloc(HYPRE_Int, current_size,
                                                               HYPRE_MEMORY_HOST);
      for (i = 0; i < current_size; i++)
      {
         HYPRE_BigInt row = current_set[i];
         if (row >= first_row && row < first_row + num_rows)
         {
            hypre_OverlapDataRowIsOwned(overlap_data)[i] = 1;
         }
         else
         {
            hypre_OverlapDataRowIsOwned(overlap_data)[i] = 0;
         }
      }
   }

   /* Build communication package for overlap if needed */
   if (current_size > num_rows && num_procs > 1)
   {
      /* Collect external row indices */
      HYPRE_BigInt *ext_indices = hypre_TAlloc(HYPRE_BigInt, current_size - num_rows, HYPRE_MEMORY_HOST);
      HYPRE_Int ext_count = 0;

      for (i = 0; i < current_size; i++)
      {
         if (!hypre_OverlapDataRowIsOwned(overlap_data)[i])
         {
            ext_indices[ext_count++] = current_set[i];
         }
      }

      /* Build comm pkg */
      hypre_ParCSRFindExtendCommPkg(comm, global_num_rows, first_row, num_rows,
                                    hypre_ParCSRMatrixRowStarts(A),
                                    hypre_ParCSRMatrixAssumedPartition(A),
                                    ext_count, ext_indices,
                                    &hypre_OverlapDataOverlapCommPkg(overlap_data));

      /* Store external row map */
      hypre_OverlapDataExternalRowMap(overlap_data) = ext_indices;
   }

   /* Clean up */
   hypre_TFree(external_indices, HYPRE_MEMORY_HOST);

   *overlap_data_ptr = overlap_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Fetch the actual row data for overlap rows from neighboring processors.
 *
 * Arguments:
 *   A            - Input ParCSR matrix
 *   overlap_data - Overlap data with communication package
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGetOverlapRows(hypre_ParCSRMatrix *A,
                                 hypre_OverlapData  *overlap_data)
{
   HYPRE_Int            num_external;
   HYPRE_BigInt        *ext_indices;
   hypre_ParCSRCommPkg *comm_pkg;
   void                *request = NULL;

   if (!overlap_data)
   {
      return hypre_error_flag;
   }

   num_external = hypre_OverlapDataNumOverlapRows(overlap_data);
   ext_indices = hypre_OverlapDataExternalRowMap(overlap_data);
   comm_pkg = hypre_OverlapDataOverlapCommPkg(overlap_data);

   /* If no external rows, nothing to do */
   if (num_external == 0 || !comm_pkg)
   {
      return hypre_error_flag;
   }

   /* Destroy existing external rows if any */
   if (hypre_OverlapDataExternalRows(overlap_data))
   {
      hypre_CSRMatrixDestroy(hypre_OverlapDataExternalRows(overlap_data));
      hypre_OverlapDataExternalRows(overlap_data) = NULL;
   }

   /* Fetch external rows (with data) */
   hypre_ParcsrGetExternalRowsInit(A, num_external, ext_indices, comm_pkg, 1, &request);
   hypre_OverlapDataExternalRows(overlap_data) = hypre_ParcsrGetExternalRowsWait(request);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Extract the local submatrix including overlap rows.
 * Returns a Square CSR matrix containing all rows in the extended domain,
 * with column indices restricted to only those in the extended domain.
 * Entries referencing columns outside the extended domain are dropped.
 *
 * Arguments:
 *   A            - Input ParCSR matrix
 *   overlap_data - Overlap data with external rows
 *   A_local_ptr  - Output CSR matrix for local extended domain (square)
 *   col_map_ptr  - Output array mapping local col index to global col index
 *   num_cols_local_ptr - Output number of columns in local matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixExtractLocalOverlap(hypre_ParCSRMatrix  *A,
                                      hypre_OverlapData   *overlap_data,
                                      hypre_CSRMatrix    **A_local_ptr,
                                      HYPRE_BigInt       **col_map_ptr,
                                      HYPRE_Int           *num_cols_local_ptr)
{
   hypre_CSRMatrix     *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix     *offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int           *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int           *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Complex       *diag_data = hypre_CSRMatrixData(diag);
   HYPRE_Int           *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int           *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Complex       *offd_data = hypre_CSRMatrixData(offd);
   HYPRE_BigInt        *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_BigInt         first_row = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt         first_col = hypre_ParCSRMatrixFirstColDiag(A);

   hypre_CSRMatrix     *ext_rows = hypre_OverlapDataExternalRows(overlap_data);
   HYPRE_Int           *ext_i = NULL;
   HYPRE_BigInt        *ext_j = NULL;
   HYPRE_Complex       *ext_data = NULL;
   HYPRE_Int            num_ext_rows = 0;

   HYPRE_BigInt        *extended_rows = hypre_OverlapDataExtendedRowIndices(overlap_data);
   HYPRE_Int            num_extended = hypre_OverlapDataNumExtendedRows(overlap_data);
   HYPRE_Int           *row_is_owned = hypre_OverlapDataRowIsOwned(overlap_data);

   hypre_CSRMatrix     *A_local;
   HYPRE_Int           *A_local_i;
   HYPRE_Int           *A_local_j;
   HYPRE_Complex       *A_local_data;
   HYPRE_Int            A_local_nnz;

   HYPRE_BigInt        *col_map = NULL;
   HYPRE_Int            num_cols_local;

   HYPRE_Int            i, jj, k;
   HYPRE_Int            ext_row_counter = 0;

   /* For a square matrix, the column map IS the extended rows (which is already sorted) */
   num_cols_local = num_extended;
   col_map = hypre_TAlloc(HYPRE_BigInt, num_cols_local, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_local; i++)
   {
      col_map[i] = extended_rows[i];
   }

   /* Get external rows data */
   if (ext_rows)
   {
      ext_i = hypre_CSRMatrixI(ext_rows);
      ext_j = hypre_CSRMatrixBigJ(ext_rows);
      ext_data = hypre_CSRMatrixData(ext_rows);
      num_ext_rows = hypre_CSRMatrixNumRows(ext_rows);
   }

   /* First pass: count nonzeros (only those with columns in extended domain) */
   A_local_nnz = 0;
   ext_row_counter = 0;

   for (i = 0; i < num_extended; i++)
   {
      HYPRE_BigInt global_row = extended_rows[i];

      if (row_is_owned[i])
      {
         /* Local row */
         HYPRE_Int local_row = (HYPRE_Int)(global_row - first_row);

         /* Count diagonal entries that are in extended domain */
         for (jj = diag_i[local_row]; jj < diag_i[local_row + 1]; jj++)
         {
            HYPRE_BigInt global_col = first_col + (HYPRE_BigInt) diag_j[jj];
            /* Check if column is in extended domain */
            if (hypre_BigBinarySearch(col_map, global_col, num_cols_local) >= 0)
            {
               A_local_nnz++;
            }
         }

         /* Count off-diagonal entries that are in extended domain */
         for (jj = offd_i[local_row]; jj < offd_i[local_row + 1]; jj++)
         {
            HYPRE_BigInt global_col = col_map_offd[offd_j[jj]];
            if (hypre_BigBinarySearch(col_map, global_col, num_cols_local) >= 0)
            {
               A_local_nnz++;
            }
         }
      }
      else
      {
         /* External row */
         if (ext_rows && ext_row_counter < num_ext_rows)
         {
            for (jj = ext_i[ext_row_counter]; jj < ext_i[ext_row_counter + 1]; jj++)
            {
               HYPRE_BigInt global_col = ext_j[jj];
               if (hypre_BigBinarySearch(col_map, global_col, num_cols_local) >= 0)
               {
                  A_local_nnz++;
               }
            }
            ext_row_counter++;
         }
      }
   }

   /* Create local CSR matrix (square: num_extended x num_extended) */
   A_local = hypre_CSRMatrixCreate(num_extended, num_cols_local, A_local_nnz);
   hypre_CSRMatrixInitialize(A_local);

   A_local_i = hypre_CSRMatrixI(A_local);
   A_local_j = hypre_CSRMatrixJ(A_local);
   A_local_data = hypre_CSRMatrixData(A_local);

   /* Second pass: fill the matrix */
   k = 0;
   ext_row_counter = 0;
   A_local_i[0] = 0;

   for (i = 0; i < num_extended; i++)
   {
      HYPRE_BigInt global_row = extended_rows[i];

      if (row_is_owned[i])
      {
         /* Local row */
         HYPRE_Int local_row = (HYPRE_Int)(global_row - first_row);

         /* Copy diagonal entries that are in extended domain */
         for (jj = diag_i[local_row]; jj < diag_i[local_row + 1]; jj++)
         {
            HYPRE_BigInt global_col = first_col + (HYPRE_BigInt) diag_j[jj];

            /* Find local column index using binary search */
            HYPRE_Int local_col = hypre_BigBinarySearch(col_map, global_col, num_cols_local);
            if (local_col >= 0)
            {
               A_local_j[k] = local_col;
               A_local_data[k] = diag_data[jj];
               k++;
            }
         }

         /* Copy off-diagonal entries that are in extended domain */
         for (jj = offd_i[local_row]; jj < offd_i[local_row + 1]; jj++)
         {
            HYPRE_BigInt global_col = col_map_offd[offd_j[jj]];

            HYPRE_Int local_col = hypre_BigBinarySearch(col_map, global_col, num_cols_local);
            if (local_col >= 0)
            {
               A_local_j[k] = local_col;
               A_local_data[k] = offd_data[jj];
               k++;
            }
         }
      }
      else
      {
         /* External row */
         if (ext_rows && ext_row_counter < num_ext_rows)
         {
            for (jj = ext_i[ext_row_counter]; jj < ext_i[ext_row_counter + 1]; jj++)
            {
               HYPRE_BigInt global_col = ext_j[jj];

               HYPRE_Int local_col = hypre_BigBinarySearch(col_map, global_col, num_cols_local);
               if (local_col >= 0)
               {
                  A_local_j[k] = local_col;
                  A_local_data[k] = ext_data[jj];
                  k++;
               }
            }
            ext_row_counter++;
         }
      }

      A_local_i[i + 1] = k;
   }

   /* Sort each row by column index */
   for (i = 0; i < num_extended; i++)
   {
      HYPRE_Int row_start = A_local_i[i];
      HYPRE_Int row_nnz = A_local_i[i + 1] - row_start;
      if (row_nnz > 1)
      {
         hypre_qsort2(&A_local_j[row_start], &A_local_data[row_start], 0, row_nnz - 1);
      }
   }

   /* Reorder the matrix to put the diagonal first */
   hypre_CSRMatrixReorder(A_local);

   /* Return results */
   *A_local_ptr = A_local;
   *col_map_ptr = col_map;
   *num_cols_local_ptr = num_cols_local;

   return hypre_error_flag;
}
