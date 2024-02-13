/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_ParCSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

#include "../seq_mv/HYPRE_seq_mv.h"
#include "../seq_mv/csr_matrix.h"

/* In addition to publically accessible interface in HYPRE_mv.h, the
   implementation in this file uses accessor macros into the sequential matrix
   structure, and so includes the .h that defines that structure. Should those
   accessor functions become proper functions at some later date, this will not
   be necessary. AJC 4/99 */

HYPRE_Int hypre_FillResponseParToCSRMatrix(void*, HYPRE_Int, HYPRE_Int, void*, MPI_Comm, void**,
                                           HYPRE_Int*);

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

/* If create is called and row_starts and col_starts are NOT null, then it is
   assumed that they are of length 2 containing the start row of the calling
   processor followed by the start row of the next processor - AHB 6/05 */

hypre_ParCSRMatrix*
hypre_ParCSRMatrixCreate( MPI_Comm      comm,
                          HYPRE_BigInt  global_num_rows,
                          HYPRE_BigInt  global_num_cols,
                          HYPRE_BigInt *row_starts_in,
                          HYPRE_BigInt *col_starts_in,
                          HYPRE_Int     num_cols_offd,
                          HYPRE_Int     num_nonzeros_diag,
                          HYPRE_Int     num_nonzeros_offd )
{
   hypre_ParCSRMatrix  *matrix;
   HYPRE_Int            num_procs, my_id;
   HYPRE_Int            local_num_rows;
   HYPRE_Int            local_num_cols;
   HYPRE_BigInt         row_starts[2];
   HYPRE_BigInt         col_starts[2];
   HYPRE_BigInt         first_row_index, first_col_diag;

   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   if (!row_starts_in)
   {
      hypre_GenerateLocalPartitioning(global_num_rows, num_procs, my_id,
                                      row_starts);
   }
   else
   {
      row_starts[0] = row_starts_in[0];
      row_starts[1] = row_starts_in[1];
   }

   if (!col_starts_in)
   {
      hypre_GenerateLocalPartitioning(global_num_cols, num_procs, my_id,
                                      col_starts);
   }
   else
   {
      col_starts[0] = col_starts_in[0];
      col_starts[1] = col_starts_in[1];
   }

   /* row_starts[0] is start of local rows.
      row_starts[1] is start of next processor's rows */
   first_row_index = row_starts[0];
   local_num_rows  = row_starts[1] - first_row_index;
   first_col_diag  = col_starts[0];
   local_num_cols  = col_starts[1] - first_col_diag;

   hypre_ParCSRMatrixComm(matrix) = comm;
   hypre_ParCSRMatrixDiag(matrix) =
      hypre_CSRMatrixCreate(local_num_rows, local_num_cols, num_nonzeros_diag);
   hypre_ParCSRMatrixOffd(matrix) =
      hypre_CSRMatrixCreate(local_num_rows, num_cols_offd, num_nonzeros_offd);
   hypre_ParCSRMatrixDiagT(matrix) = NULL;
   hypre_ParCSRMatrixOffdT(matrix) = NULL; // JSP: transposed matrices are optional
   hypre_ParCSRMatrixGlobalNumRows(matrix)   = global_num_rows;
   hypre_ParCSRMatrixGlobalNumCols(matrix)   = global_num_cols;
   hypre_ParCSRMatrixGlobalNumRownnz(matrix) = global_num_rows;
   hypre_ParCSRMatrixNumNonzeros(matrix)     = -1;   /* Uninitialized */
   hypre_ParCSRMatrixDNumNonzeros(matrix)    = -1.0; /* Uninitialized */
   hypre_ParCSRMatrixFirstRowIndex(matrix)   = first_row_index;
   hypre_ParCSRMatrixFirstColDiag(matrix)    = first_col_diag;
   hypre_ParCSRMatrixLastRowIndex(matrix) = first_row_index + local_num_rows - 1;
   hypre_ParCSRMatrixLastColDiag(matrix)  = first_col_diag + local_num_cols - 1;

   hypre_ParCSRMatrixRowStarts(matrix)[0] = row_starts[0];
   hypre_ParCSRMatrixRowStarts(matrix)[1] = row_starts[1];
   hypre_ParCSRMatrixColStarts(matrix)[0] = col_starts[0];
   hypre_ParCSRMatrixColStarts(matrix)[1] = col_starts[1];

   hypre_ParCSRMatrixColMapOffd(matrix)       = NULL;
   hypre_ParCSRMatrixDeviceColMapOffd(matrix) = NULL;
   hypre_ParCSRMatrixProcOrdering(matrix)     = NULL;

   hypre_ParCSRMatrixAssumedPartition(matrix) = NULL;
   hypre_ParCSRMatrixOwnsAssumedPartition(matrix) = 1;

   hypre_ParCSRMatrixCommPkg(matrix)  = NULL;
   hypre_ParCSRMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix)     = 1;
   hypre_ParCSRMatrixRowindices(matrix)   = NULL;
   hypre_ParCSRMatrixRowvalues(matrix)    = NULL;
   hypre_ParCSRMatrixGetrowactive(matrix) = 0;

   matrix->bdiaginv = NULL;
   matrix->bdiaginv_comm_pkg = NULL;
   matrix->bdiag_size = -1;

#if defined(HYPRE_USING_GPU)
   hypre_ParCSRMatrixSocDiagJ(matrix) = NULL;
   hypre_ParCSRMatrixSocOffdJ(matrix) = NULL;
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixDestroy( hypre_ParCSRMatrix *matrix )
{
   if (matrix)
   {
      HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(matrix);

      if ( hypre_ParCSRMatrixOwnsData(matrix) )
      {
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix));
         hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix));

         if ( hypre_ParCSRMatrixDiagT(matrix) )
         {
            hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiagT(matrix));
         }

         if ( hypre_ParCSRMatrixOffdT(matrix) )
         {
            hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffdT(matrix));
         }

         if (hypre_ParCSRMatrixColMapOffd(matrix))
         {
            hypre_TFree(hypre_ParCSRMatrixColMapOffd(matrix), HYPRE_MEMORY_HOST);
         }

         if (hypre_ParCSRMatrixDeviceColMapOffd(matrix))
         {
            hypre_TFree(hypre_ParCSRMatrixDeviceColMapOffd(matrix), HYPRE_MEMORY_DEVICE);
         }

         if (hypre_ParCSRMatrixCommPkg(matrix))
         {
            hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkg(matrix));
         }

         if (hypre_ParCSRMatrixCommPkgT(matrix))
         {
            hypre_MatvecCommPkgDestroy(hypre_ParCSRMatrixCommPkgT(matrix));
         }
      }

      /* RL: this is actually not correct since the memory_location may have been changed after allocation
       * put them in containers TODO */
      hypre_TFree(hypre_ParCSRMatrixRowindices(matrix), memory_location);
      hypre_TFree(hypre_ParCSRMatrixRowvalues(matrix), memory_location);

      if ( hypre_ParCSRMatrixAssumedPartition(matrix) &&
           hypre_ParCSRMatrixOwnsAssumedPartition(matrix) )
      {
         hypre_AssumedPartitionDestroy(hypre_ParCSRMatrixAssumedPartition(matrix));
      }

      if ( hypre_ParCSRMatrixProcOrdering(matrix) )
      {
         hypre_TFree(hypre_ParCSRMatrixProcOrdering(matrix), HYPRE_MEMORY_HOST);
      }

      hypre_TFree(matrix->bdiaginv, HYPRE_MEMORY_HOST);
      if (matrix->bdiaginv_comm_pkg)
      {
         hypre_MatvecCommPkgDestroy(matrix->bdiaginv_comm_pkg);
      }

#if defined(HYPRE_USING_GPU)
      hypre_TFree(hypre_ParCSRMatrixSocDiagJ(matrix), HYPRE_MEMORY_DEVICE);
      hypre_TFree(hypre_ParCSRMatrixSocOffdJ(matrix), HYPRE_MEMORY_DEVICE);
#endif

      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixInitialize_v2( hypre_ParCSRMatrix   *matrix,
                                 HYPRE_MemoryLocation  memory_location )
{
   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_CSRMatrixInitialize_v2(hypre_ParCSRMatrixDiag(matrix), 0, memory_location);
   hypre_CSRMatrixInitialize_v2(hypre_ParCSRMatrixOffd(matrix), 0, memory_location);

   hypre_ParCSRMatrixColMapOffd(matrix) =
      hypre_CTAlloc(HYPRE_BigInt, hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matrix)),
                    HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixInitialize( hypre_ParCSRMatrix *matrix )
{
   return hypre_ParCSRMatrixInitialize_v2(matrix, hypre_ParCSRMatrixMemoryLocation(matrix));
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixClone
 * Creates and returns a new copy S of the argument A
 * The following variables are not copied because they will be constructed
 * later if needed: CommPkg, CommPkgT, rowindices, rowvalues
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixClone_v2(hypre_ParCSRMatrix   *A,
                           HYPRE_Int             copy_data,
                           HYPRE_MemoryLocation  memory_location)
{
   hypre_ParCSRMatrix *S;

   hypre_GpuProfilingPushRange("hypre_ParCSRMatrixClone");

   S = hypre_ParCSRMatrixCreate( hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumRows(A),
                                 hypre_ParCSRMatrixGlobalNumCols(A),
                                 hypre_ParCSRMatrixRowStarts(A),
                                 hypre_ParCSRMatrixColStarts(A),
                                 hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A)),
                                 hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A)),
                                 hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A)) );

   hypre_ParCSRMatrixNumNonzeros(S)  = hypre_ParCSRMatrixNumNonzeros(A);
   hypre_ParCSRMatrixDNumNonzeros(S) = hypre_ParCSRMatrixNumNonzeros(A);

   hypre_ParCSRMatrixInitialize_v2(S, memory_location);

   hypre_ParCSRMatrixCopy(A, S, copy_data);

   hypre_GpuProfilingPopRange();

   return S;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixClone
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixClone(hypre_ParCSRMatrix *A, HYPRE_Int copy_data)
{
   return hypre_ParCSRMatrixClone_v2(A, copy_data, hypre_ParCSRMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMigrate(hypre_ParCSRMatrix   *A,
                          HYPRE_MemoryLocation  memory_location)
{
   if (!A)
   {
      return hypre_error_flag;
   }

   HYPRE_MemoryLocation old_memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrixMigrate(hypre_ParCSRMatrixDiag(A), memory_location);
   hypre_CSRMatrixMigrate(hypre_ParCSRMatrixOffd(A), memory_location);

   /* Free buffers */
   if ( hypre_GetActualMemLocation(memory_location) !=
        hypre_GetActualMemLocation(old_memory_location) )
   {
      hypre_TFree(hypre_ParCSRMatrixRowindices(A), old_memory_location);
      hypre_TFree(hypre_ParCSRMatrixRowvalues(A), old_memory_location);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetNumNonzeros_core
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetNumNonzeros_core( hypre_ParCSRMatrix *matrix,
                                       const char         *format )
{
   MPI_Comm comm;
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   comm = hypre_ParCSRMatrixComm(matrix);
   diag = hypre_ParCSRMatrixDiag(matrix);
   offd = hypre_ParCSRMatrixOffd(matrix);

#if defined(HYPRE_DEBUG)
   hypre_CSRMatrixCheckSetNumNonzeros(diag);
   hypre_CSRMatrixCheckSetNumNonzeros(offd);
#endif

   if (format[0] == 'I')
   {
      HYPRE_BigInt total_num_nonzeros;
      HYPRE_BigInt local_num_nonzeros;
      local_num_nonzeros = (HYPRE_BigInt) ( hypre_CSRMatrixNumNonzeros(diag) +
                                            hypre_CSRMatrixNumNonzeros(offd) );

      hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, HYPRE_MPI_BIG_INT,
                          hypre_MPI_SUM, comm);

      hypre_ParCSRMatrixNumNonzeros(matrix) = total_num_nonzeros;
   }
   else if (format[0] == 'D')
   {
      HYPRE_Real total_num_nonzeros;
      HYPRE_Real local_num_nonzeros;
      local_num_nonzeros = (HYPRE_Real) ( hypre_CSRMatrixNumNonzeros(diag) +
                                          hypre_CSRMatrixNumNonzeros(offd) );

      hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1,
                          HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

      hypre_ParCSRMatrixDNumNonzeros(matrix) = total_num_nonzeros;
   }
   else
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetNumNonzeros
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetNumNonzeros( hypre_ParCSRMatrix *matrix )
{
   return hypre_ParCSRMatrixSetNumNonzeros_core(matrix, "Int");
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetDNumNonzeros
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetDNumNonzeros( hypre_ParCSRMatrix *matrix )
{
   return hypre_ParCSRMatrixSetNumNonzeros_core(matrix, "Double");
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetNumRownnz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetNumRownnz( hypre_ParCSRMatrix *matrix )
{
   MPI_Comm          comm = hypre_ParCSRMatrixComm(matrix);
   hypre_CSRMatrix  *diag = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix  *offd = hypre_ParCSRMatrixOffd(matrix);
   HYPRE_Int        *rownnz_diag = hypre_CSRMatrixRownnz(diag);
   HYPRE_Int        *rownnz_offd = hypre_CSRMatrixRownnz(offd);
   HYPRE_Int         num_rownnz_diag = hypre_CSRMatrixNumRownnz(diag);
   HYPRE_Int         num_rownnz_offd = hypre_CSRMatrixNumRownnz(offd);

   HYPRE_BigInt      local_num_rownnz;
   HYPRE_BigInt      global_num_rownnz;
   HYPRE_Int         i, j;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   local_num_rownnz = i = j = 0;
   while (i < num_rownnz_diag && j < num_rownnz_offd)
   {
      local_num_rownnz++;
      if (rownnz_diag[i] < rownnz_offd[j])
      {
         i++;
      }
      else
      {
         j++;
      }
   }

   local_num_rownnz += (HYPRE_BigInt) ((num_rownnz_diag - i) + (num_rownnz_offd - j));

   hypre_MPI_Allreduce(&local_num_rownnz, &global_num_rownnz, 1,
                       HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   hypre_ParCSRMatrixGlobalNumRownnz(matrix) = global_num_rownnz;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetDataOwner( hypre_ParCSRMatrix *matrix,
                                HYPRE_Int           owns_data )
{
   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParCSRMatrixOwnsData(matrix) = owns_data;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetPatternOnly
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetPatternOnly( hypre_ParCSRMatrix *matrix,
                                  HYPRE_Int           pattern_only)
{
   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (hypre_ParCSRMatrixDiag(matrix))
   {
      hypre_CSRMatrixSetPatternOnly(hypre_ParCSRMatrixDiag(matrix), pattern_only);
   }

   if (hypre_ParCSRMatrixOffd(matrix))
   {
      hypre_CSRMatrixSetPatternOnly(hypre_ParCSRMatrixOffd(matrix), pattern_only);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCreateFromDenseBlockMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixCreateFromDenseBlockMatrix(MPI_Comm                comm,
                                             HYPRE_BigInt            global_num_rows,
                                             HYPRE_BigInt            global_num_cols,
                                             HYPRE_BigInt           *row_starts,
                                             HYPRE_BigInt           *col_starts,
                                             hypre_DenseBlockMatrix *B)
{
   /* Input matrix variables */
   HYPRE_Int             num_rows_diag      = hypre_DenseBlockMatrixNumRows(B);
   HYPRE_Int             num_nonzeros_diag  = hypre_DenseBlockMatrixNumNonzeros(B);
   HYPRE_Int             num_rows_block     = hypre_DenseBlockMatrixNumRowsBlock(B);
   HYPRE_Int             num_cols_block     = hypre_DenseBlockMatrixNumColsBlock(B);
   HYPRE_Int             num_cols_offd      = 0;
   HYPRE_Int             num_nonzeros_offd  = 0;
   HYPRE_MemoryLocation  memory_location    = hypre_DenseBlockMatrixMemoryLocation(B);

   /* Output matrix variables */
   hypre_ParCSRMatrix   *A;
   hypre_CSRMatrix      *A_diag;
   hypre_CSRMatrix      *A_offd;
   HYPRE_Int            *A_diag_i;
   HYPRE_Int            *A_diag_j;

   /* Local variables */
   HYPRE_Int             i, j, ib;

   /* Create output matrix */
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, num_cols_offd,
                                num_nonzeros_diag, num_nonzeros_offd);
   A_diag = hypre_ParCSRMatrixDiag(A);
   A_offd = hypre_ParCSRMatrixOffd(A);

   /* Set memory locations */
   hypre_CSRMatrixMemoryLocation(A_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(A_offd) = memory_location;

   /* Set diag's data pointer */
   if (hypre_DenseBlockMatrixOwnsData(B))
   {
      hypre_CSRMatrixData(A_diag) = hypre_DenseBlockMatrixData(B);
   }
   else
   {
      hypre_CSRMatrixData(A_diag) = hypre_CTAlloc(HYPRE_Complex,
                                                  num_nonzeros_diag,
                                                  memory_location);
      hypre_TMemcpy(hypre_CSRMatrixData(A_diag),
                    hypre_DenseBlockMatrixData(B),
                    HYPRE_Complex,
                    num_nonzeros_diag,
                    memory_location, memory_location);
   }
   hypre_DenseBlockMatrixOwnsData(B) = 0;

   /* Set diag's row pointer and column indices */
   A_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows_diag + 1, HYPRE_MEMORY_HOST);
   A_diag_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros_diag, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, ib, j) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_rows_diag; i++)
   {
      ib = i / num_rows_block;
      A_diag_i[i] = i * num_cols_block;
      for (j = A_diag_i[i]; j < (i + 1) * num_cols_block; j++)
      {
         A_diag_j[j] = ib * num_cols_block + (j - A_diag_i[i]);
      }
   }
   A_diag_i[num_rows_diag] = num_rows_diag * num_cols_block;

   /* Migrate to dest. memory location */
   if (memory_location != HYPRE_MEMORY_HOST)
   {
      hypre_CSRMatrixI(A_diag) = hypre_TAlloc(HYPRE_Int, num_rows_diag + 1, memory_location);
      hypre_CSRMatrixJ(A_diag) = hypre_TAlloc(HYPRE_Int, num_nonzeros_diag, memory_location);

      hypre_TMemcpy(hypre_CSRMatrixI(A_diag), A_diag_i,
                    HYPRE_Int, num_rows_diag + 1,
                    memory_location, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_CSRMatrixJ(A_diag), A_diag_j,
                    HYPRE_Int, num_nonzeros_diag,
                    memory_location, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_CSRMatrixI(A_diag) = A_diag_i;
      hypre_CSRMatrixJ(A_diag) = A_diag_j;
   }

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCreateFromParVector
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixCreateFromParVector(hypre_ParVector *b,
                                      HYPRE_BigInt     global_num_rows,
                                      HYPRE_BigInt     global_num_cols,
                                      HYPRE_BigInt    *row_starts,
                                      HYPRE_BigInt    *col_starts)
{
   /* Input vector variables */
   MPI_Comm              comm            = hypre_ParVectorComm(b);
   hypre_Vector         *local_vector    = hypre_ParVectorLocalVector(b);
   HYPRE_MemoryLocation  memory_location = hypre_ParVectorMemoryLocation(b);

   /* Auxiliary variables */
   HYPRE_Int             num_rows        = (HYPRE_Int) row_starts[1] - row_starts[0];
   HYPRE_Int             num_cols        = (HYPRE_Int) col_starts[1] - col_starts[0];
   HYPRE_Int             num_nonzeros    = hypre_min(num_rows, num_cols);

   /* Output matrix variables */
   hypre_ParCSRMatrix   *A;
   hypre_CSRMatrix      *A_diag;
   hypre_CSRMatrix      *A_offd;
   HYPRE_Int            *A_diag_i;
   HYPRE_Int            *A_diag_j;

   /* Local variables */
   HYPRE_Int             i;

   /* Sanity check */
   if (hypre_ParVectorNumVectors(b) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented for multi-component vectors");
      return NULL;
   }

   /* Create output matrix */
   A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                row_starts, col_starts, 0, num_nonzeros, 0);
   A_diag = hypre_ParCSRMatrixDiag(A);
   A_offd = hypre_ParCSRMatrixOffd(A);

   /* Set memory locations */
   hypre_CSRMatrixMemoryLocation(A_diag) = memory_location;
   hypre_CSRMatrixMemoryLocation(A_offd) = memory_location;

   /* Set diag's data pointer */
   if (hypre_VectorOwnsData(local_vector))
   {
      hypre_CSRMatrixData(A_diag) = hypre_VectorData(local_vector);
      hypre_VectorOwnsData(b) = 0;
   }
   else
   {
      hypre_CSRMatrixData(A_diag) = hypre_CTAlloc(HYPRE_Complex, num_nonzeros, memory_location);
      hypre_TMemcpy(hypre_CSRMatrixData(A_diag),
                    hypre_VectorData(local_vector),
                    HYPRE_Complex, num_nonzeros,
                    memory_location, memory_location);
   }

   /* Set diag's row pointer and column indices */
   A_diag_i = hypre_CTAlloc(HYPRE_Int, num_rows + 1, HYPRE_MEMORY_HOST);
   A_diag_j = hypre_CTAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_nonzeros; i++)
   {
      A_diag_i[i] = A_diag_j[i] = i;
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
   for (i = num_nonzeros; i < num_rows + 1; i++)
   {
      A_diag_i[i] = num_nonzeros;
   }

   /* Initialize offd portion */
   hypre_CSRMatrixInitialize_v2(A_offd, 0, memory_location);

   /* Migrate to dest. memory location */
   if (memory_location != HYPRE_MEMORY_HOST)
   {
      hypre_CSRMatrixI(A_diag) = hypre_TAlloc(HYPRE_Int, num_rows + 1, memory_location);
      hypre_CSRMatrixJ(A_diag) = hypre_TAlloc(HYPRE_Int, num_nonzeros, memory_location);

      hypre_TMemcpy(hypre_CSRMatrixI(A_diag), A_diag_i,
                    HYPRE_Int, num_rows + 1,
                    memory_location, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(hypre_CSRMatrixJ(A_diag), A_diag_j,
                    HYPRE_Int, num_nonzeros,
                    memory_location, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_CSRMatrixI(A_diag) = A_diag_i;
      hypre_CSRMatrixJ(A_diag) = A_diag_j;
   }

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRMatrixRead( MPI_Comm    comm,
                        const char *file_name )
{
   hypre_ParCSRMatrix  *matrix;
   hypre_CSRMatrix     *diag;
   hypre_CSRMatrix     *offd;

   HYPRE_Int            my_id, num_procs;
   HYPRE_Int            num_cols_offd;
   HYPRE_Int            i, local_num_rows;

   HYPRE_BigInt         row_starts[2];
   HYPRE_BigInt         col_starts[2];
   HYPRE_BigInt        *col_map_offd;
   HYPRE_BigInt         row_s, row_e, col_s, col_e;
   HYPRE_BigInt         global_num_rows, global_num_cols;

   FILE                *fp;
   char                 new_file_d[HYPRE_MAX_FILE_NAME_LEN];
   char                 new_file_o[HYPRE_MAX_FILE_NAME_LEN];
   char                 new_file_info[HYPRE_MAX_FILE_NAME_LEN];

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   hypre_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   hypre_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   fp = fopen(new_file_info, "r");
   hypre_fscanf(fp, "%b", &global_num_rows);
   hypre_fscanf(fp, "%b", &global_num_cols);
   hypre_fscanf(fp, "%d", &num_cols_offd);
   /* the bgl input file should only contain the EXACT range for local processor */
   hypre_fscanf(fp, "%b %b %b %b", &row_s, &row_e, &col_s, &col_e);
   row_starts[0] = row_s;
   row_starts[1] = row_e;
   col_starts[0] = col_s;
   col_starts[1] = col_e;

   col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);

   for (i = 0; i < num_cols_offd; i++)
   {
      hypre_fscanf(fp, "%b", &col_map_offd[i]);
   }

   fclose(fp);

   diag = hypre_CSRMatrixRead(new_file_d);
   local_num_rows = hypre_CSRMatrixNumRows(diag);

   if (num_cols_offd)
   {
      offd = hypre_CSRMatrixRead(new_file_o);
   }
   else
   {
      offd = hypre_CSRMatrixCreate(local_num_rows, 0, 0);
      hypre_CSRMatrixInitialize_v2(offd, 0, HYPRE_MEMORY_HOST);
   }

   matrix = hypre_CTAlloc(hypre_ParCSRMatrix, 1, HYPRE_MEMORY_HOST);

   hypre_ParCSRMatrixComm(matrix) = comm;
   hypre_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   hypre_ParCSRMatrixFirstRowIndex(matrix) = row_s;
   hypre_ParCSRMatrixFirstColDiag(matrix) = col_s;
   hypre_ParCSRMatrixLastRowIndex(matrix) = row_e - 1;
   hypre_ParCSRMatrixLastColDiag(matrix) = col_e - 1;

   hypre_ParCSRMatrixRowStarts(matrix)[0] = row_starts[0];
   hypre_ParCSRMatrixRowStarts(matrix)[1] = row_starts[1];
   hypre_ParCSRMatrixColStarts(matrix)[0] = col_starts[0];
   hypre_ParCSRMatrixColStarts(matrix)[1] = col_starts[1];

   hypre_ParCSRMatrixCommPkg(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRMatrixOwnsData(matrix) = 1;
   hypre_ParCSRMatrixDiag(matrix) = diag;
   hypre_ParCSRMatrixOffd(matrix) = offd;
   if (num_cols_offd)
   {
      hypre_ParCSRMatrixColMapOffd(matrix) = col_map_offd;
   }
   else
   {
      hypre_ParCSRMatrixColMapOffd(matrix) = NULL;
   }

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixPrint( hypre_ParCSRMatrix *matrix,
                         const char         *file_name )
{
   MPI_Comm      comm;
   HYPRE_BigInt  global_num_rows;
   HYPRE_BigInt  global_num_cols;
   HYPRE_BigInt *col_map_offd;
   HYPRE_Int     my_id, i, num_procs;

   char          new_file_d[HYPRE_MAX_FILE_NAME_LEN];
   char          new_file_o[HYPRE_MAX_FILE_NAME_LEN];
   char          new_file_info[HYPRE_MAX_FILE_NAME_LEN];
   FILE         *fp;
   HYPRE_Int     num_cols_offd = 0;
   HYPRE_BigInt  row_s, row_e, col_s, col_e;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   comm = hypre_ParCSRMatrixComm(matrix);
   global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrix);
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix);
   col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);
   if (hypre_ParCSRMatrixOffd(matrix))
   {
      num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matrix));
   }

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   hypre_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   hypre_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   hypre_CSRMatrixPrint(hypre_ParCSRMatrixDiag(matrix), new_file_d);
   if (num_cols_offd != 0)
   {
      hypre_CSRMatrixPrint(hypre_ParCSRMatrixOffd(matrix), new_file_o);
   }

   fp = fopen(new_file_info, "w");
   hypre_fprintf(fp, "%b\n", global_num_rows);
   hypre_fprintf(fp, "%b\n", global_num_cols);
   hypre_fprintf(fp, "%d\n", num_cols_offd);
   row_s = hypre_ParCSRMatrixFirstRowIndex(matrix);
   row_e = hypre_ParCSRMatrixLastRowIndex(matrix);
   col_s =  hypre_ParCSRMatrixFirstColDiag(matrix);
   col_e =  hypre_ParCSRMatrixLastColDiag(matrix);

   /* add 1 to the ends because this is a starts partition */
   hypre_fprintf(fp, "%b %b %b %b\n", row_s, row_e + 1, col_s, col_e + 1);
   for (i = 0; i < num_cols_offd; i++)
   {
      hypre_fprintf(fp, "%b\n", col_map_offd[i]);
   }
   fclose(fp);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixPrintIJ
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixPrintIJ( const hypre_ParCSRMatrix *matrix,
                           const HYPRE_Int           base_i,
                           const HYPRE_Int           base_j,
                           const char               *filename )
{
   hypre_ParCSRMatrix  *h_matrix;

   MPI_Comm             comm;
   HYPRE_BigInt         first_row_index;
   HYPRE_BigInt         first_col_diag;
   hypre_CSRMatrix     *diag;
   hypre_CSRMatrix     *offd;
   HYPRE_BigInt        *col_map_offd;
   HYPRE_Int            num_rows;
   const HYPRE_BigInt  *row_starts;
   const HYPRE_BigInt  *col_starts;
   HYPRE_Complex       *diag_data;
   HYPRE_Int           *diag_i;
   HYPRE_Int           *diag_j;
   HYPRE_Complex       *offd_data;
   HYPRE_Int           *offd_i = NULL;
   HYPRE_Int           *offd_j;
   HYPRE_Int            myid, num_procs, i, j;
   HYPRE_BigInt         I, J;
   char                 new_filename[HYPRE_MAX_FILE_NAME_LEN];
   FILE                *file;
   HYPRE_Int            num_nonzeros_offd;
   HYPRE_BigInt         ilower, iupper, jlower, jupper;

   HYPRE_MemoryLocation memory_location =
      hypre_ParCSRMatrixMemoryLocation((hypre_ParCSRMatrix*) matrix);

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   /* Create temporary matrix on host memory if needed */
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_HOST)
   {
      h_matrix = (hypre_ParCSRMatrix *) matrix;
   }
   else
   {
      h_matrix = hypre_ParCSRMatrixClone_v2((hypre_ParCSRMatrix *) matrix, 1, HYPRE_MEMORY_HOST);
   }

   comm            = hypre_ParCSRMatrixComm(h_matrix);
   first_row_index = hypre_ParCSRMatrixFirstRowIndex(h_matrix);
   first_col_diag  = hypre_ParCSRMatrixFirstColDiag(h_matrix);
   diag            = hypre_ParCSRMatrixDiag(h_matrix);
   offd            = hypre_ParCSRMatrixOffd(h_matrix);
   col_map_offd    = hypre_ParCSRMatrixColMapOffd(h_matrix);
   num_rows        = hypre_ParCSRMatrixNumRows(h_matrix);
   row_starts      = hypre_ParCSRMatrixRowStarts(h_matrix);
   col_starts      = hypre_ParCSRMatrixColStarts(h_matrix);
   hypre_MPI_Comm_rank(comm, &myid);
   hypre_MPI_Comm_size(comm, &num_procs);

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   diag_data = hypre_CSRMatrixData(diag);
   diag_i    = hypre_CSRMatrixI(diag);
   diag_j    = hypre_CSRMatrixJ(diag);

   num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(offd);
   if (num_nonzeros_offd)
   {
      offd_data = hypre_CSRMatrixData(offd);
      offd_i    = hypre_CSRMatrixI(offd);
      offd_j    = hypre_CSRMatrixJ(offd);
   }

   ilower = row_starts[0] + (HYPRE_BigInt) base_i;
   iupper = row_starts[1] + (HYPRE_BigInt) base_i - 1;
   jlower = col_starts[0] + (HYPRE_BigInt) base_j;
   jupper = col_starts[1] + (HYPRE_BigInt) base_j - 1;

   hypre_fprintf(file, "%b %b %b %b\n", ilower, iupper, jlower, jupper);

   for (i = 0; i < num_rows; i++)
   {
      I = first_row_index + (HYPRE_BigInt)(i + base_i);

      /* print diag columns */
      for (j = diag_i[i]; j < diag_i[i + 1]; j++)
      {
         J = first_col_diag + (HYPRE_BigInt)(diag_j[j] + base_j);
         if (diag_data)
         {
#ifdef HYPRE_COMPLEX
            hypre_fprintf(file, "%b %b %.14e , %.14e\n", I, J,
                          hypre_creal(diag_data[j]), hypre_cimag(diag_data[j]));
#else
            hypre_fprintf(file, "%b %b %.14e\n", I, J, diag_data[j]);
#endif
         }
         else
         {
            hypre_fprintf(file, "%b %b\n", I, J);
         }
      }

      /* print offd columns */
      if (num_nonzeros_offd)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            J = col_map_offd[offd_j[j]] + (HYPRE_BigInt) base_j;
            if (offd_data)
            {
#ifdef HYPRE_COMPLEX
               hypre_fprintf(file, "%b %b %.14e , %.14e\n", I, J,
                             hypre_creal(offd_data[j]), hypre_cimag(offd_data[j]));
#else
               hypre_fprintf(file, "%b %b %.14e\n", I, J, offd_data[j]);
#endif
            }
            else
            {
               hypre_fprintf(file, "%b %b\n", I, J);
            }
         }
      }
   }

   fclose(file);

   /* Free temporary matrix */
   if (hypre_GetActualMemLocation(memory_location) != hypre_MEMORY_HOST)
   {
      hypre_ParCSRMatrixDestroy(h_matrix);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixPrintBinaryIJ
 *
 * Prints a ParCSRMatrix in binary format. The data from each process is
 * printed to a separate file. Metadata info about the matrix is printed in
 * the header section of every file, and it is followed by the raw data, i.e.,
 * row, column, and coefficients.
 *
 * The header section is composed by 11 entries stored in 88 bytes (8 bytes
 * each) and their meanings are:
 *
 *    0) Header version
 *    1) Number of bytes for storing an integer type (row and columns)
 *    2) Number of bytes for storing a real type (coefficients)
 *    3) Number of rows in the matrix
 *    4) Number of columns in the matrix
 *    5) Number of nonzero coefficients in the matrix
 *    6) Number of local nonzero coefficients in the current matrix block
 *    7) Global index of the first row of the current matrix block
 *    8) Global index of the last row of the current matrix block
 *    9) Global index of the first column in the diagonal matrix block
 *   10) Global index of the last column in the diagonal matrix block
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixPrintBinaryIJ( hypre_ParCSRMatrix *matrix,
                                 HYPRE_Int           base_i,
                                 HYPRE_Int           base_j,
                                 const char         *filename )
{
   MPI_Comm              comm = hypre_ParCSRMatrixComm(matrix);
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(matrix);
   hypre_ParCSRMatrix   *h_matrix;

   HYPRE_BigInt          first_row_index;
   HYPRE_BigInt          first_col_diag;
   hypre_CSRMatrix      *diag, *offd;
   HYPRE_BigInt         *col_map_offd;
   HYPRE_Int             num_rows;
   HYPRE_BigInt         *row_starts, *col_starts;

   HYPRE_Complex        *diag_data;
   HYPRE_Int            *diag_i, *diag_j;
   HYPRE_Int             diag_nnz;

   HYPRE_Complex        *offd_data;
   HYPRE_Int            *offd_i, *offd_j;
   HYPRE_Int             offd_nnz;

   /* Local buffers */
   hypre_uint32         *i32buffer = NULL;
   hypre_uint64         *i64buffer = NULL;
   hypre_float          *f32buffer = NULL;
   hypre_double         *f64buffer = NULL;

   /* Local variables */
   char                  new_filename[HYPRE_MAX_FILE_NAME_LEN];
   FILE                 *fp;
   hypre_uint64          header[11];
   size_t                count, k;
   HYPRE_Int             one = 1;
   HYPRE_Int             myid, i, j;
   HYPRE_BigInt          bigI, bigJ;
   HYPRE_BigInt          ilower, iupper, jlower, jupper;
   HYPRE_Complex         val;

   /* Exit if trying to write from big-endian machine */
   if ((*(char*)&one) == 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Support to big-endian machines is incomplete!\n");
      return hypre_error_flag;
   }

   /* MPI variables */
   hypre_MPI_Comm_rank(comm, &myid);

   /* Create temporary matrix on host memory if needed */
   h_matrix = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
              hypre_ParCSRMatrixClone_v2(matrix, 1, HYPRE_MEMORY_HOST) : matrix;

   /* Update global number of nonzeros */
   hypre_ParCSRMatrixSetDNumNonzeros(h_matrix);

   /* Matrix variables */
   first_row_index = hypre_ParCSRMatrixFirstRowIndex(h_matrix);
   first_col_diag  = hypre_ParCSRMatrixFirstColDiag(h_matrix);
   diag            = hypre_ParCSRMatrixDiag(h_matrix);
   offd            = hypre_ParCSRMatrixOffd(h_matrix);
   col_map_offd    = hypre_ParCSRMatrixColMapOffd(h_matrix);
   num_rows        = hypre_ParCSRMatrixNumRows(h_matrix);
   row_starts      = hypre_ParCSRMatrixRowStarts(h_matrix);
   col_starts      = hypre_ParCSRMatrixColStarts(h_matrix);

   /* Diagonal matrix variables */
   diag_nnz  = hypre_CSRMatrixNumNonzeros(diag);
   diag_data = hypre_CSRMatrixData(diag);
   diag_i    = hypre_CSRMatrixI(diag);
   diag_j    = hypre_CSRMatrixJ(diag);

   /* Off-diagonal matrix variables */
   offd_nnz  = hypre_CSRMatrixNumNonzeros(offd);
   offd_data = hypre_CSRMatrixData(offd);
   offd_i    = hypre_CSRMatrixI(offd);
   offd_j    = hypre_CSRMatrixJ(offd);

   /* Set global matrix bounds */
   ilower = row_starts[0] + (HYPRE_BigInt) base_i;
   iupper = row_starts[1] + (HYPRE_BigInt) base_i - 1;
   jlower = col_starts[0] + (HYPRE_BigInt) base_j;
   jupper = col_starts[1] + (HYPRE_BigInt) base_j - 1;

   /* Open binary file */
   hypre_sprintf(new_filename, "%s.%05d.bin", filename, myid);
   if ((fp = fopen(new_filename, "wb")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not open output file!");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Write header (88 bytes)
    *---------------------------------------------*/

   count = 11;
   header[0]  = (hypre_uint64) 1; /* Header version */
   header[1]  = (hypre_uint64) sizeof(HYPRE_BigInt);
   header[2]  = (hypre_uint64) sizeof(HYPRE_Complex);
   header[3]  = (hypre_uint64) hypre_ParCSRMatrixGlobalNumRows(h_matrix);;
   header[4]  = (hypre_uint64) hypre_ParCSRMatrixGlobalNumCols(h_matrix);;
   header[5]  = (hypre_uint64) hypre_ParCSRMatrixDNumNonzeros(h_matrix);
   header[6]  = (hypre_uint64) diag_nnz + offd_nnz; /* local number of nonzeros*/
   header[7]  = (hypre_uint64) ilower;
   header[8]  = (hypre_uint64) iupper;
   header[9]  = (hypre_uint64) jlower;
   header[10] = (hypre_uint64) jupper;
   if (fwrite((const void*) header, sizeof(hypre_uint64), count, fp) != count)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not write all header entries\n");
      return hypre_error_flag;
   }

   /* Allocate memory for buffers */
   if (sizeof(HYPRE_BigInt) == sizeof(hypre_uint32))
   {
      i32buffer = hypre_TAlloc(hypre_uint32, header[6], HYPRE_MEMORY_HOST);
   }
   else if (sizeof(HYPRE_BigInt) == sizeof(hypre_uint64))
   {
      i64buffer = hypre_TAlloc(hypre_uint64, header[6], HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for row/column indices\n");
      return hypre_error_flag;
   }

   /* Allocate memory for buffers */
   if (sizeof(HYPRE_Complex) == sizeof(hypre_float))
   {
      f32buffer = hypre_TAlloc(hypre_float, header[6], HYPRE_MEMORY_HOST);
   }
   else if (sizeof(HYPRE_Complex) == sizeof(hypre_double))
   {
      f64buffer = hypre_TAlloc(hypre_double, header[6], HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Unsupported data type for matrix coefficients\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Write row indices to file
    *---------------------------------------------*/

   for (i = 0, k = 0; i < num_rows; i++)
   {
      bigI = first_row_index + (HYPRE_BigInt)(i + base_i);

      for (j = 0; j < (diag_i[i + 1] - diag_i[i]) + (offd_i[i + 1] - offd_i[i]); j++)
      {
         if (i32buffer)
         {
            i32buffer[k++] = (hypre_uint32) bigI;
         }
         else
         {
            i64buffer[k++] = (hypre_uint64) bigI;
         }
      }
   }

   /* Write buffer */
   if (i32buffer)
   {
      count = fwrite((const void*) i32buffer, sizeof(hypre_uint32), k, fp);
   }
   else if (i64buffer)
   {
      count = fwrite((const void*) i64buffer, sizeof(hypre_uint64), k, fp);
   }

   if (count != k)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not write all row indices entries\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Write columns indices to file
    *---------------------------------------------*/

   for (i = 0, k = 0; i < num_rows; i++)
   {
      for (j = diag_i[i]; j < diag_i[i + 1]; j++)
      {
         bigJ = first_col_diag + (HYPRE_BigInt)(diag_j[j] + base_j);

         if (i32buffer)
         {
            i32buffer[k++] = (hypre_uint32) bigJ;
         }
         else
         {
            i64buffer[k++] = (hypre_uint64) bigJ;
         }
      }

      for (j = offd_i[i]; j < offd_i[i + 1]; j++)
      {
         bigJ = col_map_offd[offd_j[j]] + (HYPRE_BigInt) base_j;

         if (i32buffer)
         {
            i32buffer[k++] = (hypre_uint32) bigJ;
         }
         else
         {
            i64buffer[k++] = (hypre_uint64) bigJ;
         }
      }
   }

   /* Write buffer */
   if (i32buffer)
   {
      count = fwrite((const void*) i32buffer, sizeof(hypre_uint32), k, fp);
   }
   else if (i64buffer)
   {
      count = fwrite((const void*) i64buffer, sizeof(hypre_uint64), k, fp);
   }

   if (count != k)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not write all column indices entries\n");
      return hypre_error_flag;
   }

   /*---------------------------------------------
    * Write coefficients indices to file
    *---------------------------------------------*/

   if (diag_data)
   {
      for (i = 0, k = 0; i < num_rows; i++)
      {
         for (j = diag_i[i]; j < diag_i[i + 1]; j++)
         {
            val = diag_data[j];

            if (f32buffer)
            {
               f32buffer[k++] = (hypre_float) val;
            }
            else
            {
               f64buffer[k++] = (hypre_double) val;
            }
         }

         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            val = offd_data[j];

            if (f32buffer)
            {
               f32buffer[k++] = (hypre_float) val;
            }
            else
            {
               f64buffer[k++] = (hypre_double) val;
            }
         }
      }

      /* Write buffer */
      if (f32buffer)
      {
         count = fwrite((const void*) f32buffer, sizeof(hypre_float), k, fp);
      }
      else if (f64buffer)
      {
         count = fwrite((const void*) f64buffer, sizeof(hypre_double), k, fp);
      }

      if (count != k)
      {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Could not write all matrix coefficients\n");
         return hypre_error_flag;
      }
   }

   fclose(fp);

   /*---------------------------------------------
    * Free memory
    *---------------------------------------------*/

   if (h_matrix != matrix)
   {
      hypre_ParCSRMatrixDestroy(h_matrix);
   }
   hypre_TFree(i32buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(i64buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(f32buffer, HYPRE_MEMORY_HOST);
   hypre_TFree(f64buffer, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixReadIJ
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixReadIJ( MPI_Comm             comm,
                          const char          *filename,
                          HYPRE_Int           *base_i_ptr,
                          HYPRE_Int           *base_j_ptr,
                          hypre_ParCSRMatrix **matrix_ptr)
{
   HYPRE_BigInt        global_num_rows;
   HYPRE_BigInt        global_num_cols;
   HYPRE_BigInt        first_row_index;
   HYPRE_BigInt        first_col_diag;
   HYPRE_BigInt        last_col_diag;
   hypre_ParCSRMatrix *matrix;
   hypre_CSRMatrix    *diag;
   hypre_CSRMatrix    *offd;
   HYPRE_BigInt       *col_map_offd;
   HYPRE_BigInt        row_starts[2];
   HYPRE_BigInt        col_starts[2];
   HYPRE_Int           num_rows;
   HYPRE_BigInt        big_base_i, big_base_j;
   HYPRE_Int           base_i, base_j;
   HYPRE_Complex      *diag_data;
   HYPRE_Int          *diag_i;
   HYPRE_Int          *diag_j;
   HYPRE_Complex      *offd_data = NULL;
   HYPRE_Int          *offd_i;
   HYPRE_Int          *offd_j = NULL;
   HYPRE_BigInt       *tmp_j = NULL;
   HYPRE_BigInt       *aux_offd_j;
   HYPRE_BigInt        I, J;
   HYPRE_Int           myid, num_procs, i, i2, j;
   char                new_filename[HYPRE_MAX_FILE_NAME_LEN];
   FILE               *file;
   HYPRE_Int           num_cols_offd, num_nonzeros_diag, num_nonzeros_offd;
   HYPRE_Int           i_col, num_cols;
   HYPRE_Int           diag_cnt, offd_cnt, row_cnt;
   HYPRE_Complex       data;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &myid);

   hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   hypre_fscanf(file, "%b %b", &global_num_rows, &global_num_cols);
   hypre_fscanf(file, "%d %d %d", &num_rows, &num_cols, &num_cols_offd);
   hypre_fscanf(file, "%d %d", &num_nonzeros_diag, &num_nonzeros_offd);
   hypre_fscanf(file, "%b %b %b %b", &row_starts[0], &col_starts[0], &row_starts[1], &col_starts[1]);

   big_base_i = row_starts[0];
   big_base_j = col_starts[0];
   base_i = (HYPRE_Int) row_starts[0];
   base_j = (HYPRE_Int) col_starts[0];

   matrix = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                     row_starts, col_starts, num_cols_offd,
                                     num_nonzeros_diag, num_nonzeros_offd);
   hypre_ParCSRMatrixInitialize(matrix);

   diag = hypre_ParCSRMatrixDiag(matrix);
   offd = hypre_ParCSRMatrixOffd(matrix);

   diag_data = hypre_CSRMatrixData(diag);
   diag_i    = hypre_CSRMatrixI(diag);
   diag_j    = hypre_CSRMatrixJ(diag);

   offd_i    = hypre_CSRMatrixI(offd);
   if (num_nonzeros_offd)
   {
      offd_data = hypre_CSRMatrixData(offd);
      offd_j    = hypre_CSRMatrixJ(offd);
      tmp_j     = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros_offd, HYPRE_MEMORY_HOST);
   }

   first_row_index = hypre_ParCSRMatrixFirstRowIndex(matrix);
   first_col_diag = hypre_ParCSRMatrixFirstColDiag(matrix);
   last_col_diag = first_col_diag + (HYPRE_BigInt)num_cols - 1;

   diag_cnt = 0;
   offd_cnt = 0;
   row_cnt = 0;
   for (i = 0; i < num_nonzeros_diag + num_nonzeros_offd; i++)
   {
      /* read values */
      hypre_fscanf(file, "%b %b %le", &I, &J, &data);
      i2 = (HYPRE_Int)(I - big_base_i - first_row_index);
      J -= big_base_j;
      if (i2 > row_cnt)
      {
         diag_i[i2] = diag_cnt;
         offd_i[i2] = offd_cnt;
         row_cnt++;
      }
      if (J < first_col_diag || J > last_col_diag)
      {
         tmp_j[offd_cnt] = J;
         offd_data[offd_cnt++] = data;
      }
      else
      {
         diag_j[diag_cnt] = (HYPRE_Int)(J - first_col_diag);
         diag_data[diag_cnt++] = data;
      }
   }
   diag_i[num_rows] = diag_cnt;
   offd_i[num_rows] = offd_cnt;

   fclose(file);

   /*  generate col_map_offd */
   if (num_nonzeros_offd)
   {
      aux_offd_j = hypre_CTAlloc(HYPRE_BigInt, num_nonzeros_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_nonzeros_offd; i++)
      {
         aux_offd_j[i] = (HYPRE_BigInt)offd_j[i];
      }
      hypre_BigQsort0(aux_offd_j, 0, num_nonzeros_offd - 1);
      col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);
      col_map_offd[0] = aux_offd_j[0];
      offd_cnt = 0;
      for (i = 1; i < num_nonzeros_offd; i++)
      {
         if (aux_offd_j[i] > col_map_offd[offd_cnt])
         {
            col_map_offd[++offd_cnt] = aux_offd_j[i];
         }
      }
      for (i = 0; i < num_nonzeros_offd; i++)
      {
         offd_j[i] = hypre_BigBinarySearch(col_map_offd, tmp_j[i], num_cols_offd);
      }
      hypre_TFree(aux_offd_j, HYPRE_MEMORY_HOST);
      hypre_TFree(tmp_j, HYPRE_MEMORY_HOST);
   }

   /* move diagonal element in first position in each row */
   for (i = 0; i < num_rows; i++)
   {
      i_col = diag_i[i];
      for (j = i_col; j < diag_i[i + 1]; j++)
      {
         if (diag_j[j] == i)
         {
            diag_j[j] = diag_j[i_col];
            data = diag_data[j];
            diag_data[j] = diag_data[i_col];
            diag_data[i_col] = data;
            diag_j[i_col] = i;
            break;
         }
      }
   }

   *base_i_ptr = base_i;
   *base_j_ptr = base_j;
   *matrix_ptr = matrix;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGetLocalRange
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGetLocalRange( hypre_ParCSRMatrix *matrix,
                                 HYPRE_BigInt       *row_start,
                                 HYPRE_BigInt       *row_end,
                                 HYPRE_BigInt       *col_start,
                                 HYPRE_BigInt       *col_end )
{
   HYPRE_Int my_id;

   if (!matrix)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_MPI_Comm_rank( hypre_ParCSRMatrixComm(matrix), &my_id );

   *row_start = hypre_ParCSRMatrixFirstRowIndex(matrix);
   *row_end = hypre_ParCSRMatrixLastRowIndex(matrix);
   *col_start =  hypre_ParCSRMatrixFirstColDiag(matrix);
   *col_end =  hypre_ParCSRMatrixLastColDiag(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGetRow
 * Returns global column indices and/or values for a given row in the global
 * matrix. Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the hypre_ParCSRMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.
 * Either one of col_ind and values can be left null, and those values will
 * not be returned.
 * All indices are returned in 0-based indexing, no matter what is used under
 * the hood. EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's hypre_MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixGetRowHost( hypre_ParCSRMatrix  *mat,
                              HYPRE_BigInt         row,
                              HYPRE_Int           *size,
                              HYPRE_BigInt       **col_ind,
                              HYPRE_Complex      **values )
{
   HYPRE_Int my_id;
   HYPRE_BigInt row_start, row_end;
   hypre_CSRMatrix *Aa;
   hypre_CSRMatrix *Ba;

   if (!mat)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   Aa = (hypre_CSRMatrix *) hypre_ParCSRMatrixDiag(mat);
   Ba = (hypre_CSRMatrix *) hypre_ParCSRMatrixOffd(mat);

   if (hypre_ParCSRMatrixGetrowactive(mat))
   {
      return (-1);
   }

   hypre_MPI_Comm_rank( hypre_ParCSRMatrixComm(mat), &my_id );

   hypre_ParCSRMatrixGetrowactive(mat) = 1;
   row_start = hypre_ParCSRMatrixFirstRowIndex(mat);
   row_end = hypre_ParCSRMatrixLastRowIndex(mat) + 1;
   if (row < row_start || row >= row_end)
   {
      return (-1);
   }

   /* if buffer is not allocated and some information is requested,
      allocate buffer */
   if (!hypre_ParCSRMatrixRowvalues(mat) && ( col_ind || values ))
   {
      /*
        allocate enough space to hold information from the longest row.
      */
      HYPRE_Int max = 1, tmp;
      HYPRE_Int i;
      HYPRE_Int m = row_end - row_start;

      for ( i = 0; i < m; i++ )
      {
         tmp = hypre_CSRMatrixI(Aa)[i + 1] - hypre_CSRMatrixI(Aa)[i] +
               hypre_CSRMatrixI(Ba)[i + 1] - hypre_CSRMatrixI(Ba)[i];
         if (max < tmp)
         {
            max = tmp;
         }
      }

      hypre_ParCSRMatrixRowvalues(mat)  =
         (HYPRE_Complex *) hypre_CTAlloc(HYPRE_Complex, max, hypre_ParCSRMatrixMemoryLocation(mat));
      hypre_ParCSRMatrixRowindices(mat) =
         (HYPRE_BigInt *)  hypre_CTAlloc(HYPRE_BigInt,  max, hypre_ParCSRMatrixMemoryLocation(mat));
   }

   /* Copy from dual sequential matrices into buffer */
   {
      HYPRE_Complex    *vworkA, *vworkB, *v_p;
      HYPRE_Int        i, *cworkA, *cworkB;
      HYPRE_BigInt     cstart = hypre_ParCSRMatrixFirstColDiag(mat);
      HYPRE_Int        nztot, nzA, nzB, lrow = (HYPRE_Int)(row - row_start);
      HYPRE_BigInt     *cmap, *idx_p;

      nzA = hypre_CSRMatrixI(Aa)[lrow + 1] - hypre_CSRMatrixI(Aa)[lrow];
      cworkA = &( hypre_CSRMatrixJ(Aa)[ hypre_CSRMatrixI(Aa)[lrow] ] );
      vworkA = &( hypre_CSRMatrixData(Aa)[ hypre_CSRMatrixI(Aa)[lrow] ] );

      nzB = hypre_CSRMatrixI(Ba)[lrow + 1] - hypre_CSRMatrixI(Ba)[lrow];
      cworkB = &( hypre_CSRMatrixJ(Ba)[ hypre_CSRMatrixI(Ba)[lrow] ] );
      vworkB = &( hypre_CSRMatrixData(Ba)[ hypre_CSRMatrixI(Ba)[lrow] ] );

      nztot = nzA + nzB;

      cmap = hypre_ParCSRMatrixColMapOffd(mat);

      if (values || col_ind)
      {
         if (nztot)
         {
            /* Sort by increasing column numbers, assuming A and B already sorted */
            HYPRE_Int imark = -1;

            if (values)
            {
               *values = v_p = hypre_ParCSRMatrixRowvalues(mat);
               for ( i = 0; i < nzB; i++ )
               {
                  if (cmap[cworkB[i]] < cstart)
                  {
                     v_p[i] = vworkB[i];
                  }
                  else
                  {
                     break;
                  }
               }
               imark = i;
               for ( i = 0; i < nzA; i++ )
               {
                  v_p[imark + i] = vworkA[i];
               }
               for ( i = imark; i < nzB; i++ )
               {
                  v_p[nzA + i] = vworkB[i];
               }
            }

            if (col_ind)
            {
               *col_ind = idx_p = hypre_ParCSRMatrixRowindices(mat);
               if (imark > -1)
               {
                  for ( i = 0; i < imark; i++ )
                  {
                     idx_p[i] = cmap[cworkB[i]];
                  }
               }
               else
               {
                  for ( i = 0; i < nzB; i++ )
                  {
                     if (cmap[cworkB[i]] < cstart)
                     {
                        idx_p[i] = cmap[cworkB[i]];
                     }
                     else
                     {
                        break;
                     }
                  }
                  imark = i;
               }
               for ( i = 0; i < nzA; i++ )
               {
                  idx_p[imark + i] = cstart + cworkA[i];
               }
               for ( i = imark; i < nzB; i++ )
               {
                  idx_p[nzA + i] = cmap[cworkB[i]];
               }
            }
         }
         else
         {
            if (col_ind)
            {
               *col_ind = 0;
            }
            if (values)
            {
               *values = 0;
            }
         }
      }

      *size = nztot;
   } /* End of copy */

   return hypre_error_flag;
}


HYPRE_Int
hypre_ParCSRMatrixGetRow( hypre_ParCSRMatrix  *mat,
                          HYPRE_BigInt         row,
                          HYPRE_Int           *size,
                          HYPRE_BigInt       **col_ind,
                          HYPRE_Complex      **values )
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(mat) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_ParCSRMatrixGetRowDevice(mat, row, size, col_ind, values);
   }
   else
#endif
   {
      return hypre_ParCSRMatrixGetRowHost(mat, row, size, col_ind, values);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixRestoreRow( hypre_ParCSRMatrix *matrix,
                              HYPRE_BigInt        row,
                              HYPRE_Int          *size,
                              HYPRE_BigInt      **col_ind,
                              HYPRE_Complex     **values )
{
   HYPRE_UNUSED_VAR(row);
   HYPRE_UNUSED_VAR(size);
   HYPRE_UNUSED_VAR(col_ind);
   HYPRE_UNUSED_VAR(values);

   if (!hypre_ParCSRMatrixGetrowactive(matrix))
   {
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   hypre_ParCSRMatrixGetrowactive(matrix) = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixToParCSRMatrix:
 *
 * Generates a ParCSRMatrix distributed across the processors in comm
 * from a CSRMatrix on proc 0 .
 *
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_CSRMatrixToParCSRMatrix( MPI_Comm         comm,
                               hypre_CSRMatrix *A,
                               HYPRE_BigInt    *global_row_starts,
                               HYPRE_BigInt    *global_col_starts )
{
   hypre_ParCSRMatrix *parcsr_A;

   HYPRE_BigInt       *global_data;
   HYPRE_BigInt        global_size;
   HYPRE_BigInt        global_num_rows;
   HYPRE_BigInt        global_num_cols;

   HYPRE_Int           num_procs, my_id;
   HYPRE_Int          *num_rows_proc;
   HYPRE_Int          *num_nonzeros_proc;
   HYPRE_BigInt        row_starts[2];
   HYPRE_BigInt        col_starts[2];

   hypre_CSRMatrix    *local_A;
   HYPRE_Complex      *A_data = NULL;
   HYPRE_Int          *A_i = NULL;
   HYPRE_Int          *A_j = NULL;

   hypre_MPI_Request  *requests;
   hypre_MPI_Status   *status, status0;
   hypre_MPI_Datatype *csr_matrix_datatypes;

   HYPRE_Int           free_global_row_starts = 0;
   HYPRE_Int           free_global_col_starts = 0;

   HYPRE_Int           total_size;
   HYPRE_BigInt        first_col_diag;
   HYPRE_BigInt        last_col_diag;
   HYPRE_Int           num_rows;
   HYPRE_Int           num_nonzeros;
   HYPRE_Int           i, ind;

   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   total_size = 4;
   if (my_id == 0)
   {
      total_size += 2 * (num_procs + 1);
   }

   global_data = hypre_CTAlloc(HYPRE_BigInt, total_size, HYPRE_MEMORY_HOST);
   if (my_id == 0)
   {
      global_size = 3;
      if (global_row_starts)
      {
         if (global_col_starts)
         {
            if (global_col_starts != global_row_starts)
            {
               /* contains code for what to expect,
                  if 0: global_row_starts = global_col_starts, only global_row_starts given
                  if 1: only global_row_starts given, global_col_starts = NULL
                  if 2: both global_row_starts and global_col_starts given
                  if 3: only global_col_starts given, global_row_starts = NULL */
               global_data[3] = 2;
               global_size += (HYPRE_BigInt) (2 * (num_procs + 1) + 1);
               for (i = 0; i < (num_procs + 1); i++)
               {
                  global_data[i + 4] = global_row_starts[i];
               }
               for (i = 0; i < (num_procs + 1); i++)
               {
                  global_data[i + num_procs + 5] = global_col_starts[i];
               }
            }
            else
            {
               global_data[3] = 0;
               global_size += (HYPRE_BigInt) ((num_procs + 1) + 1);
               for (i = 0; i < (num_procs + 1); i++)
               {
                  global_data[i + 4] = global_row_starts[i];
               }
            }
         }
         else
         {
            global_data[3] = 1;
            global_size += (HYPRE_BigInt) ((num_procs + 1) + 1);
            for (i = 0; i < (num_procs + 1); i++)
            {
               global_data[i + 4] = global_row_starts[i];
            }
         }
      }
      else
      {
         if (global_col_starts)
         {
            global_data[3] = 3;
            global_size += (HYPRE_BigInt) ((num_procs + 1) + 1);
            for (i = 0; i < (num_procs + 1); i++)
            {
               global_data[i + 4] = global_col_starts[i];
            }
         }
      }

      global_data[0] = (HYPRE_BigInt) hypre_CSRMatrixNumRows(A);
      global_data[1] = (HYPRE_BigInt) hypre_CSRMatrixNumCols(A);
      global_data[2] = global_size;
      A_data = hypre_CSRMatrixData(A);
      A_i = hypre_CSRMatrixI(A);
      A_j = hypre_CSRMatrixJ(A);
   }
   hypre_MPI_Bcast(global_data, 3, HYPRE_MPI_BIG_INT, 0, comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];
   global_size     = global_data[2];

   if (global_size > 3)
   {
      HYPRE_Int  send_start;

      if (global_data[3] == 2)
      {
         send_start = 4;
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &row_starts[0], 1, HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5;
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &row_starts[1], 1, HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 4 + (num_procs + 1);
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &col_starts[0], 1, HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5 + (num_procs + 1);
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &col_starts[1], 1, HYPRE_MPI_BIG_INT, 0, comm);
      }
      else if ((global_data[3] == 0) || (global_data[3] == 1))
      {
         send_start = 4;
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &row_starts[0], 1, HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5;
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &row_starts[1], 1, HYPRE_MPI_BIG_INT, 0, comm);

         if (global_data[3] == 0)
         {
            col_starts[0] = row_starts[0];
            col_starts[1] = row_starts[1];
         }
      }
      else
      {
         send_start = 4;
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &col_starts[0], 1, HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5;
         hypre_MPI_Scatter(&global_data[send_start], 1, HYPRE_MPI_BIG_INT,
                           &col_starts[1], 1, HYPRE_MPI_BIG_INT, 0, comm);
      }
   }
   hypre_TFree(global_data, HYPRE_MEMORY_HOST);

   // Create ParCSR matrix
   parcsr_A = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                       row_starts, col_starts, 0, 0, 0);

   // Allocate memory for building ParCSR matrix
   num_rows_proc     = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);
   num_nonzeros_proc = hypre_CTAlloc(HYPRE_Int, num_procs, HYPRE_MEMORY_HOST);

   if (my_id == 0)
   {
      if (!global_row_starts)
      {
         hypre_GeneratePartitioning(global_num_rows, num_procs, &global_row_starts);
         free_global_row_starts = 1;
      }
      if (!global_col_starts)
      {
         hypre_GeneratePartitioning(global_num_rows, num_procs, &global_col_starts);
         free_global_col_starts = 1;
      }

      for (i = 0; i < num_procs; i++)
      {
         num_rows_proc[i] = (HYPRE_Int) (global_row_starts[i + 1] - global_row_starts[i]);
         num_nonzeros_proc[i] = A_i[(HYPRE_Int)global_row_starts[i + 1]] -
                                A_i[(HYPRE_Int)global_row_starts[i]];
      }
      //num_nonzeros_proc[num_procs-1] = A_i[(HYPRE_Int)global_num_rows] - A_i[(HYPRE_Int)row_starts[num_procs-1]];
   }
   hypre_MPI_Scatter(num_rows_proc, 1, HYPRE_MPI_INT, &num_rows, 1, HYPRE_MPI_INT, 0, comm);
   hypre_MPI_Scatter(num_nonzeros_proc, 1, HYPRE_MPI_INT, &num_nonzeros, 1, HYPRE_MPI_INT, 0, comm);

   /* RL: this is not correct: (HYPRE_Int) global_num_cols */
   local_A = hypre_CSRMatrixCreate(num_rows, (HYPRE_Int) global_num_cols, num_nonzeros);

   csr_matrix_datatypes = hypre_CTAlloc(hypre_MPI_Datatype,  num_procs, HYPRE_MEMORY_HOST);
   if (my_id == 0)
   {
      requests = hypre_CTAlloc(hypre_MPI_Request, num_procs - 1, HYPRE_MEMORY_HOST);
      status = hypre_CTAlloc(hypre_MPI_Status, num_procs - 1, HYPRE_MEMORY_HOST);
      for (i = 1; i < num_procs; i++)
      {
         ind = A_i[(HYPRE_Int) global_row_starts[i]];

         hypre_BuildCSRMatrixMPIDataType(num_nonzeros_proc[i],
                                         num_rows_proc[i],
                                         &A_data[ind],
                                         &A_i[(HYPRE_Int) global_row_starts[i]],
                                         &A_j[ind],
                                         &csr_matrix_datatypes[i]);
         hypre_MPI_Isend(hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
                         &requests[i - 1]);
         hypre_MPI_Type_free(&csr_matrix_datatypes[i]);
      }
      hypre_CSRMatrixData(local_A) = A_data;
      hypre_CSRMatrixI(local_A) = A_i;
      hypre_CSRMatrixJ(local_A) = A_j;
      hypre_CSRMatrixOwnsData(local_A) = 0;

      hypre_MPI_Waitall(num_procs - 1, requests, status);

      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(num_rows_proc, HYPRE_MEMORY_HOST);
      hypre_TFree(num_nonzeros_proc, HYPRE_MEMORY_HOST);

      if (free_global_row_starts)
      {
         hypre_TFree(global_row_starts, HYPRE_MEMORY_HOST);
      }
      if (free_global_col_starts)
      {
         hypre_TFree(global_col_starts, HYPRE_MEMORY_HOST);
      }
   }
   else
   {
      hypre_CSRMatrixInitialize(local_A);
      hypre_BuildCSRMatrixMPIDataType(num_nonzeros,
                                      num_rows,
                                      hypre_CSRMatrixData(local_A),
                                      hypre_CSRMatrixI(local_A),
                                      hypre_CSRMatrixJ(local_A),
                                      &csr_matrix_datatypes[0]);
      hypre_MPI_Recv(hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[0], 0, 0, comm, &status0);
      hypre_MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = hypre_ParCSRMatrixFirstColDiag(parcsr_A);
   last_col_diag  = hypre_ParCSRMatrixLastColDiag(parcsr_A);

   GenerateDiagAndOffd(local_A, parcsr_A, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {
      hypre_CSRMatrixData(local_A) = NULL;
      hypre_CSRMatrixI(local_A) = NULL;
      hypre_CSRMatrixJ(local_A) = NULL;
   }
   hypre_CSRMatrixDestroy(local_A);
   hypre_TFree(csr_matrix_datatypes, HYPRE_MEMORY_HOST);

   return parcsr_A;
}

/* RL: XXX this is not a scalable routine, see `marker' therein */
HYPRE_Int
GenerateDiagAndOffd(hypre_CSRMatrix    *A,
                    hypre_ParCSRMatrix *matrix,
                    HYPRE_BigInt        first_col_diag,
                    HYPRE_BigInt        last_col_diag)
{
   HYPRE_Int  i, j;
   HYPRE_Int  jo, jd;
   HYPRE_Int  num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int  num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex *a_data = hypre_CSRMatrixData(A);
   HYPRE_Int *a_i = hypre_CSRMatrixI(A);
   /*RL: XXX FIXME if A spans global column space, the following a_j should be bigJ */
   HYPRE_Int *a_j = hypre_CSRMatrixJ(A);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);

   HYPRE_BigInt  *col_map_offd;

   HYPRE_Complex *diag_data, *offd_data;
   HYPRE_Int  *diag_i, *offd_i;
   HYPRE_Int  *diag_j, *offd_j;
   HYPRE_Int  *marker;
   HYPRE_Int num_cols_diag, num_cols_offd;
   HYPRE_Int first_elmt = a_i[0];
   HYPRE_Int num_nonzeros = a_i[num_rows] - first_elmt;
   HYPRE_Int counter;

   num_cols_diag = (HYPRE_Int)(last_col_diag - first_col_diag + 1);
   num_cols_offd = 0;

   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);

   if (num_cols - num_cols_diag)
   {
      hypre_CSRMatrixInitialize_v2(diag, 0, memory_location);
      diag_i = hypre_CSRMatrixI(diag);

      hypre_CSRMatrixInitialize_v2(offd, 0, memory_location);
      offd_i = hypre_CSRMatrixI(offd);
      marker = hypre_CTAlloc(HYPRE_Int, num_cols, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols; i++)
      {
         marker[i] = 0;
      }

      jo = 0;
      jd = 0;
      for (i = 0; i < num_rows; i++)
      {
         offd_i[i] = jo;
         diag_i[i] = jd;

         for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++)
         {
            if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
            {
               if (!marker[a_j[j]])
               {
                  marker[a_j[j]] = 1;
                  num_cols_offd++;
               }
               jo++;
            }
            else
            {
               jd++;
            }
         }
      }
      offd_i[num_rows] = jo;
      diag_i[num_rows] = jd;

      hypre_ParCSRMatrixColMapOffd(matrix) = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd,
                                                           HYPRE_MEMORY_HOST);
      col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix);

      counter = 0;
      for (i = 0; i < num_cols; i++)
      {
         if (marker[i])
         {
            col_map_offd[counter] = (HYPRE_BigInt) i;
            marker[i] = counter;
            counter++;
         }
      }

      hypre_CSRMatrixNumNonzeros(diag) = jd;
      hypre_CSRMatrixInitialize(diag);
      diag_data = hypre_CSRMatrixData(diag);
      diag_j = hypre_CSRMatrixJ(diag);

      hypre_CSRMatrixNumNonzeros(offd) = jo;
      hypre_CSRMatrixNumCols(offd) = num_cols_offd;
      hypre_CSRMatrixInitialize(offd);
      offd_data = hypre_CSRMatrixData(offd);
      offd_j = hypre_CSRMatrixJ(offd);

      jo = 0;
      jd = 0;
      for (i = 0; i < num_rows; i++)
      {
         for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++)
         {
            if (a_j[j] < (HYPRE_Int)first_col_diag || a_j[j] > (HYPRE_Int)last_col_diag)
            {
               offd_data[jo] = a_data[j];
               offd_j[jo++] = marker[a_j[j]];
            }
            else
            {
               diag_data[jd] = a_data[j];
               diag_j[jd++] = (HYPRE_Int)(a_j[j] - first_col_diag);
            }
         }
      }
      hypre_TFree(marker, HYPRE_MEMORY_HOST);
   }
   else
   {
      hypre_CSRMatrixNumNonzeros(diag) = num_nonzeros;
      hypre_CSRMatrixInitialize(diag);
      diag_data = hypre_CSRMatrixData(diag);
      diag_i = hypre_CSRMatrixI(diag);
      diag_j = hypre_CSRMatrixJ(diag);

      for (i = 0; i < num_nonzeros; i++)
      {
         diag_data[i] = a_data[i];
         diag_j[i] = a_j[i];
      }
      offd_i = hypre_CTAlloc(HYPRE_Int,  num_rows + 1, HYPRE_MEMORY_HOST);

      for (i = 0; i < num_rows + 1; i++)
      {
         diag_i[i] = a_i[i];
         offd_i[i] = 0;
      }

      hypre_CSRMatrixNumCols(offd) = 0;
      hypre_CSRMatrixI(offd) = offd_i;
   }

   return hypre_error_flag;
}

hypre_CSRMatrix *
hypre_MergeDiagAndOffdHost(hypre_ParCSRMatrix *par_matrix)
{
   hypre_CSRMatrix  *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix  *offd = hypre_ParCSRMatrixOffd(par_matrix);
   hypre_CSRMatrix  *matrix;

   HYPRE_BigInt       num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   HYPRE_BigInt       first_col_diag = hypre_ParCSRMatrixFirstColDiag(par_matrix);
   HYPRE_BigInt      *col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
   HYPRE_Int          num_rows = hypre_CSRMatrixNumRows(diag);

   HYPRE_Int          *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int          *diag_j = hypre_CSRMatrixJ(diag);
   HYPRE_Complex      *diag_data = hypre_CSRMatrixData(diag);
   HYPRE_Int          *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int          *offd_j = hypre_CSRMatrixJ(offd);
   HYPRE_Complex      *offd_data = hypre_CSRMatrixData(offd);

   HYPRE_Int          *matrix_i;
   HYPRE_BigInt       *matrix_j;
   HYPRE_Complex      *matrix_data;

   HYPRE_Int          num_nonzeros, i, j;
   HYPRE_Int          count;
   HYPRE_Int          size, rest, num_threads, ii;

   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(par_matrix);

   num_nonzeros = diag_i[num_rows] + offd_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixMemoryLocation(matrix) = memory_location;
   hypre_CSRMatrixBigInitialize(matrix);

   matrix_i = hypre_CSRMatrixI(matrix);
   matrix_j = hypre_CSRMatrixBigJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);
   num_threads = hypre_NumThreads();
   size = num_rows / num_threads;
   rest = num_rows - size * num_threads;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(ii, i, j, count) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
      HYPRE_Int ns, ne;
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }
      count = diag_i[ns] + offd_i[ns];;
      for (i = ns; i < ne; i++)
      {
         matrix_i[i] = count;
         for (j = diag_i[i]; j < diag_i[i + 1]; j++)
         {
            matrix_data[count] = diag_data[j];
            matrix_j[count++] = (HYPRE_BigInt)diag_j[j] + first_col_diag;
         }
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            matrix_data[count] = offd_data[j];
            matrix_j[count++] = col_map_offd[offd_j[j]];
         }
      }
   } /* end parallel region */

   matrix_i[num_rows] = num_nonzeros;

   return matrix;
}

hypre_CSRMatrix *
hypre_MergeDiagAndOffd(hypre_ParCSRMatrix *par_matrix)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(par_matrix) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_MergeDiagAndOffdDevice(par_matrix);
   }
   else
#endif
   {
      return hypre_MergeDiagAndOffdHost(par_matrix);
   }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixToCSRMatrixAll
 *
 * The resulting matrix is stored in the space given by memory_location
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_ParCSRMatrixToCSRMatrixAll(hypre_ParCSRMatrix *par_A)
{
   return hypre_ParCSRMatrixToCSRMatrixAll_v2(par_A, hypre_ParCSRMatrixMemoryLocation(par_A));
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixToCSRMatrixAll_v2
 *
 * Generates a CSRMatrix from a ParCSRMatrix on all processors that have
 * parts of the ParCSRMatrix
 *
 * Warning: This only works for a ParCSRMatrix with num_rows < 2,147,483,647
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_ParCSRMatrixToCSRMatrixAll_v2( hypre_ParCSRMatrix   *par_matrix,
                                     HYPRE_MemoryLocation  memory_location )
{
   MPI_Comm                   comm = hypre_ParCSRMatrixComm(par_matrix);
   HYPRE_Int                  num_rows = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(par_matrix);
   HYPRE_Int                  num_cols = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   HYPRE_BigInt               first_row_index = hypre_ParCSRMatrixFirstRowIndex(par_matrix);
   HYPRE_BigInt               last_row_index  = hypre_ParCSRMatrixLastRowIndex(par_matrix);

   hypre_ParCSRMatrix        *par_temp;
   hypre_CSRMatrix           *matrix;
   HYPRE_Int                 *matrix_i;
   HYPRE_Int                 *matrix_j;
   HYPRE_Complex             *matrix_data;

   hypre_CSRMatrix           *local_matrix;
   HYPRE_Int                  local_num_rows;
   HYPRE_Int                  local_num_nonzeros;
   HYPRE_Int                 *local_matrix_i;
   HYPRE_Int                 *local_matrix_j;
   HYPRE_Complex             *local_matrix_data;

   HYPRE_Int                  i, j;
   HYPRE_Int                  num_nonzeros;
   HYPRE_Int                  num_data;
   HYPRE_Int                  num_requests;
   HYPRE_Int                  vec_len, offset;
   HYPRE_Int                  start_index;
   HYPRE_Int                  proc_id;
   HYPRE_Int                  num_procs, my_id;
   HYPRE_Int                  num_types;
   HYPRE_Int                 *used_procs;
   HYPRE_Int                 *new_vec_starts;
   hypre_MPI_Request         *requests;
   hypre_MPI_Status          *status;
   HYPRE_Int                  num_contacts;
   HYPRE_Int                  contact_proc_list[1];
   HYPRE_Int                  contact_send_buf[1];
   HYPRE_Int                  contact_send_buf_starts[2];
   HYPRE_Int                  max_response_size;
   HYPRE_Int                 *response_recv_buf = NULL;
   HYPRE_Int                 *response_recv_buf_starts = NULL;
   hypre_DataExchangeResponse response_obj;
   hypre_ProcListElements     send_proc_obj;

   HYPRE_Int                 *send_info = NULL;
   hypre_MPI_Status           status1;
   HYPRE_Int                  count, start;
   HYPRE_Int                  tag1 = 11112, tag2 = 22223, tag3 = 33334;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Clone input matrix to host memory */
   par_temp = hypre_ParCSRMatrixClone_v2(par_matrix, 1, HYPRE_MEMORY_HOST);

   /* Creates local matrix on host memory */
   local_matrix = hypre_MergeDiagAndOffd(par_temp);
   hypre_ParCSRMatrixDestroy(par_temp);

   /* copies big_j to j */
   hypre_CSRMatrixBigJtoJ(local_matrix);

   local_matrix_i = hypre_CSRMatrixI(local_matrix);
   local_matrix_j = hypre_CSRMatrixJ(local_matrix);
   local_matrix_data = hypre_CSRMatrixData(local_matrix);
   local_num_rows = (HYPRE_Int) (last_row_index - first_row_index + 1);

   /* determine procs that have vector data and store their ids in used_procs */
   /* we need to do an exchange data for this.  If I own row then I will contact
      processor 0 with the endpoint of my local range */
   if (local_num_rows > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0]  = (HYPRE_Int) hypre_ParCSRMatrixLastRowIndex(par_matrix);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;
   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }
   /*build the response object*/
   /*send_proc_obj will  be for saving info from contacts */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = hypre_CTAlloc(HYPRE_Int, send_proc_obj.storage_length, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts = hypre_CTAlloc(HYPRE_Int, send_proc_obj.storage_length + 1,
                                            HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements = hypre_CTAlloc(HYPRE_BigInt, send_proc_obj.element_storage_length,
                                          HYPRE_MEMORY_HOST);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = hypre_FillResponseParToCSRMatrix;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/

   hypre_DataExchangeList(num_contacts,
                          contact_proc_list, contact_send_buf,
                          contact_send_buf_starts, sizeof(HYPRE_Int),
                          sizeof(HYPRE_Int), &response_obj,
                          max_response_size, 1,
                          comm, (void**) &response_recv_buf,
                          &response_recv_buf_starts);

   /* now processor 0 should have a list of ranges for processors that have rows -
      these are in send_proc_obj - it needs to create the new list of processors
      and also an array of vec starts - and send to those who own row*/
   if (my_id)
   {
      if (local_num_rows)
      {
         /* look for a message from processor 0 */
         hypre_MPI_Probe(0, tag1, comm, &status1);
         hypre_MPI_Get_count(&status1, HYPRE_MPI_INT, &count);

         send_info = hypre_CTAlloc(HYPRE_Int, count, HYPRE_MEMORY_HOST);
         hypre_MPI_Recv(send_info, count, HYPRE_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */
         num_types = send_info[0];
         used_procs =  hypre_CTAlloc(HYPRE_Int, num_types, HYPRE_MEMORY_HOST);
         new_vec_starts = hypre_CTAlloc(HYPRE_Int, num_types + 1, HYPRE_MEMORY_HOST);

         for (i = 1; i <= num_types; i++)
         {
            used_procs[i - 1] = send_info[i];
         }
         for (i = num_types + 1; i < count; i++)
         {
            new_vec_starts[i - num_types - 1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         hypre_TFree(send_proc_obj.vec_starts, HYPRE_MEMORY_HOST);
         hypre_TFree(send_proc_obj.id, HYPRE_MEMORY_HOST);
         hypre_TFree(send_proc_obj.elements, HYPRE_MEMORY_HOST);
         hypre_TFree(response_recv_buf, HYPRE_MEMORY_HOST);
         hypre_TFree(response_recv_buf_starts, HYPRE_MEMORY_HOST);
         hypre_CSRMatrixDestroy(local_matrix);

         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types      = send_proc_obj.length;
      used_procs     = hypre_CTAlloc(HYPRE_Int, num_types, HYPRE_MEMORY_HOST);
      new_vec_starts = hypre_CTAlloc(HYPRE_Int, num_types + 1, HYPRE_MEMORY_HOST);

      new_vec_starts[0] = 0;
      for (i = 0; i < num_types; i++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i + 1] = send_proc_obj.elements[i] + 1;
      }
      hypre_qsort0(used_procs, 0, num_types - 1);
      hypre_qsort0(new_vec_starts, 0, num_types);

      /* Now we need to put into an array to send */
      count = 2 * num_types + 2;
      send_info = hypre_CTAlloc(HYPRE_Int, count, HYPRE_MEMORY_HOST);
      send_info[0] = num_types;
      for (i = 1; i <= num_types; i++)
      {
         send_info[i] = (HYPRE_BigInt) used_procs[i - 1];
      }
      for (i = num_types + 1; i < count; i++)
      {
         send_info[i] = new_vec_starts[i - num_types - 1];
      }
      requests = hypre_CTAlloc(hypre_MPI_Request, num_types, HYPRE_MEMORY_HOST);
      status   = hypre_CTAlloc(hypre_MPI_Status, num_types, HYPRE_MEMORY_HOST);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (num_types && used_procs[0] == 0)
      {
         start = 1;
      }

      for (i = start; i < num_types; i++)
      {
         hypre_MPI_Isend(send_info, count, HYPRE_MPI_INT, used_procs[i], tag1,
                         comm, &requests[i - start]);
      }
      hypre_MPI_Waitall(num_types - start, requests, status);

      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
   }

   /* Clean up */
   hypre_TFree(send_proc_obj.vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.id, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.elements, HYPRE_MEMORY_HOST);
   hypre_TFree(send_info, HYPRE_MEMORY_HOST);
   hypre_TFree(response_recv_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_recv_buf_starts, HYPRE_MEMORY_HOST);

   /* now proc 0 can exit if it has no rows */
   if (!local_num_rows)
   {
      hypre_CSRMatrixDestroy(local_matrix);
      hypre_TFree(new_vec_starts, HYPRE_MEMORY_HOST);
      hypre_TFree(used_procs, HYPRE_MEMORY_HOST);

      return NULL;
   }

   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

   /* this matrix should be rather small */
   matrix_i = hypre_CTAlloc(HYPRE_Int, num_rows + 1, HYPRE_MEMORY_HOST);

   num_requests = 4 * num_types;
   requests = hypre_CTAlloc(hypre_MPI_Request, num_requests, HYPRE_MEMORY_HOST);
   status   = hypre_CTAlloc(hypre_MPI_Status, num_requests, HYPRE_MEMORY_HOST);

   /* exchange contents of local_matrix_i - here we are sending to ourself also*/
   j = 0;
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      vec_len = (HYPRE_Int)(new_vec_starts[i + 1] - new_vec_starts[i]);
      hypre_MPI_Irecv(&matrix_i[new_vec_starts[i] + 1], vec_len, HYPRE_MPI_INT,
                      proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      hypre_MPI_Isend(&local_matrix_i[1], local_num_rows, HYPRE_MPI_INT,
                      proc_id, tag2, comm, &requests[j++]);
   }
   hypre_MPI_Waitall(j, requests, status);

   /* generate matrix_i from received data */
   /* global numbering?*/
   offset = matrix_i[new_vec_starts[1]];
   for (i = 1; i < num_types; i++)
   {
      for (j = new_vec_starts[i]; j < new_vec_starts[i + 1]; j++)
      {
         matrix_i[j + 1] += offset;
      }
      offset = matrix_i[new_vec_starts[i + 1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize_v2(matrix, 0, HYPRE_MEMORY_HOST);
   matrix_j = hypre_CSRMatrixJ(matrix);
   matrix_data = hypre_CSRMatrixData(matrix);

   /* generate datatypes for further data exchange and exchange remaining
      data, i.e. column info and actual data */
   j = 0;
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      start_index = matrix_i[(HYPRE_Int)new_vec_starts[i]];
      num_data = matrix_i[(HYPRE_Int)new_vec_starts[i + 1]] - start_index;
      hypre_MPI_Irecv(&matrix_data[start_index], num_data, HYPRE_MPI_COMPLEX,
                      used_procs[i], tag1, comm, &requests[j++]);
      hypre_MPI_Irecv(&matrix_j[start_index], num_data, HYPRE_MPI_INT,
                      used_procs[i], tag3, comm, &requests[j++]);
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   for (i = 0; i < num_types; i++)
   {
      hypre_MPI_Isend(local_matrix_data, local_num_nonzeros, HYPRE_MPI_COMPLEX,
                      used_procs[i], tag1, comm, &requests[j++]);
      hypre_MPI_Isend(local_matrix_j, local_num_nonzeros, HYPRE_MPI_INT,
                      used_procs[i], tag3, comm, &requests[j++]);
   }

   hypre_MPI_Waitall(num_requests, requests, status);
   hypre_TFree(new_vec_starts, HYPRE_MEMORY_HOST);
   if (num_requests)
   {
      hypre_TFree(requests, HYPRE_MEMORY_HOST);
      hypre_TFree(status, HYPRE_MEMORY_HOST);
      hypre_TFree(used_procs, HYPRE_MEMORY_HOST);
   }
   hypre_CSRMatrixDestroy(local_matrix);

   /* Move resulting matrix to the memory location passed as input */
   hypre_CSRMatrixMigrate(matrix, memory_location);

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCopy,
 * copies B to A,
 * if copy_data = 0, only the structure of A is copied to B
 * the routine does not check whether the dimensions of A and B are compatible
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixCopy( hypre_ParCSRMatrix *A,
                        hypre_ParCSRMatrix *B,
                        HYPRE_Int copy_data )
{
   hypre_CSRMatrix *A_diag;
   hypre_CSRMatrix *A_offd;
   HYPRE_BigInt *col_map_offd_A;
   hypre_CSRMatrix *B_diag;
   hypre_CSRMatrix *B_offd;
   HYPRE_BigInt *col_map_offd_B;
   HYPRE_Int num_cols_offd_A;
   HYPRE_Int num_cols_offd_B;

   if (!A)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if (!B)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   A_diag = hypre_ParCSRMatrixDiag(A);
   A_offd = hypre_ParCSRMatrixOffd(A);
   B_diag = hypre_ParCSRMatrixDiag(B);
   B_offd = hypre_ParCSRMatrixOffd(B);

   num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);
   num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_assert(num_cols_offd_A == num_cols_offd_B);

   col_map_offd_A = hypre_ParCSRMatrixColMapOffd(A);
   col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   hypre_CSRMatrixCopy(A_diag, B_diag, copy_data);
   hypre_CSRMatrixCopy(A_offd, B_offd, copy_data);

   /* should not happen if B has been initialized */
   if (num_cols_offd_B && col_map_offd_B == NULL)
   {
      col_map_offd_B = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_B, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixColMapOffd(B) = col_map_offd_B;
   }

   hypre_TMemcpy(col_map_offd_B, col_map_offd_A, HYPRE_BigInt, num_cols_offd_B,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_FillResponseParToCSRMatrix
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FillResponseParToCSRMatrix( void       *p_recv_contact_buf,
                                  HYPRE_Int   contact_size,
                                  HYPRE_Int   contact_proc,
                                  void       *ro,
                                  MPI_Comm    comm,
                                  void      **p_send_response_buf,
                                  HYPRE_Int *response_message_size )
{
   HYPRE_UNUSED_VAR(p_send_response_buf);

   HYPRE_Int    myid;
   HYPRE_Int    i, index, count, elength;

   HYPRE_BigInt    *recv_contact_buf = (HYPRE_BigInt * ) p_recv_contact_buf;

   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse*)ro;

   hypre_ProcListElements      *send_proc_obj = (hypre_ProcListElements*)response_obj->data2;

   hypre_MPI_Comm_rank(comm, &myid );

   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 10; /*add space for 10 more processors*/
      send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id, HYPRE_Int,
                                         send_proc_obj->storage_length, HYPRE_MEMORY_HOST);
      send_proc_obj->vec_starts =
         hypre_TReAlloc(send_proc_obj->vec_starts, HYPRE_Int,
                        send_proc_obj->storage_length + 1, HYPRE_MEMORY_HOST);
   }

   /*initialize*/
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/
   send_proc_obj->id[count] = contact_proc;

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 10);
      elength += index;
      send_proc_obj->elements = hypre_TReAlloc(send_proc_obj->elements,
                                               HYPRE_BigInt,  elength, HYPRE_MEMORY_HOST);
      send_proc_obj->element_storage_length = elength;
   }
   /*populate send_proc_obj*/
   for (i = 0; i < contact_size; i++)
   {
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count + 1] = index;
   send_proc_obj->length++;

   /*output - no message to return (confirmation) */
   *response_message_size = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixUnion
 * Creates and returns a new matrix whose elements are the union of A and B.
 * Data is not copied, only structural information is created.
 * A and B must have the same communicator, numbers and distributions of rows
 * and columns (they can differ in which row-column pairs are nonzero, thus
 * in which columns are in a offd block)
 *
 * TODO (VPM): This function should use hypre_ParCSRMatrixCreate to create
 *             the matrix.
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixUnion( hypre_ParCSRMatrix *A,
                         hypre_ParCSRMatrix *B )
{
   hypre_ParCSRMatrix *C;
   HYPRE_BigInt       *col_map_offd_C = NULL;
   HYPRE_Int           my_id, p;
   MPI_Comm            comm = hypre_ParCSRMatrixComm( A );

   hypre_MPI_Comm_rank(comm, &my_id);

   C = hypre_CTAlloc( hypre_ParCSRMatrix,  1, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixComm( C ) = hypre_ParCSRMatrixComm( A );
   hypre_ParCSRMatrixGlobalNumRows( C ) = hypre_ParCSRMatrixGlobalNumRows( A );
   hypre_ParCSRMatrixGlobalNumCols( C ) = hypre_ParCSRMatrixGlobalNumCols( A );
   hypre_ParCSRMatrixFirstRowIndex( C ) = hypre_ParCSRMatrixFirstRowIndex( A );
   hypre_assert( hypre_ParCSRMatrixFirstRowIndex( B )
                 == hypre_ParCSRMatrixFirstRowIndex( A ) );
   hypre_TMemcpy(hypre_ParCSRMatrixRowStarts(C), hypre_ParCSRMatrixRowStarts(A),
                 HYPRE_BigInt, 2, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(hypre_ParCSRMatrixColStarts(C), hypre_ParCSRMatrixColStarts(A),
                 HYPRE_BigInt, 2, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   for (p = 0; p < 2; ++p)
      hypre_assert( hypre_ParCSRMatrixColStarts(A)[p]
                    == hypre_ParCSRMatrixColStarts(B)[p] );
   hypre_ParCSRMatrixFirstColDiag( C ) = hypre_ParCSRMatrixFirstColDiag( A );
   hypre_ParCSRMatrixLastRowIndex( C ) = hypre_ParCSRMatrixLastRowIndex( A );
   hypre_ParCSRMatrixLastColDiag( C ) = hypre_ParCSRMatrixLastColDiag( A );

   hypre_ParCSRMatrixDiag( C ) =
      hypre_CSRMatrixUnion( hypre_ParCSRMatrixDiag(A), hypre_ParCSRMatrixDiag(B),
                            0, 0, 0 );
   hypre_ParCSRMatrixOffd( C ) =
      hypre_CSRMatrixUnion( hypre_ParCSRMatrixOffd(A), hypre_ParCSRMatrixOffd(B),
                            hypre_ParCSRMatrixColMapOffd(A),
                            hypre_ParCSRMatrixColMapOffd(B), &col_map_offd_C );
   hypre_ParCSRMatrixColMapOffd( C ) = col_map_offd_C;
   hypre_ParCSRMatrixCommPkg( C ) = NULL;
   hypre_ParCSRMatrixCommPkgT( C ) = NULL;
   hypre_ParCSRMatrixOwnsData( C ) = 1;
   /*  SetNumNonzeros, SetDNumNonzeros are global, need hypre_MPI_Allreduce.
       I suspect, but don't know, that other parts of hypre do not assume that
       the correct values have been set.
       hypre_ParCSRMatrixSetNumNonzeros( C );
       hypre_ParCSRMatrixSetDNumNonzeros( C );*/
   hypre_ParCSRMatrixNumNonzeros( C ) = 0;
   hypre_ParCSRMatrixDNumNonzeros( C ) = 0.0;
   hypre_ParCSRMatrixRowindices( C ) = NULL;
   hypre_ParCSRMatrixRowvalues( C ) = NULL;
   hypre_ParCSRMatrixGetrowactive( C ) = 0;

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixTruncate
 *
 * Perform dual truncation of ParCSR matrix.
 * This code is adapted from original BoomerAMGInterpTruncate()
 * A: parCSR matrix to be modified
 * tol: relative tolerance or truncation factor for dropping small terms
 * max_row_elmts: maximum number of (largest) nonzero elements to keep.
 * rescale: Boolean on whether or not to scale resulting matrix. Scaling for
 * each row satisfies: sum(nonzero values before dropping)/ sum(nonzero values after dropping),
 * this way, the application of the truncated matrix on a constant vector is the same as that of
 * the original matrix.
 * nrm_type: type of norm used for dropping with tol.
 * -- 0 = infinity-norm
 * -- 1 = 1-norm
 * -- 2 = 2-norm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixTruncate(hypre_ParCSRMatrix *A,
                           HYPRE_Real          tol,
                           HYPRE_Int           max_row_elmts,
                           HYPRE_Int           rescale,
                           HYPRE_Int           nrm_type)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] -= hypre_MPI_Wtime();
#endif

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *A_diag_j_new;
   HYPRE_Real *A_diag_data_new;

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int *A_offd_j_new;
   HYPRE_Real *A_offd_data_new;

   HYPRE_Int n_fine = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int i, j, start_j;
   HYPRE_Int ierr = 0;
   HYPRE_Int next_open;
   HYPRE_Int now_checking;
   HYPRE_Int num_lost;
   HYPRE_Int num_lost_global = 0;
   HYPRE_Int next_open_offd;
   HYPRE_Int now_checking_offd;
   HYPRE_Int num_lost_offd;
   HYPRE_Int num_lost_global_offd;
   HYPRE_Int A_diag_size;
   HYPRE_Int A_offd_size;
   HYPRE_Int num_elmts;
   HYPRE_Int cnt, cnt_diag, cnt_offd;
   HYPRE_Real row_nrm;
   HYPRE_Real drop_coeff;
   HYPRE_Real row_sum;
   HYPRE_Real scale;

   HYPRE_MemoryLocation memory_location_diag = hypre_CSRMatrixMemoryLocation(A_diag);
   HYPRE_MemoryLocation memory_location_offd = hypre_CSRMatrixMemoryLocation(A_offd);

   /* Threading variables.  Entry i of num_lost_(offd_)per_thread  holds the
    * number of dropped entries over thread i's row range. Cum_lost_per_thread
    * will temporarily store the cumulative number of dropped entries up to
    * each thread. */
   HYPRE_Int my_thread_num, num_threads, start, stop;
   HYPRE_Int * max_num_threads = hypre_CTAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
   HYPRE_Int * cum_lost_per_thread;
   HYPRE_Int * num_lost_per_thread;
   HYPRE_Int * num_lost_offd_per_thread;

   /* Initialize threading variables */
   max_num_threads[0] = hypre_NumThreads();
   cum_lost_per_thread = hypre_CTAlloc(HYPRE_Int,  max_num_threads[0], HYPRE_MEMORY_HOST);
   num_lost_per_thread = hypre_CTAlloc(HYPRE_Int,  max_num_threads[0], HYPRE_MEMORY_HOST);
   num_lost_offd_per_thread = hypre_CTAlloc(HYPRE_Int,  max_num_threads[0], HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_threads[0]; i++)
   {
      num_lost_per_thread[i] = 0;
      num_lost_offd_per_thread[i] = 0;
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel private(i,my_thread_num,num_threads,row_nrm, drop_coeff,j,start_j,row_sum,scale,num_lost,now_checking,next_open,num_lost_offd,now_checking_offd,next_open_offd,start,stop,cnt_diag,cnt_offd,num_elmts,cnt)
#endif
   {
      my_thread_num = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();

      /* Compute each thread's range of rows to truncate and compress.  Note,
       * that i, j and data are all compressed as entries are dropped, but
       * that the compression only occurs locally over each thread's row
       * range.  A_diag_i is only made globally consistent at the end of this
       * routine.  During the dropping phases, A_diag_i[stop] will point to
       * the start of the next thread's row range.  */

      /* my row range */
      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }

      /*
       * Truncate based on truncation tolerance
       */
      if (tol > 0)
      {
         num_lost = 0;
         num_lost_offd = 0;

         next_open = A_diag_i[start];
         now_checking = A_diag_i[start];
         next_open_offd = A_offd_i[start];;
         now_checking_offd = A_offd_i[start];;

         for (i = start; i < stop; i++)
         {
            row_nrm = 0;
            /* compute norm for dropping small terms */
            if (nrm_type == 0)
            {
               /* infty-norm */
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  row_nrm = (row_nrm < hypre_cabs(A_diag_data[j])) ?
                            hypre_cabs(A_diag_data[j]) : row_nrm;
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  row_nrm = (row_nrm < hypre_cabs(A_offd_data[j])) ?
                            hypre_cabs(A_offd_data[j]) : row_nrm;
               }
            }
            if (nrm_type == 1)
            {
               /* 1-norm */
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  row_nrm += hypre_cabs(A_diag_data[j]);
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  row_nrm += hypre_cabs(A_offd_data[j]);
               }
            }
            if (nrm_type == 2)
            {
               /* 2-norm */
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  HYPRE_Complex v = A_diag_data[j];
                  row_nrm += v * v;
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  HYPRE_Complex v = A_offd_data[j];
                  row_nrm += v * v;
               }
               row_nrm  = hypre_sqrt(row_nrm);
            }
            drop_coeff = tol * row_nrm;

            start_j = A_diag_i[i];
            if (num_lost)
            {
               A_diag_i[i] -= num_lost;
            }
            row_sum = 0;
            scale = 0;
            for (j = start_j; j < A_diag_i[i + 1]; j++)
            {
               row_sum += A_diag_data[now_checking];
               if (hypre_cabs(A_diag_data[now_checking]) < drop_coeff)
               {
                  num_lost++;
                  now_checking++;
               }
               else
               {
                  scale += A_diag_data[now_checking];
                  A_diag_data[next_open] = A_diag_data[now_checking];
                  A_diag_j[next_open] = A_diag_j[now_checking];
                  now_checking++;
                  next_open++;
               }
            }

            start_j = A_offd_i[i];
            if (num_lost_offd)
            {
               A_offd_i[i] -= num_lost_offd;
            }

            for (j = start_j; j < A_offd_i[i + 1]; j++)
            {
               row_sum += A_offd_data[now_checking_offd];
               if (hypre_cabs(A_offd_data[now_checking_offd]) < drop_coeff)
               {
                  num_lost_offd++;
                  now_checking_offd++;
               }
               else
               {
                  scale += A_offd_data[now_checking_offd];
                  A_offd_data[next_open_offd] = A_offd_data[now_checking_offd];
                  A_offd_j[next_open_offd] = A_offd_j[now_checking_offd];
                  now_checking_offd++;
                  next_open_offd++;
               }
            }

            /* scale row of A */
            if (rescale && scale != 0.)
            {
               if (scale != row_sum)
               {
                  scale = row_sum / scale;
                  for (j = A_diag_i[i]; j < (A_diag_i[i + 1] - num_lost); j++)
                  {
                     A_diag_data[j] *= scale;
                  }
                  for (j = A_offd_i[i]; j < (A_offd_i[i + 1] - num_lost_offd); j++)
                  {
                     A_offd_data[j] *= scale;
                  }
               }
            }
         } /* end loop for (i = 0; i < n_fine; i++) */

         /* store number of dropped elements and number of threads */
         if (my_thread_num == 0)
         {
            max_num_threads[0] = num_threads;
         }
         num_lost_per_thread[my_thread_num] = num_lost;
         num_lost_offd_per_thread[my_thread_num] = num_lost_offd;

      } /* end if (trunc_factor > 0) */

      /*
       * Truncate based on capping the nnz per row
       *
       */
      if (max_row_elmts > 0)
      {
         HYPRE_Int A_mxnum, cnt1, last_index, last_index_offd;
         HYPRE_Int *A_aux_j;
         HYPRE_Real *A_aux_data;

         /* find maximum row length locally over this row range */
         A_mxnum = 0;
         for (i = start; i < stop; i++)
         {
            /* Note A_diag_i[stop] is the starting point for the next thread
             * in j and data, not the stop point for this thread */
            last_index = A_diag_i[i + 1];
            last_index_offd = A_offd_i[i + 1];
            if (i == stop - 1)
            {
               last_index -= num_lost_per_thread[my_thread_num];
               last_index_offd -= num_lost_offd_per_thread[my_thread_num];
            }
            cnt1 = last_index - A_diag_i[i] + last_index_offd - A_offd_i[i];
            if (cnt1 > A_mxnum)
            {
               A_mxnum = cnt1;
            }
         }

         /* Some rows exceed max_row_elmts, and require truncation.  Essentially,
          * each thread truncates and compresses its range of rows locally. */
         if (A_mxnum > max_row_elmts)
         {
            num_lost = 0;
            num_lost_offd = 0;

            /* two temporary arrays to hold row i for temporary operations */
            A_aux_j = hypre_CTAlloc(HYPRE_Int,  A_mxnum, HYPRE_MEMORY_HOST);
            A_aux_data = hypre_CTAlloc(HYPRE_Real,  A_mxnum, HYPRE_MEMORY_HOST);
            cnt_diag = A_diag_i[start];
            cnt_offd = A_offd_i[start];

            for (i = start; i < stop; i++)
            {
               /* Note A_diag_i[stop] is the starting point for the next thread
                * in j and data, not the stop point for this thread */
               last_index = A_diag_i[i + 1];
               last_index_offd = A_offd_i[i + 1];
               if (i == stop - 1)
               {
                  last_index -= num_lost_per_thread[my_thread_num];
                  last_index_offd -= num_lost_offd_per_thread[my_thread_num];
               }

               row_sum = 0;
               num_elmts = last_index - A_diag_i[i] + last_index_offd - A_offd_i[i];
               if (max_row_elmts < num_elmts)
               {
                  /* copy both diagonal and off-diag parts of row i to _aux_ arrays */
                  cnt = 0;
                  for (j = A_diag_i[i]; j < last_index; j++)
                  {
                     A_aux_j[cnt] = A_diag_j[j];
                     A_aux_data[cnt++] = A_diag_data[j];
                     row_sum += A_diag_data[j];
                  }
                  num_lost += cnt;
                  cnt1 = cnt;
                  for (j = A_offd_i[i]; j < last_index_offd; j++)
                  {
                     A_aux_j[cnt] = A_offd_j[j] + num_cols;
                     A_aux_data[cnt++] = A_offd_data[j];
                     row_sum += A_offd_data[j];
                  }
                  num_lost_offd += cnt - cnt1;

                  /* sort data */
                  hypre_qsort2_abs(A_aux_j, A_aux_data, 0, cnt - 1);
                  scale = 0;
                  if (i > start)
                  {
                     A_diag_i[i] = cnt_diag;
                     A_offd_i[i] = cnt_offd;
                  }
                  for (j = 0; j < max_row_elmts; j++)
                  {
                     scale += A_aux_data[j];
                     if (A_aux_j[j] < num_cols)
                     {
                        A_diag_j[cnt_diag] = A_aux_j[j];
                        A_diag_data[cnt_diag++] = A_aux_data[j];
                     }
                     else
                     {
                        A_offd_j[cnt_offd] = A_aux_j[j] - num_cols;
                        A_offd_data[cnt_offd++] = A_aux_data[j];
                     }
                  }
                  num_lost -= cnt_diag - A_diag_i[i];
                  num_lost_offd -= cnt_offd - A_offd_i[i];

                  /* scale row of A */
                  if (rescale && (scale != 0.))
                  {
                     if (scale != row_sum)
                     {
                        scale = row_sum / scale;
                        for (j = A_diag_i[i]; j < cnt_diag; j++)
                        {
                           A_diag_data[j] *= scale;
                        }
                        for (j = A_offd_i[i]; j < cnt_offd; j++)
                        {
                           A_offd_data[j] *= scale;
                        }
                     }
                  }
               }  /* end if (max_row_elmts < num_elmts) */
               else
               {
                  /* nothing dropped from this row, but still have to shift entries back
                   * by the number dropped so far */
                  if (A_diag_i[i] != cnt_diag)
                  {
                     start_j = A_diag_i[i];
                     A_diag_i[i] = cnt_diag;
                     for (j = start_j; j < last_index; j++)
                     {
                        A_diag_j[cnt_diag] = A_diag_j[j];
                        A_diag_data[cnt_diag++] = A_diag_data[j];
                     }
                  }
                  else
                  {
                     cnt_diag += last_index - A_diag_i[i];
                  }

                  if (A_offd_i[i] != cnt_offd)
                  {
                     start_j = A_offd_i[i];
                     A_offd_i[i] = cnt_offd;
                     for (j = start_j; j < last_index_offd; j++)
                     {
                        A_offd_j[cnt_offd] = A_offd_j[j];
                        A_offd_data[cnt_offd++] = A_offd_data[j];
                     }
                  }
                  else
                  {
                     cnt_offd += last_index_offd - A_offd_i[i];
                  }
               }
            } /* end for (i = 0; i < n_fine; i++) */

            num_lost_per_thread[my_thread_num] += num_lost;
            num_lost_offd_per_thread[my_thread_num] += num_lost_offd;
            hypre_TFree(A_aux_j, HYPRE_MEMORY_HOST);
            hypre_TFree(A_aux_data, HYPRE_MEMORY_HOST);

         } /* end if (A_mxnum > max_row_elmts) */
      } /* end if (max_row_elmts > 0) */


      /* Sum up num_lost_global */
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         num_lost_global = 0;
         num_lost_global_offd = 0;
         for (i = 0; i < max_num_threads[0]; i++)
         {
            num_lost_global += num_lost_per_thread[i];
            num_lost_global_offd += num_lost_offd_per_thread[i];
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /*
       * Synchronize and create new diag data structures
       */
      if (num_lost_global)
      {
         /* Each thread has it's own locally compressed CSR matrix from rows start
          * to stop.  Now, we have to copy each thread's chunk into the new
          * process-wide CSR data structures
          *
          * First, we compute the new process-wide number of nonzeros (i.e.,
          * A_diag_size), and compute cum_lost_per_thread[k] so that this
          * entry holds the cumulative sum of entries dropped up to and
          * including thread k. */
         if (my_thread_num == 0)
         {
            A_diag_size = A_diag_i[n_fine];

            for (i = 0; i < max_num_threads[0]; i++)
            {
               A_diag_size -= num_lost_per_thread[i];
               if (i > 0)
               {
                  cum_lost_per_thread[i] = num_lost_per_thread[i] + cum_lost_per_thread[i - 1];
               }
               else
               {
                  cum_lost_per_thread[i] = num_lost_per_thread[i];
               }
            }

            A_diag_j_new = hypre_CTAlloc(HYPRE_Int, A_diag_size, memory_location_diag);
            A_diag_data_new = hypre_CTAlloc(HYPRE_Real, A_diag_size, memory_location_diag);
         }
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /* points to next open spot in new data structures for this thread */
         if (my_thread_num == 0)
         {
            next_open = 0;
         }
         else
         {
            /* remember, cum_lost_per_thread[k] stores the num dropped up to and
             * including thread k */
            next_open = A_diag_i[start] - cum_lost_per_thread[my_thread_num - 1];
         }

         /* copy the j and data arrays over */
         for (i = A_diag_i[start]; i < A_diag_i[stop] - num_lost_per_thread[my_thread_num]; i++)
         {
            A_diag_j_new[next_open] = A_diag_j[i];
            A_diag_data_new[next_open] = A_diag_data[i];
            next_open += 1;
         }

#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         /* update A_diag_i with number of dropped entries by all lower ranked
          * threads */
         if (my_thread_num > 0)
         {
            for (i = start; i < stop; i++)
            {
               A_diag_i[i] -= cum_lost_per_thread[my_thread_num - 1];
            }
         }

         if (my_thread_num == 0)
         {
            /* Set last entry */
            A_diag_i[n_fine] = A_diag_size ;

            hypre_TFree(A_diag_j, memory_location_diag);
            hypre_TFree(A_diag_data, memory_location_diag);
            hypre_CSRMatrixJ(A_diag) = A_diag_j_new;
            hypre_CSRMatrixData(A_diag) = A_diag_data_new;
            hypre_CSRMatrixNumNonzeros(A_diag) = A_diag_size;
         }
      }

      /*
       * Synchronize and create new offd data structures
       */
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (num_lost_global_offd)
      {
         /* Repeat process for off-diagonal */
         if (my_thread_num == 0)
         {
            A_offd_size = A_offd_i[n_fine];
            for (i = 0; i < max_num_threads[0]; i++)
            {
               A_offd_size -= num_lost_offd_per_thread[i];
               if (i > 0)
               {
                  cum_lost_per_thread[i] = num_lost_offd_per_thread[i] + cum_lost_per_thread[i - 1];
               }
               else
               {
                  cum_lost_per_thread[i] = num_lost_offd_per_thread[i];
               }
            }

            A_offd_j_new = hypre_CTAlloc(HYPRE_Int, A_offd_size, memory_location_offd);
            A_offd_data_new = hypre_CTAlloc(HYPRE_Real, A_offd_size, memory_location_offd);
         }
#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /* points to next open spot in new data structures for this thread */
         if (my_thread_num == 0)
         {
            next_open = 0;
         }
         else
         {
            /* remember, cum_lost_per_thread[k] stores the num dropped up to and
             * including thread k */
            next_open = A_offd_i[start] - cum_lost_per_thread[my_thread_num - 1];
         }

         /* copy the j and data arrays over */
         for (i = A_offd_i[start]; i < A_offd_i[stop] - num_lost_offd_per_thread[my_thread_num]; i++)
         {
            A_offd_j_new[next_open] = A_offd_j[i];
            A_offd_data_new[next_open] = A_offd_data[i];
            next_open += 1;
         }

#ifdef HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         /* update A_offd_i with number of dropped entries by all lower ranked
          * threads */
         if (my_thread_num > 0)
         {
            for (i = start; i < stop; i++)
            {
               A_offd_i[i] -= cum_lost_per_thread[my_thread_num - 1];
            }
         }

         if (my_thread_num == 0)
         {
            /* Set last entry */
            A_offd_i[n_fine] = A_offd_size ;

            hypre_TFree(A_offd_j, memory_location_offd);
            hypre_TFree(A_offd_data, memory_location_offd);
            hypre_CSRMatrixJ(A_offd) = A_offd_j_new;
            hypre_CSRMatrixData(A_offd) = A_offd_data_new;
            hypre_CSRMatrixNumNonzeros(A_offd) = A_offd_size;
         }
      }

   } /* end parallel region */

   hypre_TFree(max_num_threads, HYPRE_MEMORY_HOST);
   hypre_TFree(cum_lost_per_thread, HYPRE_MEMORY_HOST);
   hypre_TFree(num_lost_per_thread, HYPRE_MEMORY_HOST);
   hypre_TFree(num_lost_offd_per_thread, HYPRE_MEMORY_HOST);

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_INTERP_TRUNC] += hypre_MPI_Wtime();
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixSetConstantValues( hypre_ParCSRMatrix *A,
                                     HYPRE_Complex       value )
{
   hypre_CSRMatrixSetConstantValues(hypre_ParCSRMatrixDiag(A), value);
   hypre_CSRMatrixSetConstantValues(hypre_ParCSRMatrixOffd(A), value);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCopyColMapOffdToDevice
 *--------------------------------------------------------------------------*/

void
hypre_ParCSRMatrixCopyColMapOffdToDevice(hypre_ParCSRMatrix *A)
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   if (hypre_ParCSRMatrixDeviceColMapOffd(A) == NULL)
   {
      const HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
      hypre_ParCSRMatrixDeviceColMapOffd(A) = hypre_TAlloc(HYPRE_BigInt,
                                                           num_cols_A_offd,
                                                           HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(hypre_ParCSRMatrixDeviceColMapOffd(A),
                    hypre_ParCSRMatrixColMapOffd(A),
                    HYPRE_BigInt,
                    num_cols_A_offd,
                    HYPRE_MEMORY_DEVICE,
                    HYPRE_MEMORY_HOST);
   }
#else
   HYPRE_UNUSED_VAR(A);
#endif
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixCopyColMapOffdToHost
 *--------------------------------------------------------------------------*/

void
hypre_ParCSRMatrixCopyColMapOffdToHost(hypre_ParCSRMatrix *A)
{
#if defined(HYPRE_USING_GPU)
   if (hypre_ParCSRMatrixColMapOffd(A) == NULL)
   {
      const HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
      hypre_ParCSRMatrixColMapOffd(A) = hypre_TAlloc(HYPRE_BigInt,
                                                     num_cols_A_offd,
                                                     HYPRE_MEMORY_HOST);
      hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(A),
                    hypre_ParCSRMatrixDeviceColMapOffd(A),
                    HYPRE_BigInt,
                    num_cols_A_offd,
                    HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
   }
#else
   HYPRE_UNUSED_VAR(A);
#endif
}
