/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_ParCSRBlockMatrix class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_block_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixCreate( MPI_Comm      comm,
                               HYPRE_Int     block_size,
                               HYPRE_BigInt  global_num_rows,
                               HYPRE_BigInt  global_num_cols,
                               HYPRE_BigInt *row_starts_in,
                               HYPRE_BigInt *col_starts_in,
                               HYPRE_Int     num_cols_offd,
                               HYPRE_Int     num_nonzeros_diag,
                               HYPRE_Int     num_nonzeros_offd )
{
   hypre_ParCSRBlockMatrix  *matrix;
   HYPRE_Int       num_procs, my_id;
   HYPRE_Int       local_num_rows;
   HYPRE_Int       local_num_cols;
   HYPRE_BigInt    first_row_index, first_col_diag;
   HYPRE_BigInt    row_starts[2];
   HYPRE_BigInt    col_starts[2];

   matrix = hypre_CTAlloc(hypre_ParCSRBlockMatrix, 1, HYPRE_MEMORY_HOST);

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
   local_num_rows = (HYPRE_Int)(row_starts[1] - first_row_index) ;
   first_col_diag = col_starts[0];
   local_num_cols = (HYPRE_Int)(col_starts[1] - first_col_diag);
   hypre_ParCSRBlockMatrixComm(matrix) = comm;
   hypre_ParCSRBlockMatrixDiag(matrix) =
      hypre_CSRBlockMatrixCreate(block_size, local_num_rows,
                                 local_num_cols, num_nonzeros_diag);
   hypre_ParCSRBlockMatrixOffd(matrix) =
      hypre_CSRBlockMatrixCreate(block_size, local_num_rows,
                                 num_cols_offd, num_nonzeros_offd);

   hypre_ParCSRBlockMatrixBlockSize(matrix)     = block_size;
   hypre_ParCSRBlockMatrixGlobalNumRows(matrix) = global_num_rows;
   hypre_ParCSRBlockMatrixGlobalNumCols(matrix) = global_num_cols;
   hypre_ParCSRBlockMatrixFirstRowIndex(matrix) = first_row_index;
   hypre_ParCSRBlockMatrixFirstColDiag(matrix)  = first_col_diag;
   hypre_ParCSRBlockMatrixLastRowIndex(matrix)  = first_row_index + (HYPRE_BigInt)local_num_rows - 1;
   hypre_ParCSRBlockMatrixLastColDiag(matrix)   = first_col_diag  + (HYPRE_BigInt)local_num_cols - 1;
   hypre_ParCSRBlockMatrixRowStarts(matrix)[0]  = row_starts[0];
   hypre_ParCSRBlockMatrixRowStarts(matrix)[1]  = row_starts[1];
   hypre_ParCSRBlockMatrixColStarts(matrix)[0]  = col_starts[0];
   hypre_ParCSRBlockMatrixColStarts(matrix)[1]  = col_starts[1];
   hypre_ParCSRBlockMatrixColMapOffd(matrix)    = NULL;
   hypre_ParCSRBlockMatrixCommPkg(matrix)       = NULL;
   hypre_ParCSRBlockMatrixCommPkgT(matrix)      = NULL;
   hypre_ParCSRBlockMatrixAssumedPartition(matrix) = NULL;

   /* set defaults */
   hypre_ParCSRBlockMatrixOwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixDestroy( hypre_ParCSRBlockMatrix *matrix )
{

   if (matrix)
   {
      if ( hypre_ParCSRBlockMatrixOwnsData(matrix) )
      {
         hypre_CSRBlockMatrixDestroy(hypre_ParCSRBlockMatrixDiag(matrix));
         hypre_CSRBlockMatrixDestroy(hypre_ParCSRBlockMatrixOffd(matrix));
         if (hypre_ParCSRBlockMatrixColMapOffd(matrix))
         {
            hypre_TFree(hypre_ParCSRBlockMatrixColMapOffd(matrix), HYPRE_MEMORY_HOST);
         }
         if (hypre_ParCSRBlockMatrixCommPkg(matrix))
         {
            hypre_MatvecCommPkgDestroy(hypre_ParCSRBlockMatrixCommPkg(matrix));
         }
         if (hypre_ParCSRBlockMatrixCommPkgT(matrix))
         {
            hypre_MatvecCommPkgDestroy(hypre_ParCSRBlockMatrixCommPkgT(matrix));
         }
      }

      if (hypre_ParCSRBlockMatrixAssumedPartition(matrix))
      {
         hypre_ParCSRBlockMatrixDestroyAssumedPartition(matrix);
      }

      hypre_TFree(matrix, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixInitialize( hypre_ParCSRBlockMatrix *matrix )
{
   HYPRE_Int  ierr = 0;

   hypre_CSRBlockMatrixInitialize(hypre_ParCSRBlockMatrixDiag(matrix));
   hypre_CSRBlockMatrixInitialize(hypre_ParCSRBlockMatrixOffd(matrix));
   hypre_ParCSRBlockMatrixColMapOffd(matrix) = hypre_CTAlloc(HYPRE_BigInt,
                                                             hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixOffd(matrix)), HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetNumNonzeros
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixSetNumNonzeros( hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
   HYPRE_Int *diag_i = hypre_CSRBlockMatrixI(diag);
   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
   HYPRE_Int *offd_i = hypre_CSRBlockMatrixI(offd);
   HYPRE_Int local_num_rows = hypre_CSRBlockMatrixNumRows(diag);
   HYPRE_BigInt total_num_nonzeros;
   HYPRE_BigInt local_num_nonzeros;
   HYPRE_Int ierr = 0;

   local_num_nonzeros = (HYPRE_BigInt)(diag_i[local_num_rows] + offd_i[local_num_rows]);
   hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, HYPRE_MPI_BIG_INT,
                       hypre_MPI_SUM, comm);
   hypre_ParCSRBlockMatrixNumNonzeros(matrix) = total_num_nonzeros;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetDNumNonzeros
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixSetDNumNonzeros( hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
   HYPRE_Int *diag_i = hypre_CSRBlockMatrixI(diag);
   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
   HYPRE_Int *offd_i = hypre_CSRBlockMatrixI(offd);
   HYPRE_Int local_num_rows = hypre_CSRBlockMatrixNumRows(diag);
   HYPRE_Real total_num_nonzeros;
   HYPRE_Real local_num_nonzeros;
   HYPRE_Int ierr = 0;

   local_num_nonzeros = (HYPRE_Real) diag_i[local_num_rows] + (HYPRE_Real) offd_i[local_num_rows];
   hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1,
                       HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRBlockMatrixDNumNonzeros(matrix) = total_num_nonzeros;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRBlockMatrixSetDataOwner( hypre_ParCSRBlockMatrix *matrix,
                                     HYPRE_Int              owns_data )
{
   HYPRE_Int    ierr = 0;

   hypre_ParCSRBlockMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixCompress
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixCompress( hypre_ParCSRBlockMatrix *matrix )
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
   HYPRE_BigInt global_num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(matrix);
   HYPRE_BigInt global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
   HYPRE_BigInt *row_starts = hypre_ParCSRBlockMatrixRowStarts(matrix);
   HYPRE_BigInt *col_starts = hypre_ParCSRBlockMatrixColStarts(matrix);
   HYPRE_Int num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
   HYPRE_Int num_nonzeros_diag = hypre_CSRBlockMatrixNumNonzeros(diag);
   HYPRE_Int num_nonzeros_offd = hypre_CSRBlockMatrixNumNonzeros(offd);

   hypre_ParCSRMatrix *matrix_C;

   HYPRE_Int i;

   matrix_C = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                       row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   hypre_ParCSRMatrixInitialize(matrix_C);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
   hypre_ParCSRMatrixDiag(matrix_C) = hypre_CSRBlockMatrixCompress(diag);
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
   hypre_ParCSRMatrixOffd(matrix_C) = hypre_CSRBlockMatrixCompress(offd);

   for (i = 0; i < num_cols_offd; i++)
      hypre_ParCSRMatrixColMapOffd(matrix_C)[i] =
         hypre_ParCSRBlockMatrixColMapOffd(matrix)[i];
   return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixConvertToParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix *
hypre_ParCSRBlockMatrixConvertToParCSRMatrix(hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(matrix);
   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(matrix);
   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(matrix);
   HYPRE_Int block_size = hypre_ParCSRBlockMatrixBlockSize(matrix);
   HYPRE_BigInt global_num_rows = hypre_ParCSRBlockMatrixGlobalNumRows(matrix);
   HYPRE_BigInt global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
   HYPRE_BigInt *row_starts = hypre_ParCSRBlockMatrixRowStarts(matrix);
   HYPRE_BigInt *col_starts = hypre_ParCSRBlockMatrixColStarts(matrix);
   HYPRE_Int num_cols_offd = hypre_CSRBlockMatrixNumCols(offd);
   HYPRE_Int num_nonzeros_diag = hypre_CSRBlockMatrixNumNonzeros(diag);
   HYPRE_Int num_nonzeros_offd = hypre_CSRBlockMatrixNumNonzeros(offd);

   hypre_ParCSRMatrix *matrix_C;
   HYPRE_BigInt matrix_C_row_starts[2];
   HYPRE_BigInt matrix_C_col_starts[2];

   HYPRE_Int *counter, *new_j_map;
   HYPRE_Int size_j, size_map, index, new_num_cols, removed = 0;
   HYPRE_Int *offd_j;
   HYPRE_BigInt *col_map_offd, *new_col_map_offd;


   HYPRE_Int num_procs, i, j;

   hypre_CSRMatrix *diag_nozeros, *offd_nozeros;

   hypre_MPI_Comm_size(comm, &num_procs);

   for (i = 0; i < 2; i++)
   {
      matrix_C_row_starts[i] = row_starts[i] * (HYPRE_BigInt)block_size;
      matrix_C_col_starts[i] = col_starts[i] * (HYPRE_BigInt)block_size;
   }

   matrix_C = hypre_ParCSRMatrixCreate(comm, global_num_rows * (HYPRE_BigInt)block_size,
                                       global_num_cols * (HYPRE_BigInt)block_size,
                                       matrix_C_row_starts,
                                       matrix_C_col_starts,
                                       num_cols_offd * block_size,
                                       num_nonzeros_diag * block_size * block_size,
                                       num_nonzeros_offd * block_size * block_size);
   hypre_ParCSRMatrixInitialize(matrix_C);

   /* DIAG */
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
   hypre_ParCSRMatrixDiag(matrix_C) =
      hypre_CSRBlockMatrixConvertToCSRMatrix(diag);

   /* AB - added to delete zeros */
   diag_nozeros = hypre_CSRMatrixDeleteZeros(
                     hypre_ParCSRMatrixDiag(matrix_C), 1e-14);
   if (diag_nozeros)
   {
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
      hypre_ParCSRMatrixDiag(matrix_C) = diag_nozeros;
   }

   /* OFF-DIAG */
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
   hypre_ParCSRMatrixOffd(matrix_C) =
      hypre_CSRBlockMatrixConvertToCSRMatrix(offd);

   /* AB - added to delete zeros - this just deletes from data and j arrays */
   offd_nozeros = hypre_CSRMatrixDeleteZeros(
                     hypre_ParCSRMatrixOffd(matrix_C), 1e-14);
   if (offd_nozeros)
   {
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
      hypre_ParCSRMatrixOffd(matrix_C) = offd_nozeros;
      removed = 1;

   }

   /* now convert the col_map_offd */
   for (i = 0; i < num_cols_offd; i++)
      for (j = 0; j < block_size; j++)
         hypre_ParCSRMatrixColMapOffd(matrix_C)[i * block_size + j] =
            hypre_ParCSRBlockMatrixColMapOffd(matrix)[i] * (HYPRE_BigInt)block_size + (HYPRE_BigInt)j;

   /* if we deleted zeros, then it is possible that col_map_offd can be
      compressed as well - this requires some amount of work that could be skipped... */

   if (removed)
   {
      size_map =   num_cols_offd * block_size;
      counter = hypre_CTAlloc(HYPRE_Int,  size_map, HYPRE_MEMORY_HOST);
      new_j_map = hypre_CTAlloc(HYPRE_Int,  size_map, HYPRE_MEMORY_HOST);

      offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(matrix_C));
      col_map_offd = hypre_ParCSRMatrixColMapOffd(matrix_C);

      size_j = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(matrix_C));
      /* mark which off_d entries are found in j */
      for (i = 0; i < size_j; i++)
      {
         counter[offd_j[i]] = 1;
      }
      /*now find new numbering for columns (we will delete the
        cols where counter = 0*/
      index = 0;
      for (i = 0; i < size_map; i++)
      {
         if (counter[i]) { new_j_map[i] = index++; }
      }
      new_num_cols = index;
      /* if there are some col entries to remove: */
      if (!(index == size_map))
      {
         /* go thru j and adjust entries */
         for (i = 0; i < size_j; i++)
         {
            offd_j[i] = new_j_map[offd_j[i]];
         }
         /*now go thru col map and get rid of non-needed entries */
         new_col_map_offd = hypre_CTAlloc(HYPRE_BigInt,  new_num_cols, HYPRE_MEMORY_HOST);
         index = 0;
         for (i = 0; i < size_map; i++)
         {
            if (counter[i])
            {
               new_col_map_offd[index++] = col_map_offd[i];
            }
         }
         /* set the new col map */
         hypre_TFree(col_map_offd, HYPRE_MEMORY_HOST);
         hypre_ParCSRMatrixColMapOffd(matrix_C) = new_col_map_offd;
         /* modify the number of cols */
         hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matrix_C)) = new_num_cols;
      }
      hypre_TFree(new_j_map, HYPRE_MEMORY_HOST);
      hypre_TFree(counter, HYPRE_MEMORY_HOST);

   }

   hypre_ParCSRMatrixSetNumNonzeros( matrix_C );
   hypre_ParCSRMatrixSetDNumNonzeros( matrix_C );

   /* we will not copy the comm package */
   hypre_ParCSRMatrixCommPkg(matrix_C) = NULL;

   return matrix_C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixConvertFromParCSRMatrix
 *--------------------------------------------------------------------------*/

hypre_ParCSRBlockMatrix *
hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(hypre_ParCSRMatrix *matrix,
                                               HYPRE_Int matrix_C_block_size )
{
   MPI_Comm comm = hypre_ParCSRMatrixComm(matrix);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(matrix);
   HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(matrix);
   HYPRE_BigInt global_num_cols = hypre_ParCSRMatrixGlobalNumCols(matrix);
   HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(matrix);
   HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(matrix);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_BigInt *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(matrix);
   HYPRE_BigInt *map_to_node = NULL;
   HYPRE_Int *counter = NULL, *col_in_j_map = NULL;
   HYPRE_BigInt *matrix_C_col_map_offd = NULL;

   HYPRE_Int matrix_C_num_cols_offd;
   HYPRE_Int matrix_C_num_nonzeros_offd;
   HYPRE_Int num_rows, num_nodes;

   HYPRE_Int *offd_i        = hypre_CSRMatrixI(offd);
   HYPRE_Int *offd_j        = hypre_CSRMatrixJ(offd);
   HYPRE_Complex * offd_data = hypre_CSRMatrixData(offd);

   hypre_ParCSRBlockMatrix *matrix_C;
   HYPRE_BigInt matrix_C_row_starts[2];
   HYPRE_BigInt matrix_C_col_starts[2];
   hypre_CSRBlockMatrix *matrix_C_diag;
   hypre_CSRBlockMatrix *matrix_C_offd;

   HYPRE_Int *matrix_C_offd_i = NULL, *matrix_C_offd_j = NULL;
   HYPRE_Complex *matrix_C_offd_data = NULL;

   HYPRE_Int num_procs, i, j, k, k_map, count, index, start_index, pos, row;

   hypre_MPI_Comm_size(comm, &num_procs);

   for (i = 0; i < 2; i++)
   {
      matrix_C_row_starts[i] = row_starts[i] / (HYPRE_BigInt)matrix_C_block_size;
      matrix_C_col_starts[i] = col_starts[i] / (HYPRE_BigInt)matrix_C_block_size;
   }

   /************* create the diagonal part ************/
   matrix_C_diag = hypre_CSRBlockMatrixConvertFromCSRMatrix(diag,
                                                            matrix_C_block_size);

   /*******  the offd part *******************/

   /* can't use the same function for the offd part - because this isn't square
      and the offd j entries aren't global numbering (have to consider the offd
      map) - need to look at col_map_offd first */

   /* figure out the new number of offd columns (num rows is same as diag) */
   num_cols_offd = hypre_CSRMatrixNumCols(offd);
   num_rows = hypre_CSRMatrixNumRows(diag);
   num_nodes =  num_rows / matrix_C_block_size;

   matrix_C_offd_i = hypre_CTAlloc(HYPRE_Int,  num_nodes + 1, HYPRE_MEMORY_HOST);

   matrix_C_num_cols_offd = 0;
   matrix_C_offd_i[0] = 0;
   matrix_C_num_nonzeros_offd = 0;

   if (num_cols_offd)
   {
      map_to_node = hypre_CTAlloc(HYPRE_BigInt,  num_cols_offd, HYPRE_MEMORY_HOST);
      matrix_C_num_cols_offd = 1;
      map_to_node[0] = col_map_offd[0] / (HYPRE_BigInt)matrix_C_block_size;
      for (i = 1; i < num_cols_offd; i++)
      {
         map_to_node[i] = col_map_offd[i] / (HYPRE_BigInt)matrix_C_block_size;
         if (map_to_node[i] > map_to_node[i - 1]) { matrix_C_num_cols_offd++; }
      }

      matrix_C_col_map_offd = hypre_CTAlloc(HYPRE_BigInt,  matrix_C_num_cols_offd, HYPRE_MEMORY_HOST);
      col_in_j_map = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);

      matrix_C_col_map_offd[0] = map_to_node[0];
      col_in_j_map[0] = 0;
      count = 1;
      j = 1;

      /* fill in the col_map_off_d - these are global numbers.  Then we need to
         map these to j entries (these have local numbers) */
      for (i = 1; i < num_cols_offd; i++)
      {
         if (map_to_node[i] > map_to_node[i - 1])
         {
            matrix_C_col_map_offd[count++] = map_to_node[i];
         }
         col_in_j_map[j++] = count - 1;
      }

      /* now figure the nonzeros */
      matrix_C_num_nonzeros_offd = 0;
      counter = hypre_CTAlloc(HYPRE_Int,  matrix_C_num_cols_offd, HYPRE_MEMORY_HOST);
      for (i = 0; i < matrix_C_num_cols_offd; i++)
      {
         counter[i] = -1;
      }

      for (i = 0; i < num_nodes; i++) /* for each block row */
      {
         matrix_C_offd_i[i] = matrix_C_num_nonzeros_offd;
         for (j = 0; j < matrix_C_block_size; j++)
         {
            row = i * matrix_C_block_size + j;
            for (k = offd_i[row]; k < offd_i[row + 1]; k++) /* go through single row */
            {
               k_map = col_in_j_map[offd_j[k]]; /*nodal col - see if this has
                                                  been in this block row (i)
                                                  already*/

               if (counter[k_map] < i) /* not yet counted for this nodal row */
               {
                  counter[k_map] = i;
                  matrix_C_num_nonzeros_offd++;
               }
            }
         }
      }
      /* fill in final i entry */
      matrix_C_offd_i[num_nodes] = matrix_C_num_nonzeros_offd;
   }

   /* create offd matrix */
   matrix_C_offd = hypre_CSRBlockMatrixCreate(matrix_C_block_size, num_nodes,
                                              matrix_C_num_cols_offd,
                                              matrix_C_num_nonzeros_offd);

   /* assign i */
   hypre_CSRBlockMatrixI(matrix_C_offd) = matrix_C_offd_i;


   /* create (and allocate j and data) */
   if (matrix_C_num_nonzeros_offd)
   {
      matrix_C_offd_j = hypre_CTAlloc(HYPRE_Int,  matrix_C_num_nonzeros_offd, HYPRE_MEMORY_HOST);
      matrix_C_offd_data =
         hypre_CTAlloc(HYPRE_Complex,
                       matrix_C_num_nonzeros_offd * matrix_C_block_size *
                       matrix_C_block_size, HYPRE_MEMORY_HOST);
      hypre_CSRBlockMatrixJ(matrix_C_offd) = matrix_C_offd_j;
      hypre_CSRMatrixData(matrix_C_offd) = matrix_C_offd_data;

      for (i = 0; i < matrix_C_num_cols_offd; i++)
      {
         counter[i] = -1;
      }

      index = 0; /*keep track of entry in matrix_C_offd_j*/
      start_index = 0;
      for (i = 0; i < num_nodes; i++) /* for each block row */
      {

         for (j = 0; j < matrix_C_block_size; j++) /* for each row in block */
         {
            row = i * matrix_C_block_size + j;
            for (k = offd_i[row]; k < offd_i[row + 1]; k++) /* go through single row's cols */
            {
               k_map = col_in_j_map[offd_j[k]]; /*nodal col  for off_d */
               if (counter[k_map] < start_index) /* not yet counted for this nodal row */
               {
                  counter[k_map] = index;
                  matrix_C_offd_j[index] = k_map;
                  /*copy the data: which position (corresponds to j array) + which row + which col */
                  pos =  (index * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                         (HYPRE_Int)(col_map_offd[offd_j[k]] % (HYPRE_BigInt)matrix_C_block_size);
                  matrix_C_offd_data[pos] = offd_data[k];
                  index ++;
               }
               else  /* this col has already been listed for this row */
               {

                  /*copy the data: which position (corresponds to j array) + which row + which col */
                  pos =  (counter[k_map] * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                         (HYPRE_Int)(col_map_offd[offd_j[k]] % (HYPRE_BigInt)(matrix_C_block_size));
                  matrix_C_offd_data[pos] = offd_data[k];
               }
            }
         }
         start_index = index; /* first index for current nodal row */
      }
   }

   /* *********create the new matrix  *************/
   matrix_C = hypre_ParCSRBlockMatrixCreate(comm, matrix_C_block_size,
                                            global_num_rows / (HYPRE_BigInt)matrix_C_block_size,
                                            global_num_cols / (HYPRE_BigInt)matrix_C_block_size,
                                            matrix_C_row_starts,
                                            matrix_C_col_starts,
                                            matrix_C_num_cols_offd,
                                            hypre_CSRBlockMatrixNumNonzeros(matrix_C_diag),
                                            matrix_C_num_nonzeros_offd);

   /* use the diag and off diag matrices we have already created */
   hypre_CSRBlockMatrixDestroy(hypre_ParCSRMatrixDiag(matrix_C));
   hypre_ParCSRBlockMatrixDiag(matrix_C) = matrix_C_diag;
   hypre_CSRBlockMatrixDestroy(hypre_ParCSRMatrixOffd(matrix_C));
   hypre_ParCSRBlockMatrixOffd(matrix_C) = matrix_C_offd;

   hypre_ParCSRMatrixColMapOffd(matrix_C) = matrix_C_col_map_offd;

   /* *********don't bother to copy the comm_pkg *************/

   hypre_ParCSRBlockMatrixCommPkg(matrix_C) = NULL;

   /* CLEAN UP !!!! */
   hypre_TFree(map_to_node, HYPRE_MEMORY_HOST);
   hypre_TFree(col_in_j_map, HYPRE_MEMORY_HOST);
   hypre_TFree(counter, HYPRE_MEMORY_HOST);

   return matrix_C;
}

/* ----------------------------------------------------------------------
 * hypre_BlockMatvecCommPkgCreate
 * ---------------------------------------------------------------------*/

HYPRE_Int
hypre_BlockMatvecCommPkgCreate(hypre_ParCSRBlockMatrix *A)
{
   HYPRE_Int        num_recvs, *recv_procs, *recv_vec_starts;
   HYPRE_Int        num_sends, *send_procs, *send_map_starts;
   HYPRE_Int       *send_map_elmts;

   HYPRE_Int        num_cols_off_d;
   HYPRE_BigInt    *col_map_off_d;

   HYPRE_BigInt     first_col_diag;
   HYPRE_BigInt     global_num_cols;

   MPI_Comm         comm;

   hypre_ParCSRCommPkg   *comm_pkg = NULL;
   hypre_IJAssumedPart   *apart;

   /*-----------------------------------------------------------
    * get parcsr_A information
    *----------------------------------------------------------*/
   col_map_off_d =  hypre_ParCSRBlockMatrixColMapOffd(A);
   num_cols_off_d = hypre_CSRBlockMatrixNumCols(hypre_ParCSRBlockMatrixOffd(A));

   global_num_cols = hypre_ParCSRBlockMatrixGlobalNumCols(A);

   comm = hypre_ParCSRBlockMatrixComm(A);

   first_col_diag = hypre_ParCSRBlockMatrixFirstColDiag(A);

   /* Create the assumed partition */
   if (hypre_ParCSRBlockMatrixAssumedPartition(A) == NULL)
   {
      hypre_ParCSRBlockMatrixCreateAssumedPartition(A);
   }

   apart = hypre_ParCSRBlockMatrixAssumedPartition(A);

   /*-----------------------------------------------------------
    * get commpkg info information
    *----------------------------------------------------------*/

   hypre_ParCSRCommPkgCreateApart_core( comm, col_map_off_d, first_col_diag,
                                        num_cols_off_d, global_num_cols,
                                        &num_recvs, &recv_procs, &recv_vec_starts,
                                        &num_sends, &send_procs, &send_map_starts,
                                        &send_map_elmts, apart);

   if (!num_recvs)
   {
      hypre_TFree(recv_procs, HYPRE_MEMORY_HOST);
      recv_procs = NULL;
   }
   if (!num_sends)
   {
      hypre_TFree(send_procs, HYPRE_MEMORY_HOST);
      hypre_TFree(send_map_elmts, HYPRE_MEMORY_HOST);
      send_procs = NULL;
      send_map_elmts = NULL;
   }

   /*-----------------------------------------------------------
    * setup commpkg
    *----------------------------------------------------------*/

   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   hypre_ParCSRBlockMatrixCommPkg(A) = comm_pkg;

   return hypre_error_flag;
}

/* ----------------------------------------------------------------------
 * hypre_ParCSRBlockMatrixExtractBExt: extracts rows from B which are located on
 * other processors and needed for multiplication with A locally. The rows
 * are returned as CSRBlockMatrix.
 * ---------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_ParCSRBlockMatrixExtractBExt(hypre_ParCSRBlockMatrix *B,
                                   hypre_ParCSRBlockMatrix *A, HYPRE_Int data)
{
   MPI_Comm comm = hypre_ParCSRBlockMatrixComm(B);
   HYPRE_BigInt first_col_diag = hypre_ParCSRBlockMatrixFirstColDiag(B);
   HYPRE_BigInt *col_map_offd = hypre_ParCSRBlockMatrixColMapOffd(B);
   HYPRE_Int block_size = hypre_ParCSRBlockMatrixBlockSize(B);

   hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   HYPRE_Int *recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   HYPRE_Int *send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   HYPRE_Int *send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   hypre_ParCSRCommHandle *comm_handle;
   hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;

   hypre_CSRBlockMatrix *diag = hypre_ParCSRBlockMatrixDiag(B);

   HYPRE_Int *diag_i = hypre_CSRBlockMatrixI(diag);
   HYPRE_Int *diag_j = hypre_CSRBlockMatrixJ(diag);
   HYPRE_Complex *diag_data = hypre_CSRBlockMatrixData(diag);

   hypre_CSRBlockMatrix *offd = hypre_ParCSRBlockMatrixOffd(B);

   HYPRE_Int *offd_i = hypre_CSRBlockMatrixI(offd);
   HYPRE_Int *offd_j = hypre_CSRBlockMatrixJ(offd);
   HYPRE_Complex *offd_data = hypre_CSRBlockMatrixData(offd);

   HYPRE_Int *B_int_i;
   HYPRE_BigInt *B_int_j;
   HYPRE_Complex *B_int_data = NULL;

   HYPRE_Int num_cols_B, num_nonzeros;
   HYPRE_Int num_rows_B_ext;
   HYPRE_Int num_procs, my_id;

   hypre_CSRBlockMatrix *B_ext;

   HYPRE_Int *B_ext_i;
   HYPRE_BigInt *B_ext_j;
   HYPRE_Complex *B_ext_data = NULL;

   HYPRE_Int *jdata_recv_vec_starts;
   HYPRE_Int *jdata_send_map_starts;

   HYPRE_Int i, j, k, l, counter, bnnz;
   HYPRE_Int start_index;
   HYPRE_Int j_cnt, jrow;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   bnnz = block_size * block_size;
   num_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];
   B_int_i = hypre_CTAlloc(HYPRE_Int,  send_map_starts[num_sends] + 1, HYPRE_MEMORY_HOST);
   B_ext_i = hypre_CTAlloc(HYPRE_Int,  num_rows_B_ext + 1, HYPRE_MEMORY_HOST);
   /*--------------------------------------------------------------------------
    * generate B_int_i through adding number of row-elements of offd and diag
    * for corresponding rows. B_int_i[j+1] contains the number of elements of
    * a row j (which is determined through send_map_elmts)
    *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   j_cnt = 0;
   num_nonzeros = 0;
   for (i = 0; i < num_sends; i++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
      {
         jrow = send_map_elmts[j];
         B_int_i[++j_cnt] = offd_i[jrow + 1] - offd_i[jrow]
                            + diag_i[jrow + 1] - diag_i[jrow];
         num_nonzeros += B_int_i[j_cnt];
      }
   }

   /*--------------------------------------------------------------------------
    * initialize communication
    *--------------------------------------------------------------------------*/
   comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                              &B_int_i[1], &B_ext_i[1]);

   B_int_j = hypre_CTAlloc(HYPRE_BigInt,  num_nonzeros, HYPRE_MEMORY_HOST);
   if (data) { B_int_data = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros * bnnz, HYPRE_MEMORY_HOST); }

   jdata_send_map_starts = hypre_CTAlloc(HYPRE_Int,  num_sends + 1, HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_recvs + 1, HYPRE_MEMORY_HOST);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
   counter = 0;
   for (i = 0; i < num_sends; i++)
   {
      num_nonzeros = counter;
      for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
      {
         jrow = send_map_elmts[j];
         for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
         {
            B_int_j[counter] = (HYPRE_BigInt)diag_j[k] + first_col_diag;
            if (data)
            {
               for (l = 0; l < bnnz; l++)
               {
                  B_int_data[counter * bnnz + l] = diag_data[k * bnnz + l];
               }
            }
            counter++;
         }
         for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
         {
            B_int_j[counter] = col_map_offd[offd_j[k]];
            if (data)
            {
               for (l = 0; l < bnnz; l++)
                  B_int_data[counter * bnnz + l] =
                     offd_data[k * bnnz + l];
            }
            counter++;
         }
      }
      num_nonzeros = counter - num_nonzeros;
      start_index += num_nonzeros;
      jdata_send_map_starts[i + 1] = start_index;
   }

   /* Create temporary communication package */
   hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    jdata_recv_vec_starts,
                                    num_sends,
                                    hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    jdata_send_map_starts,
                                    NULL,
                                    &tmp_comm_pkg);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
    * after communication exchange B_ext_i[j+1] contains the number of elements
    * of a row j !
    * evaluate B_ext_i and compute num_nonzeros for B_ext
    *--------------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i++)
   {
      for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
      {
         B_ext_i[j + 1] += B_ext_i[j];
      }
   }

   num_nonzeros = B_ext_i[num_rows_B_ext];

   B_ext = hypre_CSRBlockMatrixCreate(block_size, num_rows_B_ext, num_cols_B,
                                      num_nonzeros);
   B_ext_j = hypre_CTAlloc(HYPRE_BigInt,  num_nonzeros, HYPRE_MEMORY_HOST);
   if (data)
   {
      B_ext_data = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros * bnnz, HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_recvs; i++)
   {
      start_index = B_ext_i[recv_vec_starts[i]];
      num_nonzeros = B_ext_i[recv_vec_starts[i + 1]] - start_index;
      jdata_recv_vec_starts[i + 1] = B_ext_i[recv_vec_starts[i + 1]];
   }

   comm_handle = hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, B_int_j, B_ext_j);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
      comm_handle = hypre_ParCSRBlockCommHandleCreate(1, bnnz, tmp_comm_pkg,
                                                      B_int_data, B_ext_data);
      hypre_ParCSRBlockCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   hypre_CSRBlockMatrixI(B_ext) = B_ext_i;
   hypre_CSRBlockMatrixBigJ(B_ext) = B_ext_j;
   if (data)
   {
      hypre_CSRBlockMatrixData(B_ext) = B_ext_data;
   }

   /* Free memory */
   hypre_TFree(jdata_send_map_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(jdata_recv_vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(tmp_comm_pkg, HYPRE_MEMORY_HOST);
   hypre_TFree(B_int_i, HYPRE_MEMORY_HOST);
   hypre_TFree(B_int_j, HYPRE_MEMORY_HOST);
   if (data)
   {
      hypre_TFree(B_int_data, HYPRE_MEMORY_HOST);
   }

   return B_ext;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorCreateFromBlock
 *--------------------------------------------------------------------------*/

hypre_ParVector *
hypre_ParVectorCreateFromBlock(  MPI_Comm comm,
                                 HYPRE_BigInt p_global_size,
                                 HYPRE_BigInt *p_partitioning, HYPRE_Int block_size)
{
   hypre_ParVector  *vector;
   HYPRE_Int num_procs, my_id;
   HYPRE_BigInt global_size;
   HYPRE_BigInt new_partitioning[2]; /* need to create a new partitioning - son't want to write over
                                     what is passed in */

   global_size = p_global_size * (HYPRE_BigInt)block_size;

   vector = hypre_CTAlloc(hypre_ParVector, 1, HYPRE_MEMORY_HOST);
   hypre_MPI_Comm_rank(comm, &my_id);
   hypre_MPI_Comm_size(comm, &num_procs);

   if (!p_partitioning)
   {
      hypre_GenerateLocalPartitioning(global_size, num_procs, my_id, new_partitioning);
   }
   else /* adjust for block_size */
   {
      new_partitioning[0] = p_partitioning[0] * (HYPRE_BigInt)block_size;
      new_partitioning[1] = p_partitioning[1] * (HYPRE_BigInt)block_size;
   }

   hypre_ParVectorComm(vector) = comm;
   hypre_ParVectorGlobalSize(vector) = global_size;
   hypre_ParVectorFirstIndex(vector) = new_partitioning[0];
   hypre_ParVectorLastIndex(vector)  = new_partitioning[1] - 1;
   hypre_ParVectorPartitioning(vector)[0] = new_partitioning[0];
   hypre_ParVectorPartitioning(vector)[1] = new_partitioning[1];
   hypre_ParVectorLocalVector(vector) =
      hypre_SeqVectorCreate(new_partitioning[1] - new_partitioning[0]);

   /* set defaults */
   hypre_ParVectorOwnsData(vector) = 1;

   return vector;
}
