/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

#include "../HYPRE.h"

/******************************************************************************
 *
 * hypre_IJMatrixCreateParCSR
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixCreateParCSR(hypre_IJMatrix *matrix)
{
   MPI_Comm             comm = hypre_IJMatrixComm(matrix);
   HYPRE_BigInt        *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt        *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_ParCSRMatrix  *par_matrix;

   HYPRE_BigInt         row_starts[2];
   HYPRE_BigInt         col_starts[2];
   HYPRE_Int            i;

   if (hypre_IJMatrixGlobalFirstRow(matrix))
   {
      for (i = 0; i < 2; i++)
      {
         row_starts[i] = row_partitioning[i] - hypre_IJMatrixGlobalFirstRow(matrix);
      }
   }
   else
   {
      for (i = 0; i < 2; i++)
      {
         row_starts[i] = row_partitioning[i];
      }
   }

   if (hypre_IJMatrixGlobalFirstCol(matrix))
   {
      for (i = 0; i < 2; i++)
      {
         col_starts[i] = col_partitioning[i] - hypre_IJMatrixGlobalFirstCol(matrix);
      }
   }
   else
   {
      for (i = 0; i < 2; i++)
      {
         col_starts[i] = col_partitioning[i];
      }
   }

   par_matrix = hypre_ParCSRMatrixCreate(comm, hypre_IJMatrixGlobalNumRows(matrix),
                                         hypre_IJMatrixGlobalNumCols(matrix),
                                         row_starts, col_starts, 0, 0, 0);

   hypre_IJMatrixObject(matrix) = par_matrix;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetRowSizesParCSR
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixSetRowSizesParCSR(hypre_IJMatrix  *matrix,
                                const HYPRE_Int *sizes)
{
   HYPRE_Int local_num_rows, local_num_cols, i, *row_space = NULL;
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   local_num_rows = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
   local_num_cols = (HYPRE_Int)(col_partitioning[1] - col_partitioning[0]);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);

   if (aux_matrix)
   {
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   }
   if (!row_space)
   {
      row_space = hypre_CTAlloc(HYPRE_Int, local_num_rows, HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < local_num_rows; i++)
   {
      row_space[i] = sizes[i];
   }
   if (!aux_matrix)
   {
      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, local_num_cols, row_space);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixRowSpace(aux_matrix) = row_space;

#if defined(HYPRE_USING_GPU)
   hypre_AuxParCSRMatrixUsrOnProcElmts(aux_matrix) = 0;
   for (i = 0; i < local_num_rows; i++)
   {
      hypre_AuxParCSRMatrixUsrOnProcElmts(aux_matrix) += sizes[i];
   }
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetDiagOffdSizesParCSR
 * sets diag_i inside the diag part of the ParCSRMatrix
 * and offd_i inside the offd part,
 * requires exact row sizes for diag and offd
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixSetDiagOffdSizesParCSR(hypre_IJMatrix  *matrix,
                                     const HYPRE_Int *diag_sizes,
                                     const HYPRE_Int *offd_sizes)
{
   HYPRE_Int local_num_rows, local_num_cols;
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   local_num_rows = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
   local_num_cols = (HYPRE_Int)(col_partitioning[1] - col_partitioning[0]);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix *)hypre_IJMatrixTranslator(matrix);

   if (!aux_matrix)
   {
      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, local_num_cols, NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }

   if ( hypre_AuxParCSRMatrixDiagSizes(aux_matrix) == NULL)
   {
      hypre_AuxParCSRMatrixDiagSizes(aux_matrix) = hypre_TAlloc(HYPRE_Int, local_num_rows,
                                                                HYPRE_MEMORY_HOST);
   }

   if ( hypre_AuxParCSRMatrixOffdSizes(aux_matrix) == NULL)
   {
      hypre_AuxParCSRMatrixOffdSizes(aux_matrix) = hypre_TAlloc(HYPRE_Int, local_num_rows,
                                                                HYPRE_MEMORY_HOST);
   }

   hypre_TMemcpy(hypre_AuxParCSRMatrixDiagSizes(aux_matrix), diag_sizes, HYPRE_Int, local_num_rows,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   hypre_TMemcpy(hypre_AuxParCSRMatrixOffdSizes(aux_matrix), offd_sizes, HYPRE_Int, local_num_rows,
                 HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetMaxOnProcElmtsParCSR
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixSetMaxOnProcElmtsParCSR(hypre_IJMatrix *matrix,
                                      HYPRE_Int       max_on_proc_elmts)
{
#if defined(HYPRE_USING_GPU)
   MPI_Comm                 comm             = hypre_IJMatrixComm(matrix);
   HYPRE_BigInt            *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt            *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   hypre_AuxParCSRMatrix   *aux_matrix;
   HYPRE_Int                local_num_rows, local_num_cols, my_id;

   hypre_MPI_Comm_rank(comm, &my_id);
   aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);
   if (!aux_matrix)
   {
      local_num_rows = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
      local_num_cols = (HYPRE_Int)(col_partitioning[1] - col_partitioning[0]);

      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, local_num_cols, NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixUsrOnProcElmts(aux_matrix) = max_on_proc_elmts;

#else
   HYPRE_UNUSED_VAR(matrix);
   HYPRE_UNUSED_VAR(max_on_proc_elmts);
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetMaxOffProcElmtsParCSR
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixSetMaxOffProcElmtsParCSR(hypre_IJMatrix *matrix,
                                       HYPRE_Int       max_off_proc_elmts)
{
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_Int local_num_rows, local_num_cols, my_id;
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   hypre_MPI_Comm_rank(comm, &my_id);
   aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);
   if (!aux_matrix)
   {
      local_num_rows = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
      local_num_cols = (HYPRE_Int)(col_partitioning[1] - col_partitioning[0]);
      hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows,
                                  local_num_cols, NULL);
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;

#if defined(HYPRE_USING_GPU)
   hypre_AuxParCSRMatrixUsrOffProcElmts(aux_matrix) = max_off_proc_elmts;
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixInitializeParCSR
 *
 * initializes AuxParCSRMatrix and ParCSRMatrix as necessary
 *
 *****************************************************************************/
HYPRE_Int
hypre_IJMatrixInitializeParCSR(hypre_IJMatrix *matrix)
{
   return hypre_IJMatrixInitializeParCSR_v2(matrix, hypre_HandleMemoryLocation(hypre_handle()));
}

HYPRE_Int
hypre_IJMatrixInitializeParCSR_v2(hypre_IJMatrix *matrix, HYPRE_MemoryLocation memory_location)
{
   hypre_ParCSRMatrix    *par_matrix = (hypre_ParCSRMatrix *)    hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);

   HYPRE_MemoryLocation memory_location_aux =
      hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_HOST ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;

   if (hypre_IJMatrixAssembleFlag(matrix) == 0)
   {
      if (!par_matrix)
      {
         hypre_IJMatrixCreateParCSR(matrix);
         par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(matrix);
      }

      HYPRE_Int local_num_rows = hypre_ParCSRMatrixNumRows(par_matrix);
      HYPRE_Int i;
      hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
      hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);

      if (!aux_matrix)
      {
         hypre_AuxParCSRMatrixCreate(&aux_matrix, local_num_rows, hypre_ParCSRMatrixNumCols(par_matrix),
                                     NULL);
         hypre_IJMatrixTranslator(matrix) = aux_matrix;
      }

      hypre_ParCSRMatrixInitialize_v2(par_matrix, memory_location);
      hypre_AuxParCSRMatrixInitialize_v2(aux_matrix, memory_location_aux);

#if defined(HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(memory_location_aux) == HYPRE_EXEC_HOST)
#endif
      {
         if (hypre_AuxParCSRMatrixDiagSizes(aux_matrix))
         {
            for (i = 0; i < local_num_rows; i++)
            {
               hypre_CSRMatrixI(diag)[i + 1] = hypre_CSRMatrixI(diag)[i] + hypre_AuxParCSRMatrixDiagSizes(
                                                  aux_matrix)[i];
            }
            hypre_CSRMatrixNumNonzeros(diag) = hypre_CSRMatrixI(diag)[local_num_rows];
            hypre_CSRMatrixInitialize(diag);
         }

         if (hypre_AuxParCSRMatrixOffdSizes(aux_matrix))
         {
            for (i = 0; i < local_num_rows; i++)
            {
               hypre_CSRMatrixI(offd)[i + 1] = hypre_CSRMatrixI(offd)[i] + hypre_AuxParCSRMatrixOffdSizes(
                                                  aux_matrix)[i];
            }
            hypre_CSRMatrixNumNonzeros(offd) = hypre_CSRMatrixI(offd)[local_num_rows];
            hypre_CSRMatrixInitialize(offd);
         }
      }

      if (!hypre_AuxParCSRMatrixNeedAux(aux_matrix))
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < local_num_rows; i++)
         {
            hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[i] = hypre_CSRMatrixI(diag)[i];
            hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[i] = hypre_CSRMatrixI(offd)[i];
         }
      }
   }
   else if ( memory_location_aux == HYPRE_MEMORY_HOST )
   {
      /* AB 4/06 - the assemble routine destroys the aux matrix - so we need
         to recreate if initialize is called again
      */
      if (!aux_matrix)
      {
         hypre_AuxParCSRMatrixCreate(&aux_matrix, hypre_ParCSRMatrixNumRows(par_matrix),
                                     hypre_ParCSRMatrixNumCols(par_matrix), NULL);
         hypre_AuxParCSRMatrixMemoryLocation(aux_matrix) = HYPRE_MEMORY_HOST;
         hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
         hypre_IJMatrixTranslator(matrix) = aux_matrix;
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixGetRowCountsParCSR
 *
 * gets the number of columns for rows specified by the user
 *
 *****************************************************************************/

HYPRE_Int hypre_IJMatrixGetRowCountsParCSR( hypre_IJMatrix *matrix,
                                            HYPRE_Int       nrows,
                                            HYPRE_BigInt   *rows,
                                            HYPRE_Int      *ncols)
{
   HYPRE_BigInt row_index;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(matrix);

   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);

   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);
   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);

   HYPRE_Int i, my_id, index;
   HYPRE_Int print_level = hypre_IJMatrixPrintLevel(matrix);

   hypre_MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, row_index) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nrows; i++)
   {
      row_index = rows[i];
      if (row_index >= row_partitioning[0] &&
          row_index < row_partitioning[1])
      {
         /* compute local row number */
         index = (HYPRE_Int)(row_index - row_partitioning[0]);
         ncols[i] = diag_i[index + 1] - diag_i[index] + offd_i[index + 1] - offd_i[index];
      }
      else
      {
         ncols[i] = 0;
         if (print_level)
         {
            hypre_printf ("Warning! Row %b is not on Proc. %d!\n",
                          row_index, my_id);
         }
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixGetValuesParCSR
 *
 * gets values of an IJMatrix
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixGetValuesParCSR( hypre_IJMatrix *matrix,
                               HYPRE_Int       nrows,
                               HYPRE_Int      *ncols,
                               HYPRE_BigInt   *rows,
                               HYPRE_BigInt   *cols,
                               HYPRE_Complex  *values)
{
   MPI_Comm             comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix  *par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject(matrix);
   HYPRE_Int            assemble_flag = hypre_IJMatrixAssembleFlag(matrix);

   hypre_CSRMatrix     *diag;
   HYPRE_Int           *diag_i    = NULL;
   HYPRE_Int           *diag_j    = NULL;
   HYPRE_Complex       *diag_data = NULL;

   hypre_CSRMatrix     *offd;
   HYPRE_Int           *offd_i    = NULL;
   HYPRE_Int           *offd_j    = NULL;
   HYPRE_Complex       *offd_data = NULL;

   HYPRE_BigInt        *col_map_offd = NULL;
   HYPRE_BigInt        *col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   HYPRE_BigInt        *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);

   HYPRE_Int            i, j, n, ii, indx;
   HYPRE_Int            num_procs, my_id;
   HYPRE_BigInt         col_0, col_n, row, col_indx, first;
   HYPRE_Int            row_local, row_size;
   HYPRE_Int            warning = 0;
   HYPRE_Int           *counter;
   HYPRE_Int            print_level = hypre_IJMatrixPrintLevel(matrix);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (assemble_flag == 0)
   {
      hypre_error_in_arg(1);
      if (print_level)
      {
         hypre_printf("Error! Matrix not assembled yet! HYPRE_IJMatrixGetValues\n");
      }
   }

   col_0 = col_starts[0];
   col_n = col_starts[1] - 1;
   first = hypre_IJMatrixGlobalFirstCol(matrix);

   diag = hypre_ParCSRMatrixDiag(par_matrix);
   diag_i = hypre_CSRMatrixI(diag);
   diag_j = hypre_CSRMatrixJ(diag);
   diag_data = hypre_CSRMatrixData(diag);

   offd = hypre_ParCSRMatrixOffd(par_matrix);
   offd_i = hypre_CSRMatrixI(offd);
   if (num_procs > 1)
   {
      offd_j = hypre_CSRMatrixJ(offd);
      offd_data = hypre_CSRMatrixData(offd);
      col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
   }

   if (nrows < 0)
   {
      nrows = -nrows;

      counter = hypre_CTAlloc(HYPRE_Int, nrows + 1, HYPRE_MEMORY_HOST);
      counter[0] = 0;
      for (i = 0; i < nrows; i++)
      {
         counter[i + 1] = counter[i] + ncols[i];
      }

      indx = 0;
      for (i = 0; i < nrows; i++)
      {
         row = rows[i];
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (HYPRE_Int)(row - row_partitioning[0]);
            row_size = diag_i[row_local + 1] - diag_i[row_local] +
                       offd_i[row_local + 1] - offd_i[row_local];
            if (counter[i] + row_size > counter[nrows])
            {
               hypre_error_in_arg(1);
               if (print_level)
               {
                  hypre_printf ("Error! Not enough memory! HYPRE_IJMatrixGetValues\n");
               }
            }
            if (ncols[i] < row_size)
            {
               warning = 1;
            }
            for (j = diag_i[row_local]; j < diag_i[row_local + 1]; j++)
            {
               cols[indx] = (HYPRE_BigInt)diag_j[j] + col_0;
               values[indx++] = diag_data[j];
            }
            for (j = offd_i[row_local]; j < offd_i[row_local + 1]; j++)
            {
               cols[indx] = col_map_offd[offd_j[j]];
               values[indx++] = offd_data[j];
            }
            counter[i + 1] = indx;
         }
         else
         {
            if (print_level)
            {
               hypre_printf ("Warning! Row %b is not on Proc. %d!\n", row, my_id);
            }
         }
      }
      if (warning)
      {
         for (i = 0; i < nrows; i++)
         {
            ncols[i] = counter[i + 1] - counter[i];
         }
         if (print_level)
         {
            hypre_printf ("Warning!  ncols has been changed!\n");
         }
      }
      hypre_TFree(counter, HYPRE_MEMORY_HOST);
   }
   else
   {
      indx = 0;
      for (ii = 0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols[ii];
         if (n == 0) /* empty row */
         {
            continue;
         }
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (HYPRE_Int)(row - row_partitioning[0]);
            /* compute local row number */
            for (i = 0; i < n; i++)
            {
               col_indx = cols[indx] - first;
               values[indx] = 0.0;
               if (col_indx < col_0 || col_indx > col_n)
                  /* search in offd */
               {
                  for (j = offd_i[row_local]; j < offd_i[row_local + 1]; j++)
                  {
                     if (col_map_offd[offd_j[j]] == col_indx)
                     {
                        values[indx] = offd_data[j];
                        break;
                     }
                  }
               }
               else  /* search in diag */
               {
                  col_indx = col_indx - col_0;
                  for (j = diag_i[row_local]; j < diag_i[row_local + 1]; j++)
                  {
                     if (diag_j[j] == (HYPRE_Int)col_indx)
                     {
                        values[indx] = diag_data[j];
                        break;
                     }
                  }
               }
               indx++;
            }
         }
         else
         {
            if (print_level)
            {
               hypre_printf ("Warning! Row %b is not on Proc. %d!\n", row, my_id);
            }
         }
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetValuesParCSR
 *
 * sets values in an IJMatrix before assembly,
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixSetValuesParCSR( hypre_IJMatrix       *matrix,
                               HYPRE_Int             nrows,
                               HYPRE_Int            *ncols,
                               const HYPRE_BigInt   *rows,
                               const HYPRE_Int      *row_indexes,
                               const HYPRE_BigInt   *cols,
                               const HYPRE_Complex  *values )
{
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_Int num_procs, my_id;
   HYPRE_Int row_local;
   //HYPRE_Int row_len;
   HYPRE_BigInt col_0, col_n, row;
   HYPRE_Int i, ii, j, n, not_found;
   //HYPRE_Int col_indx, cnt1;
   HYPRE_BigInt **aux_j;
   HYPRE_BigInt *local_j;
   HYPRE_BigInt *tmp_j;
   HYPRE_Complex **aux_data;
   HYPRE_Complex  *local_data;
   HYPRE_Complex  *tmp_data = NULL;
   HYPRE_Int diag_space, offd_space;
   HYPRE_Int *row_length, *row_space;
   HYPRE_Int need_aux;
   HYPRE_Int tmp_indx, indx;
   HYPRE_Int space, size, old_size;
   HYPRE_Int cnt, cnt_diag, cnt_offd;
   HYPRE_Int pos_diag, pos_offd;
   HYPRE_Int len_diag, len_offd;
   HYPRE_Int offd_indx, diag_indx;
   HYPRE_Int *diag_i = NULL;
   HYPRE_Int *diag_j = NULL;
   HYPRE_Complex *diag_data = NULL;
   HYPRE_Int *offd_i = NULL;
   HYPRE_Int *offd_j = NULL;
   HYPRE_Complex *offd_data = NULL;
   HYPRE_BigInt first;
   /*HYPRE_Int current_num_elmts;*/
   /*HYPRE_Int max_off_proc_elmts;*/
   //HYPRE_Int off_proc_i_indx;
   //HYPRE_BigInt *off_proc_i;
   //HYPRE_BigInt *off_proc_j;
   HYPRE_Int print_level = hypre_IJMatrixPrintLevel(matrix);
   /*HYPRE_Complex *off_proc_data;*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);

   col_0 = col_partitioning[0];
   col_n = col_partitioning[1] - 1;
   first =  hypre_IJMatrixGlobalFirstCol(matrix);
   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      if (print_level)
      {
         hypre_printf("Error! nrows negative! HYPRE_IJMatrixSetValues\n");
      }
   }

   if (hypre_IJMatrixAssembleFlag(matrix))  /* matrix already assembled*/
   {
      HYPRE_BigInt *col_map_offd = NULL;
      HYPRE_Int num_cols_offd;
      HYPRE_Int j_offd;
      for (ii = 0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols ? ncols[ii] : 1;
         if (n == 0) /* empty row */
         {
            continue;
         }
         indx = row_indexes[ii];

         /* processor owns the row */
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (HYPRE_Int)(row - row_partitioning[0]);

            /* compute local row number */
            diag = hypre_ParCSRMatrixDiag(par_matrix);
            diag_i = hypre_CSRMatrixI(diag);
            diag_j = hypre_CSRMatrixJ(diag);
            diag_data = hypre_CSRMatrixData(diag);
            offd = hypre_ParCSRMatrixOffd(par_matrix);
            offd_i = hypre_CSRMatrixI(offd);
            num_cols_offd = hypre_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = hypre_CSRMatrixJ(offd);
               offd_data = hypre_CSRMatrixData(offd);
            }
            size = diag_i[row_local + 1] - diag_i[row_local] +
                   offd_i[row_local + 1] - offd_i[row_local];

            if (n > size)  /* Should we change this and allow this?
                              This could be same column index, i.e. only last
                              value is set, previous ones overwritten. */
            {
               hypre_error(HYPRE_ERROR_GENERIC);
               if (print_level)
               {
                  hypre_printf (" row %b too long! \n", row);
               }
               return hypre_error_flag;
            }

            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local + 1];
            len_offd = offd_i[row_local + 1];
            not_found = 1;

            for (i = 0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
                  /* insert into offd */
               {
                  j_offd = hypre_BigBinarySearch(col_map_offd, cols[indx] - first,
                                                 num_cols_offd);
                  if (j_offd == -1)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     return hypre_error_flag;
                  }
                  for (j = pos_offd; j < len_offd; j++)
                  {
                     if (offd_j[j] == j_offd)
                     {
                        offd_data[j] = values[indx];
                        not_found = 0;
                        break;
                     }
                  }
                  if (not_found)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     return hypre_error_flag;
                  }
                  not_found = 1;
               }
               /* diagonal element */
               else if (cols[indx] == row)
               {
                  if (diag_j[pos_diag] != row_local)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     /* return -1;*/
                     return hypre_error_flag;
                  }
                  diag_data[pos_diag] = values[indx];
               }
               else  /* insert into diag */
               {
                  for (j = pos_diag; j < len_diag; j++)
                  {
                     if (diag_j[j] == (HYPRE_Int)(cols[indx] - col_0))
                     {
                        diag_data[j] = values[indx];
                        not_found = 0;
                        break;
                     }
                  }
                  if (not_found)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     /* return -1; */
                     return hypre_error_flag;
                  }
               }
               indx++;
            }
         }
      }
   }
   else
   {
      aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      for (ii = 0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols ? ncols[ii] : 1;
         if (n == 0) /* empty row */
         {
            continue;
         }
         indx = row_indexes[ii];
         /* processor owns the row */
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (HYPRE_Int)(row - row_partitioning[0]);
            /* compute local row number */
            if (need_aux)
            {
               aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
               space = row_space[row_local];
               old_size = row_length[row_local];
               size = space - old_size;
               if (size < n)
               {
                  size = n - size;
                  tmp_j = hypre_CTAlloc(HYPRE_BigInt, size, HYPRE_MEMORY_HOST);
                  tmp_data = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
               }
               else
               {
                  tmp_j = NULL;
               }
               tmp_indx = 0;
               not_found = 1;
               size = old_size;
               for (i = 0; i < n; i++)
               {
                  for (j = 0; j < old_size; j++)
                  {
                     if (local_j[j] == cols[indx])
                     {
                        local_data[j] = values[indx];
                        not_found = 0;
                        break;
                     }
                  }
                  if (not_found)
                  {
                     if (size < space)
                     {
                        local_j[size] = cols[indx];
                        local_data[size++] = values[indx];
                     }
                     else
                     {
                        tmp_j[tmp_indx] = cols[indx];
                        tmp_data[tmp_indx++] = values[indx];
                     }
                  }
                  not_found = 1;
                  indx++;
               }

               row_length[row_local] = size + tmp_indx;

               if (tmp_indx)
               {
                  aux_j[row_local] = hypre_TReAlloc(aux_j[row_local], HYPRE_BigInt,
                                                    size + tmp_indx, HYPRE_MEMORY_HOST);
                  aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
                                                       HYPRE_Complex, size + tmp_indx,
                                                       HYPRE_MEMORY_HOST);
                  row_space[row_local] = size + tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }

               cnt = size;

               for (i = 0; i < tmp_indx; i++)
               {
                  local_j[cnt] = tmp_j[i];
                  local_data[cnt++] = tmp_data[i];
               }

               if (tmp_j)
               {
                  hypre_TFree(tmp_j, HYPRE_MEMORY_HOST);
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
               HYPRE_BigInt *big_offd_j = NULL;
               HYPRE_Int     col_j;

               offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
               diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = hypre_ParCSRMatrixDiag(par_matrix);
               diag_i = hypre_CSRMatrixI(diag);
               diag_j = hypre_CSRMatrixJ(diag);
               diag_data = hypre_CSRMatrixData(diag);
               offd = hypre_ParCSRMatrixOffd(par_matrix);
               offd_i = hypre_CSRMatrixI(offd);
               if (num_procs > 1)
               {
                  big_offd_j = hypre_CSRMatrixBigJ(offd);
                  offd_data = hypre_CSRMatrixData(offd);
                  if (!big_offd_j)
                  {
                     big_offd_j = hypre_CTAlloc(HYPRE_BigInt,
                                                offd_i[hypre_CSRMatrixNumRows(offd)],
                                                hypre_CSRMatrixMemoryLocation(offd));
                     hypre_CSRMatrixBigJ(offd) = big_offd_j;
                  }
               }
               cnt_diag = diag_indx;
               cnt_offd = offd_indx;
               diag_space = diag_i[row_local + 1];
               offd_space = offd_i[row_local + 1];
               not_found = 1;
               for (i = 0; i < n; i++)
               {
                  if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */
                  {
                     for (j = offd_i[row_local]; j < offd_indx; j++)
                     {
                        if (big_offd_j[j] == cols[indx])
                        {
                           offd_data[j] = values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        if (cnt_offd < offd_space)
                        {
                           big_offd_j[cnt_offd] = cols[indx];
                           offd_data[cnt_offd++] = values[indx];
                        }
                        else
                        {
                           hypre_error(HYPRE_ERROR_GENERIC);
                           if (print_level)
                           {
                              hypre_printf("Error in row %b ! Too many elements!\n",
                                           row);
                           }
                           /* return 1; */
                           return hypre_error_flag;
                        }
                     }
                     not_found = 1;
                  }
                  else  /* insert into diag */
                  {
                     col_j = (HYPRE_Int)(cols[indx] - col_0);
                     for (j = diag_i[row_local]; j < diag_indx; j++)
                     {
                        if (diag_j[j] == col_j)
                        {
                           diag_data[j] = values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        if (cnt_diag < diag_space)
                        {
                           diag_j[cnt_diag] = col_j;
                           diag_data[cnt_diag++] = values[indx];
                        }
                        else
                        {
                           hypre_error(HYPRE_ERROR_GENERIC);
                           if (print_level)
                           {
                              hypre_printf("Error in row %b ! Too many elements !\n",
                                           row);
                           }
                           /* return 1; */
                           return hypre_error_flag;
                        }
                     }
                     not_found = 1;
                  }
                  indx++;
               }

               hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;
            }
         }
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetConstantValuesParCSR
 *
 * sets all values in an already assembled IJMatrix to a constant value.
 *
 *****************************************************************************/

void
hypre_IJMatrixSetConstantValuesParCSRHost( hypre_IJMatrix *matrix,
                                           HYPRE_Complex   value )
{
   hypre_ParCSRMatrix *par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject( matrix );
   hypre_CSRMatrix    *diag       = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix    *offd       = hypre_ParCSRMatrixOffd(par_matrix);
   HYPRE_Complex      *diag_data  = hypre_CSRMatrixData(diag);
   HYPRE_Complex      *offd_data  = hypre_CSRMatrixData(offd);
   HYPRE_Int           nnz_diag   = hypre_CSRMatrixNumNonzeros(diag);
   HYPRE_Int           nnz_offd   = hypre_CSRMatrixNumNonzeros(offd);
   HYPRE_Int           ii;

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(ii) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < nnz_diag; ii++)
   {
      diag_data[ii] = value;
   }
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(ii) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < nnz_offd; ii++)
   {
      offd_data[ii] = value;
   }
}

HYPRE_Int
hypre_IJMatrixSetConstantValuesParCSR( hypre_IJMatrix *matrix,
                                       HYPRE_Complex   value )
{
   if (hypre_IJMatrixAssembleFlag(matrix))  /* matrix already assembled*/
   {
#if defined(HYPRE_USING_GPU)
      if (hypre_GetExecPolicy1(hypre_IJMatrixMemoryLocation(matrix)) == HYPRE_EXEC_DEVICE)
      {
         hypre_IJMatrixSetConstantValuesParCSRDevice(matrix, value);
      }
      else
#endif
      {
         hypre_IJMatrixSetConstantValuesParCSRHost(matrix, value);
      }
   }
   else
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Matrix not assembled! Required to set constant values!");
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixAddToValuesParCSR
 *
 * adds row values to an IJMatrix
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixAddToValuesParCSR( hypre_IJMatrix       *matrix,
                                 HYPRE_Int             nrows,
                                 HYPRE_Int            *ncols,
                                 const HYPRE_BigInt   *rows,
                                 const HYPRE_Int      *row_indexes,
                                 const HYPRE_BigInt   *cols,
                                 const HYPRE_Complex  *values )
{
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_Int num_procs, my_id;
   HYPRE_Int row_local;
   HYPRE_BigInt row;
   HYPRE_BigInt col_0, col_n;
   HYPRE_Int i, ii, j, n, not_found;
   HYPRE_BigInt **aux_j;
   HYPRE_BigInt *local_j;
   HYPRE_BigInt *tmp_j;
   HYPRE_Complex **aux_data;
   HYPRE_Complex  *local_data;
   HYPRE_Complex  *tmp_data = NULL;
   HYPRE_Int diag_space, offd_space;
   HYPRE_Int *row_length, *row_space;
   HYPRE_Int need_aux;
   HYPRE_Int tmp_indx, indx;
   HYPRE_Int space, size, old_size;
   HYPRE_Int cnt, cnt_diag, cnt_offd;
   HYPRE_Int pos_diag, pos_offd;
   HYPRE_Int len_diag, len_offd;
   HYPRE_Int offd_indx, diag_indx;
   HYPRE_BigInt first;
   HYPRE_Int *diag_i = NULL;
   HYPRE_Int *diag_j = NULL;
   HYPRE_Complex *diag_data = NULL;
   HYPRE_Int *offd_i = NULL;
   HYPRE_Int *offd_j = NULL;
   HYPRE_Complex *offd_data = NULL;
   HYPRE_Int current_num_elmts;
   HYPRE_Int max_off_proc_elmts;
   HYPRE_Int off_proc_i_indx;
   HYPRE_BigInt *off_proc_i;
   HYPRE_BigInt *off_proc_j;
   HYPRE_Complex *off_proc_data;
   HYPRE_Int print_level = hypre_IJMatrixPrintLevel(matrix);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);
   col_0 = col_partitioning[0];
   col_n = col_partitioning[1] - 1;
   first = hypre_IJMatrixGlobalFirstCol(matrix);
   if (hypre_IJMatrixAssembleFlag(matrix))
   {
      HYPRE_Int num_cols_offd;
      HYPRE_BigInt *col_map_offd = NULL;
      HYPRE_Int j_offd;

      /* AB - 4/06 - need to get this object*/
      aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);

      for (ii = 0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols ? ncols[ii] : 1;
         if (n == 0) /* empty row */
         {
            continue;
         }
         indx = row_indexes[ii];
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (HYPRE_Int)(row - row_partitioning[0]);
            /* compute local row number */
            diag = hypre_ParCSRMatrixDiag(par_matrix);
            diag_i = hypre_CSRMatrixI(diag);
            diag_j = hypre_CSRMatrixJ(diag);
            diag_data = hypre_CSRMatrixData(diag);
            offd = hypre_ParCSRMatrixOffd(par_matrix);
            offd_i = hypre_CSRMatrixI(offd);
            num_cols_offd = hypre_CSRMatrixNumCols(offd);
            if (num_cols_offd)
            {
               col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
               offd_j = hypre_CSRMatrixJ(offd);
               offd_data = hypre_CSRMatrixData(offd);
            }
            size = diag_i[row_local + 1] - diag_i[row_local] +
                   offd_i[row_local + 1] - offd_i[row_local];

            if (n > size)  /* Should we change this and allow this?
                              This could be same column index, i.e. only last
                              value is set, previous ones overwritten. */
            {
               hypre_error(HYPRE_ERROR_GENERIC);
               if (print_level)
               {
                  hypre_printf (" row %b too long! \n", row);
               }
               return hypre_error_flag;
            }

            pos_diag = diag_i[row_local];
            pos_offd = offd_i[row_local];
            len_diag = diag_i[row_local + 1];
            len_offd = offd_i[row_local + 1];
            not_found = 1;

            for (i = 0; i < n; i++)
            {
               if (cols[indx] < col_0 || cols[indx] > col_n)
                  /* insert into offd */
               {
                  j_offd = hypre_BigBinarySearch(col_map_offd, cols[indx] - first,
                                                 num_cols_offd);
                  if (j_offd == -1)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     return hypre_error_flag;
                     /* return -1; */
                  }
                  for (j = pos_offd; j < len_offd; j++)
                  {
                     if (offd_j[j] == j_offd)
                     {
                        offd_data[j] += values[indx];
                        not_found = 0;
                        break;
                     }
                  }
                  if (not_found)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     return hypre_error_flag;
                  }
                  not_found = 1;
               }
               /* diagonal element */
               else if (cols[indx] == row)
               {
                  if (diag_j[pos_diag] != row_local)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     return hypre_error_flag;
                  }
                  diag_data[pos_diag] += values[indx];
               }
               else  /* insert into diag */
               {
                  for (j = pos_diag; j < len_diag; j++)
                  {
                     if (diag_j[j] == (HYPRE_Int)(cols[indx] - col_0))
                     {
                        diag_data[j] += values[indx];
                        not_found = 0;
                        break;
                     }
                  }
                  if (not_found)
                  {
                     hypre_error(HYPRE_ERROR_GENERIC);
                     if (print_level)
                     {
                        hypre_printf (" Error, element %b %b does not exist\n",
                                      row, cols[indx]);
                     }
                     return hypre_error_flag;
                  }
               }
               indx++;
            }
         }
         /* not my row */
         else
         {
            if (!aux_matrix)
            {
               size = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
               hypre_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
               hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
               hypre_IJMatrixTranslator(matrix) = aux_matrix;
            }
            current_num_elmts
               = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
            max_off_proc_elmts
               = hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
            off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
            off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
            off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
            off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);

            if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = hypre_max(n, 1000);
               hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
               hypre_AuxParCSRMatrixOffProcI(aux_matrix)
                  = hypre_CTAlloc(HYPRE_BigInt, 2 * max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
                  = hypre_CTAlloc(HYPRE_BigInt, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixOffProcData(aux_matrix)
                  = hypre_CTAlloc(HYPRE_Complex, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
               off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
               off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
            }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3 * n;
               off_proc_i = hypre_TReAlloc(off_proc_i, HYPRE_BigInt, 2 * max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_j = hypre_TReAlloc(off_proc_j, HYPRE_BigInt, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_data = hypre_TReAlloc(off_proc_data, HYPRE_Complex,
                                              max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) = max_off_proc_elmts;
               hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
               hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
               hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
            }

            /* AB - 4/6 - the row should be negative to indicate an add */
            /* UMY - 12/28/09 - now positive since we eliminated the feature of
               setting on other processors */
            /* off_proc_i[off_proc_i_indx++] = row; */
            off_proc_i[off_proc_i_indx++] = row;

            off_proc_i[off_proc_i_indx++] = n;
            for (i = 0; i < n; i++)
            {
               off_proc_j[current_num_elmts] = cols[indx];
               off_proc_data[current_num_elmts++] = values[indx++];
            }
            hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx;
            hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix)
               = current_num_elmts;
         }
      }
   }

   /* not assembled */
   else
   {
      aux_matrix = (hypre_AuxParCSRMatrix *) hypre_IJMatrixTranslator(matrix);
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      for (ii = 0; ii < nrows; ii++)
      {
         row = rows[ii];
         n = ncols ? ncols[ii] : 1;
         if (n == 0) /* empty row */
         {
            continue;
         }
         indx = row_indexes[ii];
         if (row >= row_partitioning[0] && row < row_partitioning[1])
         {
            row_local = (HYPRE_Int)(row - row_partitioning[0]);
            /* compute local row number */
            if (need_aux)
            {
               aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
               aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
               local_j = aux_j[row_local];
               local_data = aux_data[row_local];
               space = row_space[row_local];
               old_size = row_length[row_local];
               size = space - old_size;
               if (size < n)
               {
                  size = n - size;
                  tmp_j = hypre_CTAlloc(HYPRE_BigInt, size, HYPRE_MEMORY_HOST);
                  tmp_data = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
               }
               else
               {
                  tmp_j = NULL;
               }
               tmp_indx = 0;
               not_found = 1;
               size = old_size;
               for (i = 0; i < n; i++)
               {
                  for (j = 0; j < old_size; j++)
                  {
                     if (local_j[j] == cols[indx])
                     {
                        local_data[j] += values[indx];
                        not_found = 0;
                        break;
                     }
                  }
                  if (not_found)
                  {
                     if (size < space)
                     {
                        local_j[size] = cols[indx];
                        local_data[size++] = values[indx];
                     }
                     else
                     {
                        tmp_j[tmp_indx] = cols[indx];
                        tmp_data[tmp_indx++] = values[indx];
                     }
                  }
                  not_found = 1;
                  indx++;
               }

               row_length[row_local] = size + tmp_indx;

               if (tmp_indx)
               {
                  aux_j[row_local] = hypre_TReAlloc(aux_j[row_local], HYPRE_BigInt,
                                                    size + tmp_indx, HYPRE_MEMORY_HOST);
                  aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
                                                       HYPRE_Complex, size + tmp_indx,
                                                       HYPRE_MEMORY_HOST);
                  row_space[row_local] = size + tmp_indx;
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
               }

               cnt = size;

               for (i = 0; i < tmp_indx; i++)
               {
                  local_j[cnt] = tmp_j[i];
                  local_data[cnt++] = tmp_data[i];
               }

               if (tmp_j)
               {
                  hypre_TFree(tmp_j, HYPRE_MEMORY_HOST);
                  hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
               }
            }
            else /* insert immediately into data in ParCSRMatrix structure */
            {
               HYPRE_BigInt *big_offd_j = NULL;

               offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
               diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
               diag = hypre_ParCSRMatrixDiag(par_matrix);
               diag_i = hypre_CSRMatrixI(diag);
               diag_j = hypre_CSRMatrixJ(diag);
               diag_data = hypre_CSRMatrixData(diag);
               offd = hypre_ParCSRMatrixOffd(par_matrix);
               offd_i = hypre_CSRMatrixI(offd);
               if (num_procs > 1)
               {
                  big_offd_j = hypre_CSRMatrixBigJ(offd);
                  offd_data = hypre_CSRMatrixData(offd);
                  if (!big_offd_j)
                  {
                     big_offd_j = hypre_CTAlloc(HYPRE_BigInt,
                                                offd_i[hypre_CSRMatrixNumRows(offd)],
                                                hypre_CSRMatrixMemoryLocation(offd));
                     hypre_CSRMatrixBigJ(offd) = big_offd_j;
                  }
               }
               cnt_diag = diag_indx;
               cnt_offd = offd_indx;
               diag_space = diag_i[row_local + 1];
               offd_space = offd_i[row_local + 1];
               not_found = 1;
               for (i = 0; i < n; i++)
               {
                  if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */
                  {
                     for (j = offd_i[row_local]; j < offd_indx; j++)
                     {
                        if (big_offd_j[j] == cols[indx])
                        {
                           offd_data[j] += values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        if (cnt_offd < offd_space)
                        {
                           big_offd_j[cnt_offd] = cols[indx];
                           offd_data[cnt_offd++] = values[indx];
                        }
                        else
                        {
                           hypre_error(HYPRE_ERROR_GENERIC);
                           if (print_level)
                           {
                              hypre_printf("Error in row %b ! Too many elements!\n",
                                           row);
                           }
                           /* return 1;*/
                           return hypre_error_flag;
                        }
                     }
                     not_found = 1;
                  }
                  else  /* insert into diag */
                  {
                     HYPRE_Int col_j = (HYPRE_Int)( cols[indx] - col_0);
                     for (j = diag_i[row_local]; j < diag_indx; j++)
                     {
                        if (diag_j[j] == col_j)
                        {
                           diag_data[j] += values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        if (cnt_diag < diag_space)
                        {
                           diag_j[cnt_diag] = col_j;
                           diag_data[cnt_diag++] = values[indx];
                        }
                        else
                        {
                           hypre_error(HYPRE_ERROR_GENERIC);
                           if (print_level)
                           {
                              hypre_printf("Error in row %b ! Too many elements !\n",
                                           row);
                           }
                           /* return 1; */
                           return hypre_error_flag;
                        }
                     }
                     not_found = 1;
                  }
                  indx++;
               }

               hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
               hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

            }
         }
         /* not my row */
         else
         {
            current_num_elmts
               = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
            max_off_proc_elmts
               = hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
            off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
            off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
            off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
            off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);

            if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = hypre_max(n, 1000);
               hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
                  max_off_proc_elmts;
               hypre_AuxParCSRMatrixOffProcI(aux_matrix)
                  = hypre_CTAlloc(HYPRE_BigInt, 2 * max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
                  = hypre_CTAlloc(HYPRE_BigInt, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixOffProcData(aux_matrix)
                  = hypre_CTAlloc(HYPRE_Complex, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
               off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
               off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
            }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3 * n;
               off_proc_i = hypre_TReAlloc(off_proc_i, HYPRE_BigInt, 2 * max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_j = hypre_TReAlloc(off_proc_j, HYPRE_BigInt, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_data = hypre_TReAlloc(off_proc_data, HYPRE_Complex,
                                              max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
                  = max_off_proc_elmts;
               hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
               hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
               hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
            }
            off_proc_i[off_proc_i_indx++] = row;
            off_proc_i[off_proc_i_indx++] = n;
            for (i = 0; i < n; i++)
            {
               off_proc_j[current_num_elmts] = cols[indx];
               off_proc_data[current_num_elmts++] = values[indx++];
            }
            hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx;
            hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix)
               = current_num_elmts;
         }
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixDestroyParCSR
 *
 * frees an IJMatrix
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixDestroyParCSR(hypre_IJMatrix *matrix)
{
   hypre_ParCSRMatrixDestroy((hypre_ParCSRMatrix *)hypre_IJMatrixObject(matrix));
   hypre_AuxParCSRMatrixDestroy((hypre_AuxParCSRMatrix*)hypre_IJMatrixTranslator(matrix));

   /* Reset pointers to NULL */
   hypre_IJMatrixObject(matrix)     = NULL;
   hypre_IJMatrixTranslator(matrix) = NULL;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixTransposeParCSR
 *
 * Tranposes an IJMatrix of type ParCSRMatrix
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixTransposeParCSR( hypre_IJMatrix  *matrix_A,
                               hypre_IJMatrix  *matrix_AT )
{
   hypre_ParCSRMatrix *par_A  = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix_A);
   hypre_ParCSRMatrix *par_AT;

   /* Free old object if existent */
   if (hypre_IJMatrixObject(matrix_AT))
   {
      par_AT = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix_AT);
      hypre_ParCSRMatrixDestroy(par_AT);
      hypre_IJMatrixObject(matrix_AT) = NULL;
   }

   hypre_ParCSRMatrixTranspose(par_A, &par_AT, 1);
   hypre_ParCSRMatrixSetNumNonzeros(par_AT);
   hypre_ParCSRMatrixSetDNumNonzeros(par_AT);
   hypre_MatvecCommPkgCreate(par_AT);

   hypre_IJMatrixObject(matrix_AT) = (void *) par_AT;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixNormParCSR
 *
 * Computes the Infinity norm of an IJMatrix of type ParCSRMatrix
 *
 * TODO: Add other norms
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixNormParCSR( hypre_IJMatrix *matrix,
                          HYPRE_Real     *norm )
{
   hypre_ParCSRMatrix *par_matrix = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix);

   hypre_ParCSRMatrixInfNorm(par_matrix, norm);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixAddParCSR
 *
 * Performs C = alpha*A + beta*B, where A, B and C are IJMatrices of
 * type ParCSRMatrix.
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixAddParCSR( HYPRE_Complex    alpha,
                         hypre_IJMatrix  *matrix_A,
                         HYPRE_Complex    beta,
                         hypre_IJMatrix  *matrix_B,
                         hypre_IJMatrix  *matrix_C )
{
   hypre_ParCSRMatrix *par_A  = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix_A);
   hypre_ParCSRMatrix *par_B  = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix_B);
   hypre_ParCSRMatrix *par_C;

   /* Free old object if existent */
   if (hypre_IJMatrixObject(matrix_C))
   {
      par_C = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix_C);
      hypre_ParCSRMatrixDestroy(par_C);
      hypre_IJMatrixObject(matrix_C) = NULL;
   }

   hypre_ParCSRMatrixAdd(alpha, par_A, beta, par_B, &par_C);
   hypre_ParCSRMatrixSetNumNonzeros(par_C);
   hypre_ParCSRMatrixSetDNumNonzeros(par_C);
   if (!hypre_ParCSRMatrixCommPkg(par_C))
   {
      hypre_MatvecCommPkgCreate(par_C);
   }

   hypre_IJMatrixObject(matrix_C) = (void *) par_C;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixAssembleOffProcValsParCSR
 *
 * This is for handling set and get values calls to off-proc. entries -
 * it is called from matrix assemble.  There is an alternate version for
 * when the assumed partition is being used.
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixAssembleOffProcValsParCSR( hypre_IJMatrix       *matrix,
                                         HYPRE_Int             off_proc_i_indx,
                                         HYPRE_Int             max_off_proc_elmts,
                                         HYPRE_Int             current_num_elmts,
                                         HYPRE_MemoryLocation  memory_location,
                                         HYPRE_BigInt         *off_proc_i,
                                         HYPRE_BigInt         *off_proc_j,
                                         HYPRE_Complex        *off_proc_data )
{
   HYPRE_UNUSED_VAR(max_off_proc_elmts);

   MPI_Comm comm = hypre_IJMatrixComm(matrix);

   HYPRE_Int i, j, k, in_i;
   HYPRE_Int myid;

   HYPRE_Int proc_id, last_proc, prev_id, tmp_id;
   HYPRE_Int max_response_size;
   HYPRE_BigInt global_num_cols;
   HYPRE_BigInt global_first_col;
   HYPRE_BigInt global_first_row;
   HYPRE_Int ex_num_contacts = 0, num_rows = 0;
   HYPRE_BigInt range_start, range_end;
   HYPRE_Int num_elements;
   HYPRE_Int storage;
   HYPRE_Int indx;
   HYPRE_BigInt row;
   HYPRE_Int num_ranges, row_index = 0;
   HYPRE_Int num_recvs;
   HYPRE_BigInt upper_bound;
   HYPRE_Int counter;
   HYPRE_Int num_real_procs;
   HYPRE_Int /*current_proc,*/ original_proc_indx;

   HYPRE_BigInt *row_list = NULL;
   HYPRE_BigInt *row_list_num_elements = NULL;
   HYPRE_Int *a_proc_id = NULL, *orig_order = NULL;
   HYPRE_Int *real_proc_id = NULL, *us_real_proc_id = NULL;
   HYPRE_Int *ex_contact_procs = NULL, *ex_contact_vec_starts = NULL;
   HYPRE_BigInt *ex_contact_buf = NULL;
   HYPRE_Int *recv_starts = NULL;
   HYPRE_BigInt *response_buf = NULL;
   HYPRE_Int *response_buf_starts = NULL;
   HYPRE_Int *num_rows_per_proc = NULL, *num_elements_total = NULL;
   HYPRE_Int *argsort_contact_procs = NULL;

   HYPRE_Int  obj_size_bytes, complex_size;
   HYPRE_BigInt big_int_size;
   HYPRE_Int  tmp_int;
   HYPRE_BigInt  tmp_big_int;
   HYPRE_BigInt *col_ptr;
   HYPRE_BigInt *big_int_data = NULL;
   HYPRE_Int big_int_data_size = 0, complex_data_size = 0;

   void *void_contact_buf = NULL;
   void *index_ptr;
   void *recv_data_ptr;

   HYPRE_Complex  tmp_complex;
   HYPRE_Complex *col_data_ptr;
   HYPRE_Complex *complex_data = NULL;

   hypre_DataExchangeResponse  response_obj1, response_obj2;
   hypre_ProcListElements      send_proc_obj;

   hypre_IJAssumedPart   *apart;

   hypre_MPI_Comm_rank(comm, &myid);
   global_num_cols = hypre_IJMatrixGlobalNumCols(matrix);
   global_first_col = hypre_IJMatrixGlobalFirstCol(matrix);
   global_first_row = hypre_IJMatrixGlobalFirstRow(matrix);

   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      HYPRE_BigInt  *tmp             = hypre_TAlloc(HYPRE_BigInt,    current_num_elmts,
                                                    HYPRE_MEMORY_HOST);
      HYPRE_BigInt  *off_proc_i_h    = hypre_TAlloc(HYPRE_BigInt,  2 * current_num_elmts,
                                                    HYPRE_MEMORY_HOST);
      HYPRE_BigInt  *off_proc_j_h    = hypre_TAlloc(HYPRE_BigInt,    current_num_elmts,
                                                    HYPRE_MEMORY_HOST);
      HYPRE_Complex *off_proc_data_h = hypre_TAlloc(HYPRE_Complex,   current_num_elmts,
                                                    HYPRE_MEMORY_HOST);

      hypre_TMemcpy(tmp,             off_proc_i,    HYPRE_BigInt,  current_num_elmts, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(off_proc_j_h,    off_proc_j,    HYPRE_BigInt,  current_num_elmts, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(off_proc_data_h, off_proc_data, HYPRE_Complex, current_num_elmts, HYPRE_MEMORY_HOST,
                    HYPRE_MEMORY_DEVICE);

      for (i = 0; i < current_num_elmts; i++)
      {
#if defined(HYPRE_DEBUG)
         hypre_assert(tmp[i] < hypre_IJMatrixRowPartitioning(matrix)[0] ||
                      tmp[i] >= hypre_IJMatrixRowPartitioning(matrix)[1]);
         hypre_assert(tmp[i] >= global_first_row &&
                      tmp[i] < global_first_row + hypre_IJMatrixGlobalNumRows(matrix));
         hypre_assert(off_proc_j_h[i] >= global_first_col &&
                      off_proc_j_h[i] < global_first_col + global_num_cols);
#endif
         off_proc_i_h[2 * i]   = tmp[i];
         off_proc_i_h[2 * i + 1] = 1;
      }

      off_proc_i_indx = current_num_elmts * 2;

      off_proc_i    = off_proc_i_h;
      off_proc_j    = off_proc_j_h;
      off_proc_data = off_proc_data_h;

      hypre_TFree(tmp, HYPRE_MEMORY_HOST);
   }

   /* call hypre_IJMatrixAddToValuesParCSR directly inside this function
    * with one chunk of data */
   HYPRE_Int      off_proc_nelm_recv_cur = 0;
   HYPRE_Int      off_proc_nelm_recv_max = 0;
   HYPRE_BigInt  *off_proc_i_recv = NULL;
   HYPRE_BigInt  *off_proc_j_recv = NULL;
   HYPRE_Complex *off_proc_data_recv = NULL;
   HYPRE_BigInt  *off_proc_i_recv_d = NULL;
   HYPRE_BigInt  *off_proc_j_recv_d = NULL;
   HYPRE_Complex *off_proc_data_recv_d = NULL;

   num_rows = off_proc_i_indx / 2;

   /* verify that we have created the assumed partition */
   if  (hypre_IJMatrixAssumedPart(matrix) == NULL)
   {
      hypre_IJMatrixCreateAssumedPartition(matrix);
   }

   apart = (hypre_IJAssumedPart*) hypre_IJMatrixAssumedPart(matrix);

   /*if  (hypre_ParCSRMatrixAssumedPartition(par_matrix) == NULL)
     {
     hypre_ParCSRMatrixCreateAssumedPartition(par_matrix);
     }

     apart = hypre_ParCSRMatrixAssumedPartition(par_matrix);*/

   row_list              = hypre_CTAlloc(HYPRE_BigInt, num_rows, HYPRE_MEMORY_HOST);
   row_list_num_elements = hypre_CTAlloc(HYPRE_BigInt, num_rows, HYPRE_MEMORY_HOST);
   a_proc_id             = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
   orig_order            = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);
   real_proc_id          = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);

   /* get the assumed processor id for each row */
   if (num_rows > 0 )
   {
      for (i = 0; i < num_rows; i++)
      {
         row = off_proc_i[i * 2];
         //if (row < 0) row = -row - 1;
         row_list[i] = row;
         row_list_num_elements[i] = off_proc_i[i * 2 + 1];

         hypre_GetAssumedPartitionProcFromRow(comm, row, global_first_row,
                                              global_num_cols, &proc_id);
         a_proc_id[i] = proc_id;
         orig_order[i] = i;
      }

      /* now we need to find the actual order of each row  - sort on row -
         this will result in proc ids sorted also...*/

      hypre_BigQsortb2i(row_list, a_proc_id, orig_order, 0, num_rows - 1);

      /* calculate the number of contacts */
      ex_num_contacts = 1;
      last_proc = a_proc_id[0];
      for (i = 1; i < num_rows; i++)
      {
         if (a_proc_id[i] > last_proc)
         {
            ex_num_contacts++;
            last_proc = a_proc_id[i];
         }
      }
   }

   /* now we will go through a create a contact list - need to contact assumed
      processors and find out who the actual row owner is - we will contact with
      a range (2 numbers) */

   ex_contact_procs = hypre_CTAlloc(HYPRE_Int,  ex_num_contacts, HYPRE_MEMORY_HOST);
   ex_contact_vec_starts =  hypre_CTAlloc(HYPRE_Int,  ex_num_contacts + 1, HYPRE_MEMORY_HOST);
   ex_contact_buf =  hypre_CTAlloc(HYPRE_BigInt,  ex_num_contacts * 2, HYPRE_MEMORY_HOST);

   counter = 0;
   range_end = -1;
   for (i = 0; i < num_rows; i++)
   {
      if (row_list[i] > range_end)
      {
         /* assumed proc */
         proc_id = a_proc_id[i];

         /* end of prev. range */
         if (counter > 0)
         {
            ex_contact_buf[counter * 2 - 1] = row_list[i - 1];
         }

         /*start new range*/
         ex_contact_procs[counter] = proc_id;
         ex_contact_vec_starts[counter] = counter * 2;
         ex_contact_buf[counter * 2] =  row_list[i];
         counter++;

         hypre_GetAssumedPartitionRowRange(comm, proc_id, global_first_col, global_num_cols,
                                           &range_start, &range_end);
      }
   }
   /* finish the starts */
   ex_contact_vec_starts[counter] = counter * 2;
   /* finish the last range */
   if (counter > 0)
   {
      ex_contact_buf[counter * 2 - 1] = row_list[num_rows - 1];
   }

   /* don't allocate space for responses */

   /* create response object - can use same fill response as used in the commpkg
      routine */
   response_obj1.fill_response = hypre_RangeFillResponseIJDetermineRecvProcs;
   response_obj1.data1 =  apart; /* this is necessary so we can fill responses*/
   response_obj1.data2 = NULL;

   max_response_size = 6;  /* 6 means we can fit 3 ranges*/

   hypre_DataExchangeList(ex_num_contacts, ex_contact_procs,
                          ex_contact_buf, ex_contact_vec_starts, sizeof(HYPRE_BigInt),
                          sizeof(HYPRE_BigInt), &response_obj1, max_response_size, 1,
                          comm, (void**) &response_buf, &response_buf_starts);

   /* now response_buf contains a proc_id followed by a range upper bound */

   hypre_TFree(ex_contact_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_vec_starts, HYPRE_MEMORY_HOST);

   hypre_TFree(a_proc_id, HYPRE_MEMORY_HOST);

   /*how many ranges were returned?*/
   num_ranges = response_buf_starts[ex_num_contacts];
   num_ranges = num_ranges / 2;

   prev_id = -1;
   j = 0;
   counter = 0;
   num_real_procs = 0;

   /* loop through ranges - create a list of actual processor ids*/
   for (i = 0; i < num_ranges; i++)
   {
      upper_bound = response_buf[i * 2 + 1];
      counter = 0;
      tmp_id = response_buf[i * 2];

      /* loop through row_list entries - counting how many are in the range */
      while (j < num_rows && row_list[j] <= upper_bound)
      {
         real_proc_id[j] = tmp_id;
         j++;
         counter++;
      }
      if (counter > 0 && tmp_id != prev_id)
      {
         num_real_procs++;
      }

      prev_id = tmp_id;
   }

   /* now we have the list of real processor ids (real_proc_id) - and the number
      of distinct ones - so now we can set up data to be sent - we have
      HYPRE_Int data and HYPRE_Complex data.  that we will need to pack
      together */

   /* first find out how many rows and elements we need to send per proc - so we
      can do storage */

   ex_contact_procs = hypre_CTAlloc(HYPRE_Int,  num_real_procs, HYPRE_MEMORY_HOST);
   num_rows_per_proc = hypre_CTAlloc(HYPRE_Int,  num_real_procs, HYPRE_MEMORY_HOST);
   num_elements_total  =  hypre_CTAlloc(HYPRE_Int,  num_real_procs, HYPRE_MEMORY_HOST);

   counter = 0;

   if (num_real_procs > 0 )
   {
      ex_contact_procs[0] = real_proc_id[0];
      num_rows_per_proc[0] = 1;
      num_elements_total[0] = row_list_num_elements[orig_order[0]];

      /* loop through real procs - these are sorted (row_list is sorted also)*/
      for (i = 1; i < num_rows; i++)
      {
         if (real_proc_id[i] == ex_contact_procs[counter]) /* same processor */
         {
            num_rows_per_proc[counter] += 1; /*another row */
            num_elements_total[counter] += row_list_num_elements[orig_order[i]];
         }
         else /* new processor */
         {
            counter++;
            ex_contact_procs[counter] = real_proc_id[i];
            num_rows_per_proc[counter] = 1;
            num_elements_total[counter] = row_list_num_elements[orig_order[i]];
         }
      }
   }

   /* to pack together, we need to use the largest obj. size of
      (HYPRE_Int) and (HYPRE_Complex) - if these are much different, then we are
      wasting some storage, but I do not think that it will be a
      large amount since this function should not be used on really
      large amounts of data anyway*/
   big_int_size = sizeof(HYPRE_BigInt);
   complex_size = sizeof(HYPRE_Complex);

   obj_size_bytes = (HYPRE_Int) hypre_max(big_int_size, complex_size);

   /* set up data to be sent to send procs */
   /* for each proc, ex_contact_buf contains #rows, row #,
      no. elements, col indicies, col data, row #, no. elements, col
      indicies, col data, etc. */

   /* first calculate total storage and make vec_starts arrays */
   storage = 0;
   ex_contact_vec_starts = hypre_CTAlloc(HYPRE_Int,  num_real_procs + 1, HYPRE_MEMORY_HOST);
   ex_contact_vec_starts[0] = -1;

   for (i = 0; i < num_real_procs; i++)
   {
      storage += 1 + 2 * num_rows_per_proc[i] + 2 * num_elements_total[i];
      ex_contact_vec_starts[i + 1] = -storage - 1; /* need negative for next loop */
   }

   hypre_TFree(num_elements_total, HYPRE_MEMORY_HOST);

   /*void_contact_buf = hypre_TAlloc(char, storage*obj_size_bytes);*/
   void_contact_buf = hypre_CTAlloc(char, storage * obj_size_bytes, HYPRE_MEMORY_HOST);
   index_ptr = void_contact_buf; /* step through with this index */

   /* for each proc: #rows, row #, no. elements,
      col indicies, col data, row #, no. elements, col indicies, col data, etc. */

   /* un-sort real_proc_id - we want to access data arrays in order, so
      cheaper to do this*/
   us_real_proc_id =  hypre_CTAlloc(HYPRE_Int,  num_rows, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_rows; i++)
   {
      us_real_proc_id[orig_order[i]] = real_proc_id[i];
   }
   hypre_TFree(real_proc_id, HYPRE_MEMORY_HOST);

   counter = 0; /* index into data arrays */
   prev_id = -1;
   for (i = 0; i < num_rows; i++)
   {
      proc_id = us_real_proc_id[i];
      /* can't use row list[i] - you loose the negative signs that differentiate
         add/set values */
      row = off_proc_i[i * 2];
      num_elements = row_list_num_elements[i];
      /* find position of this processor */
      indx = hypre_BinarySearch(ex_contact_procs, proc_id, num_real_procs);
      in_i = ex_contact_vec_starts[indx];

      index_ptr = (void *) ((char *) void_contact_buf + in_i * obj_size_bytes);

      /* first time for this processor - add the number of rows to the buffer */
      if (in_i < 0)
      {
         in_i = -in_i - 1;
         /* re-calc. index_ptr since in_i was negative */
         index_ptr = (void *) ((char *) void_contact_buf + in_i * obj_size_bytes);

         tmp_int =  num_rows_per_proc[indx];
         hypre_TMemcpy( index_ptr,  &tmp_int, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);

         in_i++;
      }
      /* add row # */
      hypre_TMemcpy( index_ptr,  &row, HYPRE_BigInt, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in_i++;

      /* add number of elements */
      hypre_TMemcpy( index_ptr,  &num_elements, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
      in_i++;

      /* now add col indices  */
      for (j = 0; j < num_elements; j++)
      {
         tmp_big_int = off_proc_j[counter + j]; /* col number */

         hypre_TMemcpy( index_ptr,  &tmp_big_int, HYPRE_BigInt, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
         in_i ++;
      }

      /* now add data */
      for (j = 0; j < num_elements; j++)
      {
         tmp_complex = off_proc_data[counter++]; /* value */

         hypre_TMemcpy( index_ptr,  &tmp_complex, HYPRE_Complex, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         index_ptr = (void *) ((char *) index_ptr + obj_size_bytes);
         in_i++;
      }

      /* increment the indexes to keep track of where we are - we
       * adjust below to be actual starts*/
      ex_contact_vec_starts[indx] = in_i;
   }

   /* some clean up */

   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);

   hypre_TFree(us_real_proc_id, HYPRE_MEMORY_HOST);
   hypre_TFree(orig_order, HYPRE_MEMORY_HOST);
   hypre_TFree(row_list, HYPRE_MEMORY_HOST);
   hypre_TFree(row_list_num_elements, HYPRE_MEMORY_HOST);
   hypre_TFree(num_rows_per_proc, HYPRE_MEMORY_HOST);

   for (i = num_real_procs; i > 0; i--)
   {
      ex_contact_vec_starts[i] =   ex_contact_vec_starts[i - 1];
   }

   ex_contact_vec_starts[0] = 0;

   /* now send the data */

   /***********************************/

   /* first get the integer info in send_proc_obj */

   /* the response we expect is just a confirmation*/
   response_buf = NULL;
   response_buf_starts = NULL;

   /*build the response object*/

   /* use the send_proc_obj for the info kept from contacts */
   /*estimate inital storage allocation */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = num_real_procs + 5;
   send_proc_obj.id =
      hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length + 1, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts =
      hypre_CTAlloc(HYPRE_Int,  send_proc_obj.storage_length + 1, HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = storage + 20;
   send_proc_obj.v_elements =
      hypre_TAlloc(char, obj_size_bytes * send_proc_obj.element_storage_length, HYPRE_MEMORY_HOST);

   response_obj2.fill_response = hypre_FillResponseIJOffProcVals;
   response_obj2.data1 = NULL;
   response_obj2.data2 = &send_proc_obj;

   max_response_size = 0;

   hypre_DataExchangeList(num_real_procs, ex_contact_procs,
                          void_contact_buf, ex_contact_vec_starts, obj_size_bytes,
                          0, &response_obj2, max_response_size, 2,
                          comm,  (void **) &response_buf, &response_buf_starts);

   hypre_TFree(response_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(response_buf_starts, HYPRE_MEMORY_HOST);

   hypre_TFree(ex_contact_procs, HYPRE_MEMORY_HOST);
   hypre_TFree(void_contact_buf, HYPRE_MEMORY_HOST);
   hypre_TFree(ex_contact_vec_starts, HYPRE_MEMORY_HOST);

   /* Now we can unpack the send_proc_objects and call set
      and add to values functions.  We unpack messages in a
      deterministic order, using processor rank */

   num_recvs = send_proc_obj.length;
   argsort_contact_procs = hypre_CTAlloc(HYPRE_Int,  num_recvs, HYPRE_MEMORY_HOST);
   for (i = 0; i < num_recvs; i++)
   {
      argsort_contact_procs[i] = i;
   }
   /* This sort's the id array, but the original indices are stored in
    * argsort_contact_procs */
   hypre_qsort2i( send_proc_obj.id, argsort_contact_procs, 0, num_recvs - 1 );

   /* alias */
   recv_data_ptr = send_proc_obj.v_elements;
   recv_starts = send_proc_obj.vec_starts;

   for (i = 0; i < num_recvs; i++)
   {

      /* Find the current processor in order, and reset recv_data_ptr to that processor's message */
      original_proc_indx = argsort_contact_procs[i];
      /*current_proc = send_proc_obj.id[i];*/
      indx = recv_starts[original_proc_indx];
      recv_data_ptr = (void *) ((char *) send_proc_obj.v_elements + indx * obj_size_bytes);

      /* get the number of rows for this recv */
      hypre_TMemcpy( &num_rows, recv_data_ptr, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
      indx++;


      for (j = 0; j < num_rows; j++) /* for each row: unpack info */
      {
         /* row # */
         hypre_TMemcpy( &row,  recv_data_ptr, HYPRE_BigInt, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* num elements for this row */
         hypre_TMemcpy( &num_elements,  recv_data_ptr, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
         indx++;

         /* col indices */ /* Need to check this again !!!! */
         if (big_int_size == obj_size_bytes)
         {
            col_ptr = (HYPRE_BigInt *) recv_data_ptr;
            recv_data_ptr = (void *) ((char *)recv_data_ptr + num_elements * obj_size_bytes);
         }
         else /* copy data */
         {
            if (big_int_data_size < num_elements)
            {
               big_int_data = hypre_TReAlloc(big_int_data,  HYPRE_BigInt,  num_elements + 10, HYPRE_MEMORY_HOST);
            }
            for (k = 0; k < num_elements; k++)
            {
               hypre_TMemcpy( &big_int_data[k],  recv_data_ptr, HYPRE_BigInt, 1, HYPRE_MEMORY_HOST,
                              HYPRE_MEMORY_HOST);
               recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
            }
            col_ptr = big_int_data;
         }

         /* col data */
         if (complex_size == obj_size_bytes)
         {
            col_data_ptr = (HYPRE_Complex *) recv_data_ptr;
            recv_data_ptr = (void *) ((char *)recv_data_ptr + num_elements * obj_size_bytes);
         }
         else /* copy data */
         {
            if (complex_data_size < num_elements)
            {
               complex_data =
                  hypre_TReAlloc(complex_data,  HYPRE_Complex,  num_elements + 10, HYPRE_MEMORY_HOST);
            }
            for (k = 0; k < num_elements; k++)
            {
               hypre_TMemcpy( &complex_data[k],  recv_data_ptr, HYPRE_Complex, 1, HYPRE_MEMORY_HOST,
                              HYPRE_MEMORY_HOST);
               recv_data_ptr = (void *) ((char *)recv_data_ptr + obj_size_bytes);
            }
            col_data_ptr = complex_data;

         }

         if (memory_location == HYPRE_MEMORY_HOST)
         {
            hypre_IJMatrixAddToValuesParCSR(matrix, 1, &num_elements, &row, &row_index, col_ptr, col_data_ptr);
         }
         else
         {
            HYPRE_Int nelm_new = off_proc_nelm_recv_cur + num_elements;

            if (nelm_new > off_proc_nelm_recv_max)
            {
               off_proc_nelm_recv_max = nelm_new * 2;
               off_proc_i_recv    = hypre_TReAlloc(off_proc_i_recv,    HYPRE_BigInt,  off_proc_nelm_recv_max,
                                                   HYPRE_MEMORY_HOST);
               off_proc_j_recv    = hypre_TReAlloc(off_proc_j_recv,    HYPRE_BigInt,  off_proc_nelm_recv_max,
                                                   HYPRE_MEMORY_HOST);
               off_proc_data_recv = hypre_TReAlloc(off_proc_data_recv, HYPRE_Complex, off_proc_nelm_recv_max,
                                                   HYPRE_MEMORY_HOST);
            }

            HYPRE_Int i;
            for (i = 0; i < num_elements; i++)
            {
               off_proc_i_recv[off_proc_nelm_recv_cur + i] = row;
            }
            hypre_TMemcpy(off_proc_j_recv + off_proc_nelm_recv_cur, col_ptr, HYPRE_BigInt, num_elements,
                          HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
            hypre_TMemcpy(off_proc_data_recv + off_proc_nelm_recv_cur, col_data_ptr, HYPRE_Complex,
                          num_elements,
                          HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

            off_proc_nelm_recv_cur = nelm_new;
         }

         indx += (num_elements * 2);
      }
   }

   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      off_proc_i_recv_d    = hypre_TAlloc(HYPRE_BigInt,  off_proc_nelm_recv_cur, HYPRE_MEMORY_DEVICE);
      off_proc_j_recv_d    = hypre_TAlloc(HYPRE_BigInt,  off_proc_nelm_recv_cur, HYPRE_MEMORY_DEVICE);
      off_proc_data_recv_d = hypre_TAlloc(HYPRE_Complex, off_proc_nelm_recv_cur, HYPRE_MEMORY_DEVICE);

      hypre_TMemcpy(off_proc_i_recv_d,    off_proc_i_recv,    HYPRE_BigInt,  off_proc_nelm_recv_cur,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(off_proc_j_recv_d,    off_proc_j_recv,    HYPRE_BigInt,  off_proc_nelm_recv_cur,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(off_proc_data_recv_d, off_proc_data_recv, HYPRE_Complex, off_proc_nelm_recv_cur,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_GPU)
      hypre_IJMatrixSetAddValuesParCSRDevice(matrix, off_proc_nelm_recv_cur, NULL, off_proc_i_recv_d,
                                             NULL, off_proc_j_recv_d,
                                             off_proc_data_recv_d, "add");
#endif
   }

   hypre_TFree(send_proc_obj.v_elements, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.vec_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(send_proc_obj.id, HYPRE_MEMORY_HOST);
   hypre_TFree(argsort_contact_procs, HYPRE_MEMORY_HOST);

   if (big_int_data)
   {
      hypre_TFree(big_int_data, HYPRE_MEMORY_HOST);
   }

   if (complex_data)
   {
      hypre_TFree(complex_data, HYPRE_MEMORY_HOST);
   }

   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      hypre_TFree(off_proc_i,    HYPRE_MEMORY_HOST);
      hypre_TFree(off_proc_j,    HYPRE_MEMORY_HOST);
      hypre_TFree(off_proc_data, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(off_proc_i_recv,    HYPRE_MEMORY_HOST);
   hypre_TFree(off_proc_j_recv,    HYPRE_MEMORY_HOST);
   hypre_TFree(off_proc_data_recv, HYPRE_MEMORY_HOST);

   hypre_TFree(off_proc_i_recv_d,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(off_proc_j_recv_d,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(off_proc_data_recv_d, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_FillResponseIJOffProcVals
 * Fill response function for the previous function (2nd data exchange)
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FillResponseIJOffProcVals(void      *p_recv_contact_buf,
                                HYPRE_Int  contact_size,
                                HYPRE_Int  contact_proc,
                                void      *ro,
                                MPI_Comm   comm,
                                void     **p_send_response_buf,
                                HYPRE_Int *response_message_size )


{
   HYPRE_UNUSED_VAR(p_send_response_buf);

   HYPRE_Int    myid;
   HYPRE_Int    index, count, elength;

   HYPRE_Int object_size;
   void *index_ptr;

   hypre_DataExchangeResponse  *response_obj = (hypre_DataExchangeResponse*) ro;

   hypre_ProcListElements      *send_proc_obj = (hypre_ProcListElements*) response_obj->data2;

   object_size = hypre_max(sizeof(HYPRE_BigInt), sizeof(HYPRE_Complex));

   hypre_MPI_Comm_rank(comm, &myid );


   /*check to see if we need to allocate more space in send_proc_obj for vec starts
    * and id */
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 20; /*add space for 20 more contact*/
      send_proc_obj->vec_starts = hypre_TReAlloc(send_proc_obj->vec_starts, HYPRE_Int,
                                                 send_proc_obj->storage_length + 1, HYPRE_MEMORY_HOST);
      if ( send_proc_obj->id != NULL)
      {
         send_proc_obj->id = hypre_TReAlloc(send_proc_obj->id, HYPRE_Int,
                                            send_proc_obj->storage_length + 1, HYPRE_MEMORY_HOST);
      }
   }

   /*initialize*/
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /* current number of elements */
   if ( send_proc_obj->id != NULL)
   {
      send_proc_obj->id[count] = contact_proc;
   }

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = hypre_max(contact_size, 100);
      elength += index;
      send_proc_obj->v_elements = hypre_TReAlloc((char*)send_proc_obj->v_elements,
                                                 char, elength * object_size, HYPRE_MEMORY_HOST);
      send_proc_obj->element_storage_length = elength;
   }
   /*populate send_proc_obj*/
   index_ptr = (void *) ((char *) send_proc_obj->v_elements + index * object_size);

   hypre_TMemcpy(index_ptr,  p_recv_contact_buf, char, object_size * contact_size, HYPRE_MEMORY_HOST,
                 HYPRE_MEMORY_HOST);

   send_proc_obj->vec_starts[count + 1] = index + contact_size;
   send_proc_obj->length++;

   /* output - no message to return (confirmation) */
   *response_message_size = 0;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------*/

HYPRE_Int hypre_FindProc(HYPRE_BigInt *list, HYPRE_BigInt value, HYPRE_Int list_length)
{
   HYPRE_Int low, high, m;

   low = 0;
   high = list_length;
   if (value >= list[high] || value < list[low])
   {
      return -1;
   }
   else
   {
      while (low + 1 < high)
      {
         m = (low + high) / 2;
         if (value < list[m])
         {
            high = m;
         }
         else if (value >= list[m])
         {
            low = m;
         }
      }
      return low;
   }
}

/******************************************************************************
 *
 * hypre_IJMatrixAssembleParCSR
 *
 * assembles IJMatrix from AuxParCSRMatrix auxiliary structure
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixAssembleParCSR(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   hypre_ParCSRMatrix *par_matrix = (hypre_ParCSRMatrix*) hypre_IJMatrixObject(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);
   HYPRE_BigInt *row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   HYPRE_BigInt *col_partitioning = hypre_IJMatrixColPartitioning(matrix);

   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(par_matrix);
   HYPRE_Int *diag_i = hypre_CSRMatrixI(diag);
   HYPRE_Int *offd_i = hypre_CSRMatrixI(offd);
   HYPRE_Int *diag_j;
   HYPRE_Int *offd_j = NULL;
   HYPRE_Complex *diag_data;
   HYPRE_Complex *offd_data = NULL;
   HYPRE_Int i, j, j0;
   HYPRE_Int num_cols_offd;
   HYPRE_Int *diag_pos;
   HYPRE_BigInt *col_map_offd;
   HYPRE_Int *rownnz;
   HYPRE_Int *row_length;
   HYPRE_BigInt **aux_j;
   HYPRE_Complex **aux_data;
   HYPRE_Int my_id, num_procs;
   HYPRE_Int num_rows;
   HYPRE_Int num_rownnz;
   HYPRE_Int i_diag, i_offd;
   HYPRE_BigInt col_0, col_n;
   HYPRE_Int nnz_offd;
   HYPRE_BigInt *big_offd_j;
   HYPRE_BigInt *tmp_j;
   HYPRE_Complex temp;
   HYPRE_BigInt base = hypre_IJMatrixGlobalFirstCol(matrix);
   HYPRE_Int off_proc_i_indx;
   HYPRE_Int max_off_proc_elmts;
   HYPRE_Int current_num_elmts;
   HYPRE_BigInt *off_proc_i;
   HYPRE_BigInt *off_proc_j;
   HYPRE_Complex *off_proc_data;
   HYPRE_Int offd_proc_elmts;
   //HYPRE_Int new_off_proc_i_indx;
   //HYPRE_Int cancel_indx;
   //HYPRE_Int col_indx;
   //HYPRE_Int current_indx;
   //HYPRE_Int current_i;
   //HYPRE_Int row_len;
   HYPRE_Int max_num_threads;
   HYPRE_Int aux_flag, aux_flag_global;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   max_num_threads = hypre_NumThreads();

   /* first find out if anyone has an aux_matrix, and create one if you don't
    * have one, but other procs do */
   aux_flag = 0;
   aux_flag_global = 0;
   if (aux_matrix)
   {
      aux_flag = 1;
   }
   hypre_MPI_Allreduce(&aux_flag, &aux_flag_global, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   if (aux_flag_global && (!aux_flag))
   {
      hypre_MPI_Comm_rank(comm, &my_id);
      num_rows = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
      hypre_AuxParCSRMatrixCreate(&aux_matrix, num_rows, num_rows, NULL);
      hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }

   if (aux_matrix)
   {
      /* first delete all cancelled elements */
      /*cancel_indx = hypre_AuxParCSRMatrixCancelIndx(aux_matrix);
      if (cancel_indx)
      {
         current_num_elmts=hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
         off_proc_i=hypre_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j=hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
         off_proc_data=hypre_AuxParCSRMatrixOffProcData(aux_matrix);
         off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         col_indx = 0;
         current_i = 0;
         current_indx = 0;
         new_off_proc_i_indx = off_proc_i_indx;
         for (i=0; i < off_proc_i_indx; i= i+2)
         {
            row_len = off_proc_i[i+1];
            for (j=0; j < off_proc_i[i+1]; j++)
            {
               if (off_proc_j[col_indx] == -1)
               {
                  col_indx++;
                  row_len--;
                  current_num_elmts--;
               }
               else
               {
                  off_proc_j[current_indx] = off_proc_j[col_indx];
                  off_proc_data[current_indx++] = off_proc_data[col_indx++];
               }
            }
            if (row_len)
            {
               off_proc_i[current_i] = off_proc_i[i];
               off_proc_i[current_i+1] = row_len;
               current_i += 2;
            }
            else
            {
               new_off_proc_i_indx -= 2;
            }
         }
         hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = new_off_proc_i_indx;
         hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) = current_num_elmts;
      }*/
      off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
      hypre_MPI_Allreduce(&off_proc_i_indx, &offd_proc_elmts, 1, HYPRE_MPI_INT,
                          hypre_MPI_SUM, comm);
      if (offd_proc_elmts)
      {
         max_off_proc_elmts = hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
         current_num_elmts = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
         off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
         off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
         hypre_IJMatrixAssembleOffProcValsParCSR(
            matrix, off_proc_i_indx, max_off_proc_elmts, current_num_elmts,
            HYPRE_MEMORY_HOST,
            off_proc_i, off_proc_j, off_proc_data);
      }
   }

   if (hypre_IJMatrixAssembleFlag(matrix) == 0)
   {
      hypre_MPI_Comm_size(comm, &num_procs);
      hypre_MPI_Comm_rank(comm, &my_id);
      num_rows = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
      col_0 = col_partitioning[0];
      col_n = col_partitioning[1] - 1;
      /* move data into ParCSRMatrix if not there already */
      if (hypre_AuxParCSRMatrixNeedAux(aux_matrix))
      {
         HYPRE_Int  *diag_array;
         HYPRE_Int  *offd_array;

         /* Update nonzero rows of aux_matrix */
         hypre_AuxParCSRMatrixSetRownnz(aux_matrix);

         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
         row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
         num_rownnz = hypre_AuxParCSRMatrixLocalNumRownnz(aux_matrix);
         rownnz     = hypre_AuxParCSRMatrixRownnz(aux_matrix);

         diag_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);
         offd_array = hypre_CTAlloc(HYPRE_Int, max_num_threads, HYPRE_MEMORY_HOST);
         diag_pos   = hypre_TAlloc(HYPRE_Int, num_rownnz, HYPRE_MEMORY_HOST);

         i_diag = i_offd = 0;
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel private(i, j, i_diag, i_offd)
#endif
         {
            HYPRE_BigInt *local_j;
            HYPRE_Complex *local_data;
            HYPRE_Int ii, rest, size, ns, ne;
            HYPRE_Int num_threads, my_thread_num;
            num_threads = hypre_NumActiveThreads();
            my_thread_num = hypre_GetThreadNum();

            size = num_rownnz / num_threads;
            rest = num_rownnz - size * num_threads;

            if (my_thread_num < rest)
            {
               ns = my_thread_num * (size + 1);
               ne = (my_thread_num + 1) * (size + 1);
            }
            else
            {
               ns = my_thread_num * size + rest;
               ne = (my_thread_num + 1) * size + rest;
            }

            i_diag = i_offd = 0;
            for (i = ns; i < ne; i++)
            {
               ii = rownnz ? rownnz[i] : i;
               local_j = aux_j[ii];
               local_data = aux_data[ii];
               diag_pos[i] = -1;
               for (j = 0; j < row_length[ii]; j++)
               {
                  if (local_j[j] < col_0 || local_j[j] > col_n)
                  {
                     i_offd++;
                  }
                  else
                  {
                     i_diag++;
                     if ((HYPRE_Int)(local_j[j] - col_0) == i)
                     {
                        diag_pos[i] = j;
                     }
                  }
               }
            }
            diag_array[my_thread_num] = i_diag;
            offd_array[my_thread_num] = i_offd;
#ifdef HYPRE_USING_OPENMP
            #pragma omp barrier
#endif
            if (my_thread_num == 0)
            {
               i_diag = 0;
               i_offd = 0;
               for (i = 0; i < num_threads; i++)
               {
                  i_diag += diag_array[i];
                  i_offd += offd_array[i];
                  diag_array[i] = i_diag;
                  offd_array[i] = i_offd;
               }
               diag_i[num_rows] = i_diag;
               offd_i[num_rows] = i_offd;

               hypre_TFree(hypre_CSRMatrixJ(diag),    hypre_CSRMatrixMemoryLocation(diag));
               hypre_TFree(hypre_CSRMatrixData(diag), hypre_CSRMatrixMemoryLocation(diag));
               hypre_TFree(hypre_CSRMatrixJ(offd),    hypre_CSRMatrixMemoryLocation(offd));
               hypre_TFree(hypre_CSRMatrixData(offd), hypre_CSRMatrixMemoryLocation(offd));
               hypre_TFree(hypre_CSRMatrixBigJ(offd), hypre_CSRMatrixMemoryLocation(offd));

               diag_j     = hypre_CTAlloc(HYPRE_Int,     i_diag, hypre_CSRMatrixMemoryLocation(diag));
               diag_data  = hypre_CTAlloc(HYPRE_Complex, i_diag, hypre_CSRMatrixMemoryLocation(diag));
               offd_j     = hypre_CTAlloc(HYPRE_Int,     i_offd, hypre_CSRMatrixMemoryLocation(offd));
               offd_data  = hypre_CTAlloc(HYPRE_Complex, i_offd, hypre_CSRMatrixMemoryLocation(offd));
               big_offd_j = hypre_CTAlloc(HYPRE_BigInt,  i_offd, hypre_CSRMatrixMemoryLocation(offd));
            }
#ifdef HYPRE_USING_OPENMP
            #pragma omp barrier
#endif
            if (my_thread_num)
            {
               i_diag = diag_array[my_thread_num - 1];
               i_offd = offd_array[my_thread_num - 1];
            }
            else
            {
               i_diag = 0;
               i_offd = 0;
            }

            for (i = ns; i < ne; i++)
            {
               ii = rownnz ? rownnz[i] : i;
               diag_i[ii] = i_diag;
               offd_i[ii] = i_offd;
               local_j = aux_j[ii];
               local_data = aux_data[ii];
               if (diag_pos[i] > -1)
               {
                  diag_j[i_diag] = (HYPRE_Int)(local_j[diag_pos[i]] - col_0);
                  diag_data[i_diag++] = local_data[diag_pos[i]];
               }
               for (j = 0; j < row_length[ii]; j++)
               {
                  if (local_j[j] < col_0 || local_j[j] > col_n)
                  {
                     big_offd_j[i_offd] = local_j[j];
                     offd_data[i_offd++] = local_data[j];
                  }
                  else if (j != diag_pos[i])
                  {
                     diag_j[i_diag] = (HYPRE_Int)(local_j[j] - col_0);
                     diag_data[i_diag++] = local_data[j];
                  }
               }
            }

            /* Correct diag_i and offd_i */
            if (rownnz != NULL)
            {
#ifdef HYPRE_USING_OPENMP
               #pragma omp barrier
#endif
               for (i = ns; i < (ne - 1); i++)
               {
                  for (ii = rownnz[i] + 1; ii < rownnz[i + 1]; ii++)
                  {
                     diag_i[ii] = diag_i[rownnz[i + 1]];
                     offd_i[ii] = offd_i[rownnz[i + 1]];
                  }
               }

               if (my_thread_num < (num_threads - 1))
               {
                  for (ii = rownnz[ne - 1] + 1; ii < rownnz[ne]; ii++)
                  {
                     diag_i[ii] = diag_i[rownnz[ne]];
                     offd_i[ii] = offd_i[rownnz[ne]];
                  }
               }
               else
               {
                  for (ii = rownnz[ne - 1] + 1; ii < num_rows; ii++)
                  {
                     diag_i[ii] = diag_i[num_rows];
                     offd_i[ii] = offd_i[num_rows];
                  }
               }
            }
         } /* end parallel region */

         hypre_TFree(diag_array, HYPRE_MEMORY_HOST);
         hypre_TFree(offd_array, HYPRE_MEMORY_HOST);

         hypre_CSRMatrixJ(diag) = diag_j;
         hypre_CSRMatrixData(diag) = diag_data;
         hypre_CSRMatrixNumNonzeros(diag) = diag_i[num_rows];
         if (offd_i[num_rows] > 0)
         {
            hypre_CSRMatrixJ(offd) = offd_j;
            hypre_CSRMatrixBigJ(offd) = big_offd_j;
            hypre_CSRMatrixData(offd) = offd_data;
         }
         hypre_CSRMatrixNumNonzeros(offd) = offd_i[num_rows];
         hypre_TFree(diag_pos, HYPRE_MEMORY_HOST);
      }
      else
      {
         /* move diagonal element into first space */
         big_offd_j = hypre_CSRMatrixBigJ(offd);
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private (i,j,j0,temp)
#endif
         for (i = 0; i < num_rows; i++)
         {
            j0 = diag_i[i];
            for (j = j0; j < diag_i[i + 1]; j++)
            {
               if (diag_j[j] == i)
               {
                  temp = diag_data[j0];
                  diag_data[j0] = diag_data[j];
                  diag_data[j] = temp;
                  diag_j[j] = diag_j[j0];
                  diag_j[j0] = i;
                  break;
               }
            }
         }

         offd_j = hypre_CSRMatrixJ(offd);
         if (!offd_j && offd_i[num_rows])
         {
            offd_j = hypre_CTAlloc(HYPRE_Int, offd_i[num_rows], hypre_CSRMatrixMemoryLocation(offd));
            hypre_CSRMatrixJ(offd) = offd_j;
         }
      }

      /*  generate col_map_offd */
      nnz_offd = offd_i[num_rows];
      if (nnz_offd)
      {
         tmp_j = hypre_CTAlloc(HYPRE_BigInt, nnz_offd, HYPRE_MEMORY_HOST);
         for (i = 0; i < nnz_offd; i++)
         {
            tmp_j[i] = big_offd_j[i];
         }
         hypre_BigQsort0(tmp_j, 0, nnz_offd - 1);
         num_cols_offd = 1;
         for (i = 0; i < nnz_offd - 1; i++)
         {
            if (tmp_j[i + 1] > tmp_j[i])
            {
               tmp_j[num_cols_offd++] = tmp_j[i + 1];
            }
         }
         col_map_offd = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_cols_offd; i++)
         {
            col_map_offd[i] = tmp_j[i];
         }
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i)
#endif
         for (i = 0; i < nnz_offd; i++)
         {
            offd_j[i] = hypre_BigBinarySearch(col_map_offd, big_offd_j[i], num_cols_offd);
         }

         if (base)
         {
            for (i = 0; i < num_cols_offd; i++)
            {
               col_map_offd[i] -= base;
            }
         }
         hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;
         hypre_CSRMatrixNumCols(offd) = num_cols_offd;
         hypre_TFree(tmp_j, HYPRE_MEMORY_HOST);
         hypre_TFree(big_offd_j, hypre_CSRMatrixMemoryLocation(offd));
         hypre_CSRMatrixBigJ(offd) = NULL;
      }
      hypre_IJMatrixAssembleFlag(matrix) = 1;

      /* Generate the nonzero rows in the diag and offd matrices */
      hypre_CSRMatrixSetRownnz(diag);
      hypre_CSRMatrixSetRownnz(offd);
   }

   /* Free memory */
   hypre_AuxParCSRMatrixDestroy(aux_matrix);
   hypre_IJMatrixTranslator(matrix) = NULL;

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/******************************************************************************
 *
 * IJMatrix_ParCSR interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"

#include "../HYPRE.h"


/******************************************************************************
 *
 * hypre_IJMatrixSetValuesOMPParCSR
 *
 * sets values in an IJMatrix before assembly,
 * use of this routine requires that the values in rows are different from each
 * other, i.e rows[i] != rows[j] for i != j
 * to ensure accurate threading
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixSetValuesOMPParCSR( hypre_IJMatrix       *matrix,
                                  HYPRE_Int             nrows,
                                  HYPRE_Int            *ncols,
                                  const HYPRE_BigInt   *rows,
                                  const HYPRE_Int      *row_indexes,
                                  const HYPRE_BigInt   *cols,
                                  const HYPRE_Complex  *values )
{
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_Int num_procs, my_id;
   HYPRE_BigInt col_0, col_n, first;
   //HYPRE_Int cancel_indx;
   HYPRE_BigInt **aux_j;
   HYPRE_Complex **aux_data;
   HYPRE_Int *row_length, *row_space;
   HYPRE_Int need_aux;
   HYPRE_Int *diag_i = NULL;
   HYPRE_Int *diag_j = NULL;
   HYPRE_Complex *diag_data = NULL;
   HYPRE_Int *offd_i = NULL;
   HYPRE_Int *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Complex *offd_data = NULL;
   /*HYPRE_Int current_num_elmts;*/
   /*HYPRE_Int max_off_proc_elmts;*/
   //HYPRE_Int off_proc_i_indx;
   //HYPRE_BigInt *off_proc_i;
   //HYPRE_BigInt *off_proc_j;
   //HYPRE_Int *offproc_cnt;

   HYPRE_Int print_level = hypre_IJMatrixPrintLevel(matrix);
   //HYPRE_Int max_num_threads;
   //HYPRE_Int error_flag = 0;

   /*HYPRE_Complex *off_proc_data;*/
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   //max_num_threads = hypre_NumThreads();
   par_matrix = (hypre_ParCSRMatrix *) hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);

   //offproc_cnt = hypre_CTAlloc(HYPRE_Int,  max_num_threads, HYPRE_MEMORY_HOST);

   col_0 = col_partitioning[0];
   col_n = col_partitioning[1] - 1;
   first =  hypre_IJMatrixGlobalFirstCol(matrix);
   if (nrows < 0)
   {
      hypre_error_in_arg(2);
      if (print_level)
      {
         hypre_printf("Error! nrows negative! HYPRE_IJMatrixSetValues\n");
      }
      return hypre_error_flag;
   }

   if (hypre_IJMatrixAssembleFlag(matrix))  /* matrix already assembled*/
   {
      HYPRE_BigInt *col_map_offd = NULL;
      HYPRE_Int num_cols_offd;

      diag = hypre_ParCSRMatrixDiag(par_matrix);
      diag_i = hypre_CSRMatrixI(diag);
      diag_j = hypre_CSRMatrixJ(diag);
      diag_data = hypre_CSRMatrixData(diag);
      offd = hypre_ParCSRMatrixOffd(par_matrix);
      offd_i = hypre_CSRMatrixI(offd);
      num_cols_offd = hypre_CSRMatrixNumCols(offd);
      if (num_cols_offd)
      {
         col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
         offd_j = hypre_CSRMatrixJ(offd);
         offd_data = hypre_CSRMatrixData(offd);
      }
      aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);
      /*if (aux_matrix)
      {
         current_num_elmts
           = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
         off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
         cancel_indx = hypre_AuxParCSRMatrixCancelIndx(aux_matrix);
      }*/

#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel
#endif
      {
         HYPRE_Int j_offd;
         HYPRE_Int num_threads, my_thread_num;
         HYPRE_Int len, rest, ns, ne;
         HYPRE_Int pos_diag, pos_offd;
         HYPRE_Int len_diag, len_offd;
         //HYPRE_Int row_len;
         HYPRE_Int row_local;
         HYPRE_Int i, j, ii, n;
         HYPRE_BigInt row;
         HYPRE_Int not_found, size, indx;

         num_threads = hypre_NumActiveThreads();
         my_thread_num = hypre_GetThreadNum();

         len = nrows / num_threads;
         rest = nrows - len * num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num * (len + 1);
            ne = (my_thread_num + 1) * (len + 1);
         }
         else
         {
            ns = my_thread_num * len + rest;
            ne = (my_thread_num + 1) * len + rest;
         }

         for (ii = ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols ? ncols[ii] : 1;
            if (n == 0) /* empty row */
            {
               continue;
            }
            indx = row_indexes[ii];
            /* processor owns the row */
            if (row >= row_partitioning[0] && row < row_partitioning[1])
            {
               row_local = (HYPRE_Int)(row - row_partitioning[0]);

               /* compute local row number */
               size = diag_i[row_local + 1] - diag_i[row_local]
                      + offd_i[row_local + 1] - offd_i[row_local];

               if (n > size)
               {
                  hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                  #pragma omp atomic
#endif
                  error_flag++;
#endif
                  if (print_level)
                  {
                     hypre_printf (" row %b too long! \n", row);
                  }
                  break;
                  /*return hypre_error_flag; */
               }

               pos_diag = diag_i[row_local];
               pos_offd = offd_i[row_local];
               len_diag = diag_i[row_local + 1];
               len_offd = offd_i[row_local + 1];
               not_found = 1;

               for (i = 0; i < n; i++)
               {
                  if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */
                  {
                     j_offd = hypre_BigBinarySearch(col_map_offd, cols[indx] - first,
                                                    num_cols_offd);
                     if (j_offd == -1)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
#endif
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag; */
                     }
                     for (j = pos_offd; j < len_offd; j++)
                     {
                        if (offd_j[j] == j_offd)
                        {
                           offd_data[j] = values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
#endif
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag;*/
                     }
                     not_found = 1;
                  }
                  /* diagonal element */
                  else if (cols[indx] == row)
                  {
                     if (diag_j[pos_diag] != row_local)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
#endif
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag; */
                     }
                     diag_data[pos_diag] = values[indx];
                  }
                  else  /* insert into diag */
                  {
                     for (j = pos_diag; j < len_diag; j++)
                     {
                        if (diag_j[j] == (HYPRE_Int)(cols[indx] - col_0))
                        {
                           diag_data[j] = values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
#endif
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag;*/
                     }
                  }
                  indx++;
               }
            }

            /* processor does not own the row */

            //else /*search for previous occurrences and cancel them */
            /*{
               if (aux_matrix)
               {
                  col_indx = 0;
                  for (i=0; i < off_proc_i_indx; i=i+2)
                  {
                     row_len = off_proc_i[i+1];
                     if (off_proc_i[i] == row)
                     {
                        for (j=0; j < n; j++)
                        {
                           cnt1 = col_indx;
                           for (k=0; k < row_len; k++)
                           {
                              if (off_proc_j[cnt1] == cols[j])
                              {
                                 off_proc_j[cnt1++] = -1;
                                 offproc_cnt[my_thread_num]++; */
            /*cancel_indx++;*/
            /* if no repetition allowed */
            /* off_proc_j[col_indx] = -1;
               col_indx -= k;
               break; */
            /*}
            else
            {
               cnt1++;
            }
            }
            }
            col_indx += row_len;
            }
            else
            {
            col_indx += row_len;
            }
            }*/
            /*hypre_AuxParCSRMatrixCancelIndx(aux_matrix) = cancel_indx;*/
            //}
            //}
         }
      } /*end parallel region */
   }
   else  /* matrix not assembled */
   {
      aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);
      /*if (aux_matrix)
      {
         current_num_elmts
           = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
         off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
         cancel_indx = hypre_AuxParCSRMatrixCancelIndx(aux_matrix);
      }*/
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      if (need_aux)
      {
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
      }
      else
      {
         diag = hypre_ParCSRMatrixDiag(par_matrix);
         diag_i = hypre_CSRMatrixI(diag);
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         offd = hypre_ParCSRMatrixOffd(par_matrix);
         offd_i = hypre_CSRMatrixI(offd);
         if (num_procs > 1)
         {
            offd_data = hypre_CSRMatrixData(offd);
            big_offd_j = hypre_CSRMatrixBigJ(offd);
            if (!big_offd_j)
            {
               big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[hypre_CSRMatrixNumRows(offd)],
                                          hypre_CSRMatrixMemoryLocation(offd));
               hypre_CSRMatrixBigJ(offd) = big_offd_j;
            }
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel
#endif
      {
         HYPRE_Int num_threads, my_thread_num;
         HYPRE_Int len, rest, ns, ne;
         HYPRE_BigInt *tmp_j = NULL;
         HYPRE_BigInt *local_j = NULL;
         HYPRE_Complex *tmp_data = NULL;
         HYPRE_Complex *local_data = NULL;
         HYPRE_Int tmp_indx;
         //HYPRE_Int row_len;
         HYPRE_Int row_local;
         HYPRE_Int i, j, ii, n;
         HYPRE_BigInt row;
         HYPRE_Int not_found, size, indx;
         HYPRE_Int old_size, space, cnt;

         num_threads = hypre_NumActiveThreads();
         my_thread_num = hypre_GetThreadNum();

         len = nrows / num_threads;
         rest = nrows - len * num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num * (len + 1);
            ne = (my_thread_num + 1) * (len + 1);
         }
         else
         {
            ns = my_thread_num * len + rest;
            ne = (my_thread_num + 1) * len + rest;
         }

         for (ii = ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols ? ncols[ii] : 1;
            if (n == 0) /* empty row */
            {
               continue;
            }
            indx = row_indexes[ii];
            /* processor owns the row */
            if (row >= row_partitioning[0] && row < row_partitioning[1])
            {
               row_local = (HYPRE_Int)(row - row_partitioning[0]);
               /* compute local row number */
               if (need_aux)
               {
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
                  space = row_space[row_local];
                  old_size = row_length[row_local];
                  size = space - old_size;
                  if (size < n)
                  {
                     size = n - size;
                     tmp_j = hypre_CTAlloc(HYPRE_BigInt, size, HYPRE_MEMORY_HOST);
                     tmp_data = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
                  }
                  tmp_indx = 0;
                  not_found = 1;
                  size = old_size;
                  for (i = 0; i < n; i++)
                  {
                     for (j = 0; j < old_size; j++)
                     {
                        if (local_j[j] == cols[indx])
                        {
                           local_data[j] = values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        if (size < space)
                        {
                           local_j[size] = cols[indx];
                           local_data[size++] = values[indx];
                        }
                        else
                        {
                           tmp_j[tmp_indx] = cols[indx];
                           tmp_data[tmp_indx++] = values[indx];
                        }
                     }
                     not_found = 1;
                     indx++;
                  }

                  row_length[row_local] = size + tmp_indx;

                  if (tmp_indx)
                  {
                     aux_j[row_local] = hypre_TReAlloc(aux_j[row_local], HYPRE_BigInt,
                                                       size + tmp_indx, HYPRE_MEMORY_HOST);
                     aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
                                                          HYPRE_Complex, size + tmp_indx, HYPRE_MEMORY_HOST);
                     row_space[row_local] = size + tmp_indx;
                     local_j = aux_j[row_local];
                     local_data = aux_data[row_local];
                  }

                  cnt = size;

                  for (i = 0; i < tmp_indx; i++)
                  {
                     local_j[cnt] = tmp_j[i];
                     local_data[cnt++] = tmp_data[i];
                  }

                  if (tmp_j)
                  {
                     hypre_TFree(tmp_j, HYPRE_MEMORY_HOST);
                     hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
                  }
               }
               else /* insert immediately into data in ParCSRMatrix structure */
               {
                  HYPRE_Int offd_indx, diag_indx;
                  HYPRE_Int offd_space, diag_space;
                  HYPRE_Int cnt_diag, cnt_offd;
                  offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
                  diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
                  cnt_diag = diag_indx;
                  cnt_offd = offd_indx;
                  diag_space = diag_i[row_local + 1];
                  offd_space = offd_i[row_local + 1];
                  not_found = 1;
                  for (i = 0; i < n; i++)
                  {
                     if (cols[indx] < col_0 || cols[indx] > col_n)
                        /* insert into offd */
                     {
                        for (j = offd_i[row_local]; j < offd_indx; j++)
                        {
                           if (big_offd_j[j] == cols[indx])
                           {
                              offd_data[j] = values[indx];
                              not_found = 0;
                              break;
                           }
                        }
                        if (not_found)
                        {
                           if (cnt_offd < offd_space)
                           {
                              big_offd_j[cnt_offd] = cols[indx];
                              offd_data[cnt_offd++] = values[indx];
                           }
                           else
                           {
                              hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                              #pragma omp atomic
#endif
                              error_flag++;
#endif
                              if (print_level)
                              {
                                 hypre_printf("Error in row %b ! Too many elements!\n",
                                              row);
                              }
                              break;
                              /*return hypre_error_flag;*/
                           }
                        }
                        not_found = 1;
                     }
                     else  /* insert into diag */
                     {
                        for (j = diag_i[row_local]; j < diag_indx; j++)
                        {
                           if (diag_j[j] == (HYPRE_Int)(cols[indx] - col_0))
                           {
                              diag_data[j] = values[indx];
                              not_found = 0;
                              break;
                           }
                        }
                        if (not_found)
                        {
                           if (cnt_diag < diag_space)
                           {
                              diag_j[cnt_diag] = (HYPRE_Int)(cols[indx] - col_0);
                              diag_data[cnt_diag++] = values[indx];
                           }
                           else
                           {
                              hypre_error(HYPRE_ERROR_GENERIC);
#if 0 /* error_flag is currently not used anywhere */
#ifdef HYPRE_USING_OPENMP
                              #pragma omp atomic
#endif
                              error_flag++;
#endif
                              if (print_level)
                              {
                                 hypre_printf("Error in row %b ! Too many elements !\n",
                                              row);
                              }
                              break;
                              /*return hypre_error_flag;*/
                           }
                        }
                        not_found = 1;
                     }
                     indx++;
                  }

                  hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
                  hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

               }
            }

            /* processor does not own the row */
            /*else
            {
               if (aux_matrix)
               {
                  col_indx = 0;
                  for (i=0; i < off_proc_i_indx; i=i+2)
                  {
                     row_len = off_proc_i[i+1];
                     if (off_proc_i[i] == row)
                     {
                        for (j=0; j < n; j++)
                        {
                           cnt1 = col_indx;
                           for (k=0; k < row_len; k++)
                           {
                              if (off_proc_j[cnt1] == cols[j])
                              {
                                 off_proc_j[cnt1++] = -1; */
            /*cancel_indx++;*/
            //offproc_cnt[my_thread_num]++;
            /* if no repetition allowed */
            /* off_proc_j[col_indx] = -1;
               col_indx -= k;
               break; */
            /*          }
                      else
                      {
                         cnt1++;
                      }
                   }
                }
                col_indx += row_len;
             }
             else
             {
                col_indx += row_len;
             }
            }*/
            /*hypre_AuxParCSRMatrixCancelIndx(aux_matrix) = cancel_indx;*/
            /*}
            }*/
         }
      } /* end parallel region */
   }
   /*if (error_flag)
   {
      return hypre_error_flag;
   }
   if (aux_matrix)
   {
      for (i1=0; i1 < max_num_threads; i1++)
      {
         cancel_indx += offproc_cnt[i1];
      }
      hypre_AuxParCSRMatrixCancelIndx(aux_matrix) = cancel_indx;
   }*/
   //hypre_TFree(offproc_cnt, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre_IJMatrixAddToValuesOMPParCSR
 *
 * adds row values to an IJMatrix
 *
 *****************************************************************************/

HYPRE_Int
hypre_IJMatrixAddToValuesOMPParCSR( hypre_IJMatrix       *matrix,
                                    HYPRE_Int             nrows,
                                    HYPRE_Int            *ncols,
                                    const HYPRE_BigInt   *rows,
                                    const HYPRE_Int      *row_indexes,
                                    const HYPRE_BigInt   *cols,
                                    const HYPRE_Complex  *values )
{
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   HYPRE_BigInt *row_partitioning;
   HYPRE_BigInt *col_partitioning;
   MPI_Comm comm = hypre_IJMatrixComm(matrix);
   HYPRE_Int num_procs, my_id;
   HYPRE_BigInt col_0, col_n, first;
   HYPRE_BigInt **aux_j;
   HYPRE_Complex **aux_data;
   HYPRE_Int *row_length, *row_space;
   HYPRE_Int need_aux;
   HYPRE_Int *diag_i = NULL;
   HYPRE_Int *diag_j = NULL;
   HYPRE_Complex *diag_data = NULL;
   HYPRE_Int *offd_i = NULL;
   HYPRE_Int *offd_j = NULL;
   HYPRE_BigInt *big_offd_j = NULL;
   HYPRE_Complex *offd_data = NULL;
   HYPRE_Int current_num_elmts;
   HYPRE_Int max_off_proc_elmts;
   HYPRE_Int off_proc_i_indx;
   HYPRE_BigInt *off_proc_i;
   HYPRE_BigInt *off_proc_j;
   HYPRE_Complex *off_proc_data;
   HYPRE_Int **offproc_cnt;

   HYPRE_Int print_level = hypre_IJMatrixPrintLevel(matrix);
   HYPRE_Int max_num_threads;
   HYPRE_Int error_flag = 0;
   HYPRE_Int i1;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   max_num_threads = hypre_NumThreads();
   par_matrix = (hypre_ParCSRMatrix*) hypre_IJMatrixObject( matrix );
   row_partitioning = hypre_IJMatrixRowPartitioning(matrix);
   col_partitioning = hypre_IJMatrixColPartitioning(matrix);

   offproc_cnt = hypre_CTAlloc(HYPRE_Int *,  max_num_threads, HYPRE_MEMORY_HOST);

   for (i1 = 0; i1 < max_num_threads; i1++)
   {
      offproc_cnt[i1] = NULL;
   }

   col_0 = col_partitioning[0];
   col_n = col_partitioning[1] - 1;
   first = hypre_IJMatrixGlobalFirstCol(matrix);
   if (hypre_IJMatrixAssembleFlag(matrix)) /* matrix already assembled */
   {
      HYPRE_Int num_cols_offd;
      HYPRE_BigInt *col_map_offd = NULL;

      diag = hypre_ParCSRMatrixDiag(par_matrix);
      diag_i = hypre_CSRMatrixI(diag);
      diag_j = hypre_CSRMatrixJ(diag);
      diag_data = hypre_CSRMatrixData(diag);
      offd = hypre_ParCSRMatrixOffd(par_matrix);
      offd_i = hypre_CSRMatrixI(offd);
      num_cols_offd = hypre_CSRMatrixNumCols(offd);
      if (num_cols_offd)
      {
         col_map_offd = hypre_ParCSRMatrixColMapOffd(par_matrix);
         offd_j = hypre_CSRMatrixJ(offd);
         offd_data = hypre_CSRMatrixData(offd);
      }
      aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);
      if (aux_matrix)
      {
         current_num_elmts
            = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
         off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel
#endif
      {
         HYPRE_Int j_offd;
         HYPRE_Int num_threads, my_thread_num;
         HYPRE_Int len, rest, ns, ne;
         HYPRE_Int pos_diag, pos_offd;
         HYPRE_Int len_diag, len_offd;
         HYPRE_Int row_local;
         HYPRE_Int i, j, ii, n;
         HYPRE_BigInt row;
         HYPRE_Int not_found, size, indx;
         HYPRE_Int *my_offproc_cnt = NULL;

         num_threads = hypre_NumActiveThreads();
         my_thread_num = hypre_GetThreadNum();

         len = nrows / num_threads;
         rest = nrows - len * num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num * (len + 1);
            ne = (my_thread_num + 1) * (len + 1);
         }
         else
         {
            ns = my_thread_num * len + rest;
            ne = (my_thread_num + 1) * len + rest;
         }

         for (ii = ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols ? ncols[ii] : 1;
            if (n == 0) /* empty row */
            {
               continue;
            }
            indx = row_indexes[ii];
            if (row >= row_partitioning[0] && row < row_partitioning[1])
            {
               row_local = (HYPRE_Int)(row - row_partitioning[0]);
               /* compute local row number */
               size = diag_i[row_local + 1] - diag_i[row_local]
                      + offd_i[row_local + 1] - offd_i[row_local];

               if (n > size)
               {
                  hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                  #pragma omp atomic
#endif
                  error_flag++;
                  if (print_level)
                  {
                     hypre_printf (" row %b too long! \n", row);
                  }
                  break;
                  /*return hypre_error_flag; */
               }

               pos_diag = diag_i[row_local];
               pos_offd = offd_i[row_local];
               len_diag = diag_i[row_local + 1];
               len_offd = offd_i[row_local + 1];
               not_found = 1;

               for (i = 0; i < n; i++)
               {
                  if (cols[indx] < col_0 || cols[indx] > col_n)
                     /* insert into offd */
                  {
                     j_offd = hypre_BigBinarySearch(col_map_offd, cols[indx] - first,
                                                    num_cols_offd);
                     if (j_offd == -1)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag;*/
                     }
                     for (j = pos_offd; j < len_offd; j++)
                     {
                        if (offd_j[j] == j_offd)
                        {
                           offd_data[j] += values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag;*/
                     }
                     not_found = 1;
                  }
                  /* diagonal element */
                  else if (cols[indx] == row)
                  {
                     if (diag_j[pos_diag] != row_local)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag;*/
                     }
                     diag_data[pos_diag] += values[indx];
                  }
                  else  /* insert into diag */
                  {
                     for (j = pos_diag; j < len_diag; j++)
                     {
                        if (diag_j[j] == (HYPRE_Int)(cols[indx] - col_0))
                        {
                           diag_data[j] += values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                        #pragma omp atomic
#endif
                        error_flag++;
                        if (print_level)
                        {
                           hypre_printf (" Error, element %b %b does not exist\n",
                                         row, cols[indx]);
                        }
                        break;
                        /*return hypre_error_flag;*/
                     }
                  }
                  indx++;
               }
            }
            /* not my row */
            /* need to find solution for threaded version!!!! */
            /* could save row number and process later .... */
            else
            {
               if (!my_offproc_cnt)
               {
                  my_offproc_cnt = hypre_CTAlloc(HYPRE_Int,  200, HYPRE_MEMORY_HOST);
                  offproc_cnt[my_thread_num] = my_offproc_cnt;
                  my_offproc_cnt[0] = 200;
                  my_offproc_cnt[1] = 2;
               }
               i = my_offproc_cnt[1];
               if (i + 2 < my_offproc_cnt[0])
               {
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i + 1] = indx;
                  my_offproc_cnt[1] += 2;
               }
               else
               {
                  size = my_offproc_cnt[0];
                  my_offproc_cnt = hypre_TReAlloc(my_offproc_cnt, HYPRE_Int, size + 200, HYPRE_MEMORY_HOST);
                  my_offproc_cnt[0] += 200;
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i + 1] = indx;
                  my_offproc_cnt[1] += 2;
               }
            }
         }
      } /* end parallel region */
   }

   /* not assembled */
   else
   {
      aux_matrix = (hypre_AuxParCSRMatrix*) hypre_IJMatrixTranslator(matrix);
      if (aux_matrix)
      {
         current_num_elmts
            = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
         off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
         off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
         off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
      }
      row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
      row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
      need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
      if (need_aux)
      {
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
      }
      else
      {
         diag = hypre_ParCSRMatrixDiag(par_matrix);
         diag_i = hypre_CSRMatrixI(diag);
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         offd = hypre_ParCSRMatrixOffd(par_matrix);
         offd_i = hypre_CSRMatrixI(offd);
         if (num_procs > 1)
         {
            big_offd_j = hypre_CSRMatrixBigJ(offd);
            offd_data = hypre_CSRMatrixData(offd);
            if (!big_offd_j)
            {
               big_offd_j = hypre_CTAlloc(HYPRE_BigInt, offd_i[hypre_CSRMatrixNumRows(offd)],
                                          hypre_CSRMatrixMemoryLocation(offd));
               hypre_CSRMatrixBigJ(offd) = big_offd_j;
            }
         }
      }
#ifdef HYPRE_USING_OPENMP
      #pragma omp parallel
#endif
      {
         HYPRE_Int num_threads, my_thread_num;
         HYPRE_Int len, rest, ns, ne;
         HYPRE_BigInt *tmp_j = NULL;
         HYPRE_BigInt *local_j = NULL;
         HYPRE_Complex *tmp_data = NULL;
         HYPRE_Complex *local_data = NULL;
         HYPRE_Int tmp_indx;
         HYPRE_Int row_local;
         HYPRE_BigInt row;
         HYPRE_Int i, j, ii, n;
         HYPRE_Int not_found, size, indx;
         HYPRE_Int old_size, space, cnt;
         HYPRE_Int *my_offproc_cnt = NULL;

         num_threads = hypre_NumActiveThreads();
         my_thread_num = hypre_GetThreadNum();

         len = nrows / num_threads;
         rest = nrows - len * num_threads;

         if (my_thread_num < rest)
         {
            ns = my_thread_num * (len + 1);
            ne = (my_thread_num + 1) * (len + 1);
         }
         else
         {
            ns = my_thread_num * len + rest;
            ne = (my_thread_num + 1) * len + rest;
         }

         for (ii = ns; ii < ne; ii++)
         {
            row = rows[ii];
            n = ncols ? ncols[ii] : 1;
            if (n == 0) /* empty row */
            {
               continue;
            }
            indx = row_indexes[ii];
            if (row >= row_partitioning[0] && row < row_partitioning[1])
            {
               row_local = (HYPRE_Int)(row - row_partitioning[0]);
               /* compute local row number */
               if (need_aux)
               {
                  local_j = aux_j[row_local];
                  local_data = aux_data[row_local];
                  space = row_space[row_local];
                  old_size = row_length[row_local];
                  size = space - old_size;
                  if (size < n)
                  {
                     size = n - size;
                     tmp_j = hypre_CTAlloc(HYPRE_BigInt, size, HYPRE_MEMORY_HOST);
                     tmp_data = hypre_CTAlloc(HYPRE_Complex, size, HYPRE_MEMORY_HOST);
                  }
                  tmp_indx = 0;
                  not_found = 1;
                  size = old_size;
                  for (i = 0; i < n; i++)
                  {
                     for (j = 0; j < old_size; j++)
                     {
                        if (local_j[j] == cols[indx])
                        {
                           local_data[j] += values[indx];
                           not_found = 0;
                           break;
                        }
                     }
                     if (not_found)
                     {
                        if (size < space)
                        {
                           local_j[size] = cols[indx];
                           local_data[size++] = values[indx];
                        }
                        else
                        {
                           tmp_j[tmp_indx] = cols[indx];
                           tmp_data[tmp_indx++] = values[indx];
                        }
                     }
                     not_found = 1;
                     indx++;
                  }

                  row_length[row_local] = size + tmp_indx;

                  if (tmp_indx)
                  {
                     aux_j[row_local] = hypre_TReAlloc(aux_j[row_local], HYPRE_BigInt,
                                                       size + tmp_indx, HYPRE_MEMORY_HOST);
                     aux_data[row_local] = hypre_TReAlloc(aux_data[row_local],
                                                          HYPRE_Complex, size + tmp_indx, HYPRE_MEMORY_HOST);
                     row_space[row_local] = size + tmp_indx;
                     local_j = aux_j[row_local];
                     local_data = aux_data[row_local];
                  }

                  cnt = size;

                  for (i = 0; i < tmp_indx; i++)
                  {
                     local_j[cnt] = tmp_j[i];
                     local_data[cnt++] = tmp_data[i];
                  }

                  if (tmp_j)
                  {
                     hypre_TFree(tmp_j, HYPRE_MEMORY_HOST);
                     hypre_TFree(tmp_data, HYPRE_MEMORY_HOST);
                  }
               }
               else /* insert immediately into data in ParCSRMatrix structure */
               {
                  HYPRE_Int offd_indx, diag_indx;
                  HYPRE_Int offd_space, diag_space;
                  HYPRE_Int cnt_diag, cnt_offd;
                  offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
                  diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
                  cnt_diag = diag_indx;
                  cnt_offd = offd_indx;
                  diag_space = diag_i[row_local + 1];
                  offd_space = offd_i[row_local + 1];
                  not_found = 1;
                  for (i = 0; i < n; i++)
                  {
                     if (cols[indx] < col_0 || cols[indx] > col_n)
                        /* insert into offd */
                     {
                        for (j = offd_i[row_local]; j < offd_indx; j++)
                        {
                           if (big_offd_j[j] == cols[indx])
                           {
                              offd_data[j] += values[indx];
                              not_found = 0;
                              break;
                           }
                        }
                        if (not_found)
                        {
                           if (cnt_offd < offd_space)
                           {
                              big_offd_j[cnt_offd] = cols[indx];
                              offd_data[cnt_offd++] = values[indx];
                           }
                           else
                           {
                              hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                              #pragma omp atomic
#endif
                              error_flag++;
                              if (print_level)
                              {
                                 hypre_printf("Error in row %b ! Too many elements!\n",
                                              row);
                              }
                              break;
                              /*return hypre_error_flag;*/
                           }
                        }
                        not_found = 1;
                     }
                     else  /* insert into diag */
                     {
                        for (j = diag_i[row_local]; j < diag_indx; j++)
                        {
                           if (diag_j[j] == (HYPRE_Int)(cols[indx] - col_0))
                           {
                              diag_data[j] += values[indx];
                              not_found = 0;
                              break;
                           }
                        }
                        if (not_found)
                        {
                           if (cnt_diag < diag_space)
                           {
                              diag_j[cnt_diag] = (HYPRE_Int)(cols[indx] - col_0);
                              diag_data[cnt_diag++] = values[indx];
                           }
                           else
                           {
                              hypre_error(HYPRE_ERROR_GENERIC);
#ifdef HYPRE_USING_OPENMP
                              #pragma omp atomic
#endif
                              error_flag++;
                              if (print_level)
                              {
                                 hypre_printf("Error in row %b ! Too many elements !\n",
                                              row);
                              }
                              break;
                              /*return hypre_error_flag;*/
                           }
                        }
                        not_found = 1;
                     }
                     indx++;
                  }

                  hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = cnt_diag;
                  hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt_offd;

               }
            }
            /* not my row */
            else
            {
               if (!my_offproc_cnt)
               {
                  my_offproc_cnt = hypre_CTAlloc(HYPRE_Int,  200, HYPRE_MEMORY_HOST);
                  offproc_cnt[my_thread_num] = my_offproc_cnt;
                  my_offproc_cnt[0] = 200;
                  my_offproc_cnt[1] = 2;
               }
               i = my_offproc_cnt[1];
               if (i + 2 < my_offproc_cnt[0])
               {
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i + 1] = indx;
                  my_offproc_cnt[1] += 2;
               }
               else
               {
                  size = my_offproc_cnt[0];
                  my_offproc_cnt = hypre_TReAlloc(my_offproc_cnt, HYPRE_Int,
                                                  size + 200, HYPRE_MEMORY_HOST);
                  offproc_cnt[my_thread_num] = my_offproc_cnt;
                  my_offproc_cnt[0] += 200;
                  my_offproc_cnt[i] = ii;
                  my_offproc_cnt[i + 1] = indx;
                  my_offproc_cnt[1] += 2;
               }
            }
         }
      } /*end parallel region */
   }
   if (error_flag)
   {
      return hypre_error_flag;
   }
   if (!aux_matrix)
   {
      HYPRE_Int size = (HYPRE_Int)(row_partitioning[1] - row_partitioning[0]);
      hypre_AuxParCSRMatrixCreate(&aux_matrix, size, size, NULL);
      hypre_AuxParCSRMatrixNeedAux(aux_matrix) = 0;
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }
   for (i1 = 0; i1 < max_num_threads; i1++)
   {
      if (offproc_cnt[i1])
      {
         HYPRE_Int *my_offproc_cnt = offproc_cnt[i1];
         HYPRE_Int i, i2, ii, n, indx;
         HYPRE_BigInt row;
         for (i2 = 2; i2 < my_offproc_cnt[1]; i2 += 2)
         {
            ii = my_offproc_cnt[i2];
            row = rows[ii];
            n = ncols ? ncols[ii] : 1;
            if (n == 0) /* empty row */
            {
               continue;
            }
            indx = my_offproc_cnt[i2 + 1];
            current_num_elmts
               = hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix);
            max_off_proc_elmts
               = hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix);
            off_proc_i_indx = hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix);
            off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
            off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
            off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);

            if (!max_off_proc_elmts)
            {
               max_off_proc_elmts = hypre_max(n, 1000);
               hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix) =
                  max_off_proc_elmts;
               hypre_AuxParCSRMatrixOffProcI(aux_matrix)
                  = hypre_CTAlloc(HYPRE_BigInt, 2 * max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixOffProcJ(aux_matrix)
                  = hypre_CTAlloc(HYPRE_BigInt, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixOffProcData(aux_matrix)
                  = hypre_CTAlloc(HYPRE_Complex, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_i = hypre_AuxParCSRMatrixOffProcI(aux_matrix);
               off_proc_j = hypre_AuxParCSRMatrixOffProcJ(aux_matrix);
               off_proc_data = hypre_AuxParCSRMatrixOffProcData(aux_matrix);
            }
            else if (current_num_elmts + n > max_off_proc_elmts)
            {
               max_off_proc_elmts += 3 * n;
               off_proc_i = hypre_TReAlloc(off_proc_i, HYPRE_BigInt, 2 * max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_j = hypre_TReAlloc(off_proc_j, HYPRE_BigInt, max_off_proc_elmts, HYPRE_MEMORY_HOST);
               off_proc_data = hypre_TReAlloc(off_proc_data, HYPRE_Complex,
                                              max_off_proc_elmts, HYPRE_MEMORY_HOST);
               hypre_AuxParCSRMatrixMaxOffProcElmts(aux_matrix)
                  = max_off_proc_elmts;
               hypre_AuxParCSRMatrixOffProcI(aux_matrix) = off_proc_i;
               hypre_AuxParCSRMatrixOffProcJ(aux_matrix) = off_proc_j;
               hypre_AuxParCSRMatrixOffProcData(aux_matrix) = off_proc_data;
            }
            off_proc_i[off_proc_i_indx++] = row;
            off_proc_i[off_proc_i_indx++] = n;
            for (i = 0; i < n; i++)
            {
               off_proc_j[current_num_elmts] = cols[indx];
               off_proc_data[current_num_elmts++] = values[indx++];
            }
            hypre_AuxParCSRMatrixOffProcIIndx(aux_matrix) = off_proc_i_indx;
            hypre_AuxParCSRMatrixCurrentOffProcElmts(aux_matrix) = current_num_elmts;
         }
         hypre_TFree(offproc_cnt[i1], HYPRE_MEMORY_HOST);
      }
   }
   hypre_TFree(offproc_cnt, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
