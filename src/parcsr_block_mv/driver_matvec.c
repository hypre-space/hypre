/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "../utilities/_hypre_utilities.h"
#include "../seq_mv/seq_mv.h"
#include "../parcsr_mv/_hypre_parcsr_mv.h"
#include "../parcsr_ls/_hypre_parcsr_ls.h"
#include "../krylov/krylov.h"
#include "par_csr_block_matrix.h"

extern HYPRE_Int MyBuildParLaplacian9pt(HYPRE_ParCSRMatrix  *A_ptr);

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface
 *--------------------------------------------------------------------------*/

HYPRE_Int main( HYPRE_Int   argc, char *argv[] )
{
   hypre_ParCSRMatrix      *par_matrix, *g_matrix, **submatrices;
   hypre_CSRMatrix         *A_diag, *A_offd;
   hypre_CSRBlockMatrix    *diag;
   hypre_CSRBlockMatrix    *offd;
   hypre_ParCSRBlockMatrix *par_blk_matrix, *par_blk_matrixT, *rap_matrix;
   hypre_Vector        *x_local;
   hypre_Vector        *y_local;
   hypre_ParVector     *x;
   hypre_ParVector     *y;
   HYPRE_Solver        gmres_solver, precon;
   HYPRE_Int                 *diag_i, *diag_j, *offd_i, *offd_j;
   HYPRE_Int                 *diag_i2, *diag_j2, *offd_i2, *offd_j2;
   HYPRE_Complex       *diag_d, *diag_d2, *offd_d, *offd_d2;
   HYPRE_Int                   mypid, local_size, nprocs;
   HYPRE_Int                   global_num_rows, global_num_cols, num_cols_offd;
   HYPRE_Int                   num_nonzeros_diag, num_nonzeros_offd, *colMap;
   HYPRE_Int                   ii, jj, kk, row, col, nnz, *indices, *colMap2;
   HYPRE_Complex               *data, ddata, *y_data;
   HYPRE_Int                   *row_starts, *col_starts, *rstarts, *cstarts;
   HYPRE_Int                   *row_starts2, *col_starts2;
   HYPRE_Int                 block_size = 2, bnnz = 4, *index_set;
   FILE                *fp;

   /* --------------------------------------------- */
   /* Initialize MPI                                */
   /* --------------------------------------------- */

   hypre_MPI_Init(&argc, &argv);
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &mypid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nprocs);

   /* build and fetch matrix */
   MyBuildParLaplacian9pt((HYPRE_ParCSRMatrix *) &par_matrix);
   global_num_rows = hypre_ParCSRMatrixGlobalNumRows(par_matrix);
   global_num_cols = hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   A_diag = hypre_ParCSRMatrixDiag(par_matrix);
   A_offd = hypre_ParCSRMatrixOffd(par_matrix);
   num_cols_offd     = hypre_CSRMatrixNumCols(A_offd);
   num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(A_diag);
   num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(A_offd);

   /* --------------------------------------------- */
   /* build vector and apply matvec                 */
   /* --------------------------------------------- */

   x = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_cols, col_starts);
   hypre_ParVectorInitialize(x);
   x_local = hypre_ParVectorLocalVector(x);
   data    = hypre_VectorData(x_local);
   local_size = col_starts[mypid + 1] - col_starts[mypid];
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   y = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
   hypre_ParVectorInitialize(y);
   hypre_ParCSRMatrixMatvec (1.0, par_matrix, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { hypre_printf("y inner product = %e\n", ddata); }
   hypre_ParVectorDestroy(x);
   hypre_ParVectorDestroy(y);

   /* --------------------------------------------- */
   /* build block matrix                            */
   /* --------------------------------------------- */

   rstarts = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { rstarts[ii] = row_starts[ii]; }
   cstarts = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { cstarts[ii] = col_starts[ii]; }

   par_blk_matrix = hypre_ParCSRBlockMatrixCreate(hypre_MPI_COMM_WORLD, block_size,
                                                  global_num_rows, global_num_cols, rstarts,
                                                  cstarts, num_cols_offd, num_nonzeros_diag,
                                                  num_nonzeros_offd);
   colMap  = hypre_ParCSRMatrixColMapOffd(par_matrix);
   if (num_cols_offd > 0) { colMap2 = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST); }
   else { colMap2 = NULL; }
   for (ii = 0; ii < num_cols_offd; ii++) { colMap2[ii] = colMap[ii]; }
   hypre_ParCSRBlockMatrixColMapOffd(par_blk_matrix) = colMap2;
   diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
   diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(par_matrix));
   diag_d = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(par_matrix));
   diag = hypre_ParCSRBlockMatrixDiag(par_blk_matrix);
   diag_i2 = hypre_CTAlloc(HYPRE_Int,  local_size + 1, HYPRE_MEMORY_HOST);
   diag_j2 = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_HOST);
   diag_d2 = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros_diag * bnnz, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { diag_i2[ii] = diag_i[ii]; }
   for (ii = 0; ii < num_nonzeros_diag; ii++) { diag_j2[ii] = diag_j[ii]; }
   hypre_CSRBlockMatrixI(diag) = diag_i2;
   hypre_CSRBlockMatrixJ(diag) = diag_j2;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      for (jj = 0; jj < block_size; jj++)
         for (kk = 0; kk < block_size; kk++)
         {
            if (jj <= kk)
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = diag_d[ii];
            }
            else
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = 0.0;
            }
         }
   }
   hypre_CSRBlockMatrixData(diag) = diag_d2;

   offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
   offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix));
   offd_d = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(par_matrix));
   offd   = hypre_ParCSRBlockMatrixOffd(par_blk_matrix);
   offd_i2 = hypre_CTAlloc(HYPRE_Int,  local_size + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { offd_i2[ii] = offd_i[ii]; }
   hypre_CSRBlockMatrixI(offd) = offd_i2;
   if (num_cols_offd)
   {
      offd_j2 = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++) { offd_j2[ii] = offd_j[ii]; }
      hypre_CSRBlockMatrixJ(offd) = offd_j2;
      offd_d2 = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros_offd * bnnz, HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++)
      {
         for (jj = 0; jj < block_size; jj++)
            for (kk = 0; kk < block_size; kk++)
            {
               if (jj <= kk)
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = offd_d[ii];
               }
               else
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = 0.0;
               }
            }
      }
      hypre_CSRBlockMatrixData(offd) = offd_d2;
   }
   else
   {
      hypre_CSRBlockMatrixJ(offd) = NULL;
      hypre_CSRBlockMatrixData(offd) = NULL;
   }

   /* --------------------------------------------- */
   /* build block matrix transpose                  */
   /* --------------------------------------------- */

   rstarts = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { rstarts[ii] = row_starts[ii]; }
   cstarts = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { cstarts[ii] = col_starts[ii]; }

   par_blk_matrixT = hypre_ParCSRBlockMatrixCreate(hypre_MPI_COMM_WORLD, block_size,
                                                   global_num_rows, global_num_cols, rstarts,
                                                   cstarts, num_cols_offd, num_nonzeros_diag,
                                                   num_nonzeros_offd);
   colMap  = hypre_ParCSRMatrixColMapOffd(par_matrix);
   colMap2 = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
   for (ii = 0; ii < num_cols_offd; ii++) { colMap2[ii] = colMap[ii]; }
   hypre_ParCSRBlockMatrixColMapOffd(par_blk_matrixT) = colMap2;
   diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
   diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(par_matrix));
   diag_d = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(par_matrix));
   diag = hypre_ParCSRBlockMatrixDiag(par_blk_matrixT);
   diag_i2 = hypre_CTAlloc(HYPRE_Int,  local_size + 1, HYPRE_MEMORY_HOST);
   diag_j2 = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_diag, HYPRE_MEMORY_HOST);
   diag_d2 = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros_diag * bnnz, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { diag_i2[ii] = diag_i[ii]; }
   for (ii = 0; ii < num_nonzeros_diag; ii++) { diag_j2[ii] = diag_j[ii]; }
   hypre_CSRBlockMatrixI(diag) = diag_i2;
   hypre_CSRBlockMatrixJ(diag) = diag_j2;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      for (jj = 0; jj < block_size; jj++)
         for (kk = 0; kk < block_size; kk++)
         {
            if (jj >= kk)
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = diag_d[ii];
            }
            else
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = 0.0;
            }
         }
   }
   hypre_CSRBlockMatrixData(diag) = diag_d2;

   offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
   offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix));
   offd_d = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(par_matrix));
   offd   = hypre_ParCSRBlockMatrixOffd(par_blk_matrixT);
   offd_i2 = hypre_CTAlloc(HYPRE_Int,  local_size + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { offd_i2[ii] = offd_i[ii]; }
   hypre_CSRBlockMatrixI(offd) = offd_i2;
   if (num_cols_offd)
   {
      offd_j2 = hypre_CTAlloc(HYPRE_Int,  num_nonzeros_offd, HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++) { offd_j2[ii] = offd_j[ii]; }
      hypre_CSRBlockMatrixJ(offd) = offd_j2;
      offd_d2 = hypre_CTAlloc(HYPRE_Complex,  num_nonzeros_offd * bnnz, HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++)
      {
         for (jj = 0; jj < block_size; jj++)
            for (kk = 0; kk < block_size; kk++)
            {
               if (jj >= kk)
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = offd_d[ii];
               }
               else
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = 0.0;
               }
            }
      }
      hypre_CSRBlockMatrixData(offd) = offd_d2;
   }
   else
   {
      hypre_CSRBlockMatrixJ(offd) = NULL;
      hypre_CSRBlockMatrixData(offd) = NULL;
   }

   /* --------------------------------------------- */
   /* block matvec                                  */
   /* --------------------------------------------- */

   col_starts2 = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++)
   {
      col_starts2[ii] = col_starts[ii] * block_size;
   }
   x = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_cols * block_size,
                             col_starts2);
   hypre_ParVectorInitialize(x);
   x_local = hypre_ParVectorLocalVector(x);
   data = hypre_VectorData(x_local);
   local_size = col_starts2[mypid + 1] - col_starts2[mypid];
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   row_starts2 = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++)
   {
      row_starts2[ii] = row_starts[ii] * block_size;
   }
   y = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_rows * block_size,
                             row_starts2);
   hypre_ParVectorInitialize(y);
   y_local = hypre_ParVectorLocalVector(y);
   y_data  = hypre_VectorData(y_local);

   hypre_BlockMatvecCommPkgCreate(par_blk_matrix);
   ddata = hypre_ParVectorInnerProd(x, x);
   if (mypid == 0) { hypre_printf("block x inner product = %e\n", ddata); }
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { hypre_printf("block y inner product = %e\n", ddata); }

   /* --------------------------------------------- */
   /* RAP                                           */
   /* --------------------------------------------- */

   hypre_printf("Verifying RAP\n");
   hypre_ParCSRBlockMatrixRAP(par_blk_matrix, par_blk_matrix,
                              par_blk_matrix, &rap_matrix);
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, x, 0.0, y);
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, y, 0.0, x);
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrixT, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { hypre_printf("(1) A^2 block inner product = %e\n", ddata); }
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   hypre_ParCSRBlockMatrixMatvec (1.0, rap_matrix, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { hypre_printf("(2) A^2 block inner product = %e\n", ddata); }
   if (mypid == 0) { hypre_printf("(1) and (2) should be equal.\n"); }

#if 0
   /* --------------------------------------------- */
   /* diagnostics: print out the matrix             */
   /* --------------------------------------------- */

   diag_i = hypre_CSRBlockMatrixI(A_diag);
   diag_j = hypre_CSRBlockMatrixJ(A_diag);
   diag_d = hypre_CSRBlockMatrixData(A_diag);
   for (ii = 0; ii < hypre_ParCSRMatrixNumRows(par_matrix); ii++)
      for (jj = diag_i[ii]; jj < diag_i[ii + 1]; jj++)
      {
         hypre_printf("A %4d %4d = %e\n", ii, diag_j[jj], diag_d[jj]);
      }

   diag = hypre_ParCSRBlockMatrixDiag(rap_matrix);
   diag_i = hypre_CSRBlockMatrixI(diag);
   diag_j = hypre_CSRBlockMatrixJ(diag);
   diag_d = hypre_CSRBlockMatrixData(diag);
   hypre_printf("RAP block size = %d\n", hypre_ParCSRBlockMatrixBlockSize(rap_matrix));
   hypre_printf("RAP num rows   = %d\n", hypre_ParCSRBlockMatrixNumRows(rap_matrix));
   for (ii = 0; ii < hypre_ParCSRBlockMatrixNumRows(rap_matrix); ii++)
      for (row = 0; row < block_size; row++)
         for (jj = diag_i[ii]; jj < diag_i[ii + 1]; jj++)
            for (col = 0; col < block_size; col++)
               hypre_printf("RAP %4d %4d = %e\n", ii * block_size + row,
                            diag_j[jj]*block_size + col, diag_d[(jj + row)*block_size + col]);
   offd = hypre_ParCSRBlockMatrixOffd(rap_matrix);
   offd_i = hypre_CSRBlockMatrixI(offd);
   offd_j = hypre_CSRBlockMatrixJ(offd);
   offd_d = hypre_CSRBlockMatrixData(offd);
   if (num_cols_offd)
   {
      for (ii = 0; ii < hypre_ParCSRBlockMatrixNumRows(rap_matrix); ii++)
         for (row = 0; row < block_size; row++)
            for (jj = offd_i[ii]; jj < offd_i[ii + 1]; jj++)
               for (col = 0; col < block_size; col++)
                  hypre_printf("RAPOFFD %4d %4d = %e\n", ii * block_size + row,
                               offd_j[jj]*block_size + col, offd_d[(jj + row)*block_size + col]);
   }
#endif
   hypre_ParVectorDestroy(x);
   hypre_ParVectorDestroy(y);
   hypre_ParCSRMatrixDestroy(par_matrix);
   hypre_ParCSRBlockMatrixDestroy(par_blk_matrix);
   hypre_ParCSRBlockMatrixDestroy(par_blk_matrixT);
   hypre_ParCSRBlockMatrixDestroy(rap_matrix);

#if 0
   /* --------------------------------------------- */
   /* read in A_ee and create a HYPRE_ParCSRMatrix  */
   /* --------------------------------------------- */

   if (nprocs == 1)
   {
      fp = fopen("Amat_ee", "r");
      hypre_fscanf(fp, "%d %d", &global_num_rows, &num_nonzeros_diag);
      diag_i = hypre_TAlloc(HYPRE_Int, (global_num_rows + 1), HYPRE_MEMORY_HOST);
      diag_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_diag, HYPRE_MEMORY_HOST);
      diag_d = hypre_TAlloc(HYPRE_Complex, num_nonzeros_diag, HYPRE_MEMORY_HOST);
      row = 0;
      nnz = 0;
      diag_i[0] = 0;
      for (ii = 0; ii < num_nonzeros_diag; ii++)
      {
         hypre_fscanf(fp, "%d %d %lg", &jj, &col, &ddata);
         if ((jj - 1) != row)
         {
            row++;
            diag_i[row] = nnz;
         }
         diag_j[nnz] = col - 1;
         diag_d[nnz++] = ddata;
      }
      diag_i[global_num_rows] = nnz;
      fclose(fp);
      hypre_printf("nrows = %d, nnz = %d\n", row + 1, nnz);

      row_starts = hypre_TAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
      col_starts = hypre_TAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
      row_starts[0] = col_starts[0] = 0;
      row_starts[1] = col_starts[1] = global_num_rows;
      num_cols_offd = 0;
      num_nonzeros_offd = 0;
      par_matrix = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_WORLD, global_num_rows,
                                            global_num_rows, row_starts, col_starts, num_cols_offd,
                                            num_nonzeros_diag, num_nonzeros_offd);
      A_diag = hypre_ParCSRMatrixDiag(par_matrix);
      hypre_CSRMatrixI(A_diag) = diag_i;
      hypre_CSRMatrixJ(A_diag) = diag_j;
      hypre_CSRMatrixData(A_diag) = diag_d;

      /* --------------------------------------------- */
      /* read in discrete gradient matrix              */
      /* --------------------------------------------- */

      fp = fopen("Gmat", "r");
      hypre_fscanf(fp, "%d %d %d", &global_num_rows, &global_num_cols,
                   &num_nonzeros_diag);
      diag_i = hypre_TAlloc(HYPRE_Int, (global_num_rows + 1), HYPRE_MEMORY_HOST);
      diag_j = hypre_TAlloc(HYPRE_Int, num_nonzeros_diag, HYPRE_MEMORY_HOST);
      diag_d = hypre_TAlloc(HYPRE_Complex, num_nonzeros_diag, HYPRE_MEMORY_HOST);
      row = 0;
      nnz = 0;
      diag_i[0] = 0;
      for (ii = 0; ii < num_nonzeros_diag; ii++)
      {
         hypre_fscanf(fp, "%d %d %lg", &jj, &col, &ddata);
         if ((jj - 1) != row)
         {
            row++;
            diag_i[row] = nnz;
         }
         diag_j[nnz] = col - 1;
         diag_d[nnz++] = ddata;
      }
      diag_i[global_num_rows] = nnz;
      fclose(fp);

      row_starts = hypre_TAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
      col_starts = hypre_TAlloc(HYPRE_Int, 2, HYPRE_MEMORY_HOST);
      row_starts[0] = col_starts[0] = 0;
      row_starts[1] = global_num_rows;
      col_starts[1] = global_num_cols;
      num_cols_offd = 0;
      num_nonzeros_offd = 0;
      g_matrix = hypre_ParCSRMatrixCreate(hypre_MPI_COMM_WORLD, global_num_rows,
                                          global_num_cols, row_starts, col_starts, num_cols_offd,
                                          num_nonzeros_diag, num_nonzeros_offd);
      A_diag = hypre_ParCSRMatrixDiag(g_matrix);
      hypre_CSRMatrixI(A_diag) = diag_i;
      hypre_CSRMatrixJ(A_diag) = diag_j;
      hypre_CSRMatrixData(A_diag) = diag_d;

      /* --------------------------------------------- */
      /* Check spanning tree and matrix extraction     */
      /* --------------------------------------------- */

      hypre_ParCSRMatrixGenSpanningTree(g_matrix, &indices, 0);
      submatrices = (hypre_ParCSRMatrix **)
                    hypre_TAlloc(hypre_ParCSRMatrix*, 4, HYPRE_MEMORY_HOST);
      hypre_ParCSRMatrixExtractSubmatrices(par_matrix, indices, &submatrices);
   }
#endif

   /* test block tridiagonal solver */

   if (nprocs == 1)
   {
      MyBuildParLaplacian9pt((HYPRE_ParCSRMatrix *) &par_matrix);
      row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
      col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
      HYPRE_ParCSRGMRESCreate(hypre_MPI_COMM_WORLD, &gmres_solver);
      HYPRE_GMRESSetKDim(gmres_solver, 10);
      HYPRE_GMRESSetMaxIter(gmres_solver, 1000);
      HYPRE_GMRESSetTol(gmres_solver, 1.0e-6);
      HYPRE_GMRESSetLogging(gmres_solver, 1);
      HYPRE_GMRESSetPrintLevel(gmres_solver, 2);
      HYPRE_BlockTridiagCreate(&precon);
      HYPRE_BlockTridiagSetPrintLevel(precon, 0);
      HYPRE_BlockTridiagSetAMGNumSweeps(precon, 1);
      local_size = col_starts[mypid + 1] - col_starts[mypid];
      index_set = hypre_CTAlloc(HYPRE_Int,  local_size + 1, HYPRE_MEMORY_HOST);
      jj = 0;
      /* for (ii = 0; ii < local_size/2; ii++) index_set[jj++] = ii * 2; */
      for (ii = 0; ii < local_size / 2; ii++) { index_set[jj++] = ii; }
      HYPRE_BlockTridiagSetIndexSet(precon, jj, index_set);
      HYPRE_GMRESSetPrecond(gmres_solver,
                            (HYPRE_PtrToSolverFcn) HYPRE_BlockTridiagSolve,
                            (HYPRE_PtrToSolverFcn) HYPRE_BlockTridiagSetup,
                            precon);
      col_starts2 = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
      for (ii = 0; ii <= nprocs; ii++) { col_starts2[ii] = col_starts[ii]; }
      x = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_cols, col_starts2);
      hypre_ParVectorInitialize(x);
      x_local = hypre_ParVectorLocalVector(x);
      local_size = col_starts2[mypid + 1] - col_starts2[mypid];
      data = hypre_VectorData(x_local);
      for (ii = 0; ii < local_size; ii++) { data[ii] = 0.0; }
      row_starts2 = hypre_CTAlloc(HYPRE_Int,  nprocs + 1, HYPRE_MEMORY_HOST);
      for (ii = 0; ii <= nprocs; ii++) { row_starts2[ii] = row_starts[ii]; }
      y = hypre_ParVectorCreate(hypre_MPI_COMM_WORLD, global_num_rows, row_starts2);
      hypre_ParVectorInitialize(y);
      y_local = hypre_ParVectorLocalVector(y);
      data = hypre_VectorData(y_local);
      for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }

      HYPRE_GMRESSetup(gmres_solver, (HYPRE_Matrix) par_matrix,
                       (HYPRE_Vector) y, (HYPRE_Vector) x);
      HYPRE_GMRESSolve(gmres_solver, (HYPRE_Matrix) par_matrix,
                       (HYPRE_Vector) y, (HYPRE_Vector) x);

      hypre_ParVectorDestroy(x);
      hypre_ParVectorDestroy(y);
      hypre_ParCSRMatrixDestroy(par_matrix);
   }

   /* Finalize MPI */
   hypre_MPI_Finalize();
   return 0;
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

HYPRE_Int MyBuildParLaplacian9pt(HYPRE_ParCSRMatrix  *A_ptr)
{
   HYPRE_Int                 nx, ny;
   HYPRE_Int                 P, Q;
   HYPRE_ParCSRMatrix  A;
   HYPRE_Int                 num_procs, myid;
   HYPRE_Int                 p, q;
   HYPRE_Complex      *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &num_procs );
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 200;
   ny = 200;
   P  = 2;
   if (num_procs == 1) { P = 1; }
   Q  = num_procs / P;

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      hypre_printf("  Laplacian 9pt:\n");
      hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = hypre_CTAlloc(HYPRE_Complex,  2, HYPRE_MEMORY_HOST);
   values[1] = -1.;
   values[0] = 0.;
   if (nx > 1) { values[0] += 2.0; }
   if (ny > 1) { values[0] += 2.0; }
   if (nx > 1 && ny > 1) { values[0] += 4.0; }
   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);
   hypre_TFree(values, HYPRE_MEMORY_HOST);
   *A_ptr = A;
   return (0);
}
