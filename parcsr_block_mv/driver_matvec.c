 
#include "../utilities/utilities.h"
#include "../seq_mv/seq_mv.h"
#include "../parcsr_mv/parcsr_mv.h"
#include "../parcsr_ls/parcsr_ls.h"
#include "../krylov/krylov.h"
#include "par_csr_block_matrix.h"
 
extern int MyBuildParLaplacian9pt(HYPRE_ParCSRMatrix  *A_ptr);

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface 
 *--------------------------------------------------------------------------*/
 
int main( int   argc, char *argv[] )
{
   hypre_ParCSRMatrix      *par_matrix, *g_matrix, **submatrices;
   hypre_CSRMatrix         *A_diag, *A_offd;
   hypre_CSRBlockMatrix    *diag;
   hypre_CSRBlockMatrix    *offd;
   hypre_ParCSRBlockMatrix *par_blk_matrix, *par_blk_matrix2, *rap_matrix;
   hypre_Vector        *x_local;
   hypre_Vector        *y_local;
   hypre_ParVector     *x;
   hypre_ParVector     *y;
   HYPRE_Solver        gmres_solver, precon;
   int                 *diag_i, *diag_j, *offd_i, *offd_j;
   int                 *diag_i2, *diag_j2, *offd_i2, *offd_j2;
   double              *diag_d, *diag_d2, *offd_d, *offd_d2;
   int		       mypid, local_size, nprocs;
   int		       global_num_rows, global_num_cols, num_cols_offd;
   int		       num_nonzeros_diag, num_nonzeros_offd, *colMap;
   int 		       ii, jj, kk, row, col, nnz, *indices, *colMap2;
   double 	       *data, ddata, *y_data;
   int 		       *row_starts, *col_starts, *rstarts, *cstarts;
   int 		       *row_starts2, *col_starts2;
   int                 block_size=2, bnnz=4, *index_set;
   FILE                *fp;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

   /* build vector and apply matvec */
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,global_num_cols,col_starts);
   hypre_ParVectorSetPartitioningOwner(x,0);
   hypre_ParVectorInitialize(x);
   x_local = hypre_ParVectorLocalVector(x);
   data    = hypre_VectorData(x_local);
   local_size = col_starts[mypid+1] - col_starts[mypid];
   for (ii = 0; ii < local_size; ii++) data[ii] = 1.0;
   y = hypre_ParVectorCreate(MPI_COMM_WORLD,global_num_rows,row_starts);
   hypre_ParVectorSetPartitioningOwner(y,0);
   hypre_ParVectorInitialize(y);
   hypre_ParCSRMatrixMatvec (1.0, par_matrix, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) printf("y inner product = %e\n", ddata);
   hypre_ParVectorDestroy(x);
   hypre_ParVectorDestroy(y);

   /* build block matrix */
   rstarts = (int *) malloc((nprocs + 1) * sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) rstarts[ii] = row_starts[ii];
   cstarts = (int *) malloc((nprocs + 1) * sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) cstarts[ii] = col_starts[ii];

   par_blk_matrix = hypre_ParCSRBlockMatrixCreate(MPI_COMM_WORLD,block_size,
                          global_num_rows, global_num_cols, rstarts,
                          cstarts, num_cols_offd, num_nonzeros_diag,
                          num_nonzeros_offd);
   colMap  = hypre_ParCSRMatrixColMapOffd(par_matrix);
   colMap2 = (int *) malloc(num_cols_offd * sizeof(int));
   for (ii = 0; ii < num_cols_offd; ii++) colMap2[ii] = colMap[ii];
   hypre_ParCSRBlockMatrixColMapOffd(par_blk_matrix) = colMap2;
   diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
   diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(par_matrix));
   diag_d = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(par_matrix));
   diag = hypre_ParCSRBlockMatrixDiag(par_blk_matrix);
   diag_i2 = (int *) malloc((local_size+1) * sizeof(int));
   diag_j2 = (int *) malloc(num_nonzeros_diag * sizeof(int));
   diag_d2 = (double *) malloc(num_nonzeros_diag * bnnz * sizeof(double));
   for (ii = 0; ii <= local_size; ii++) diag_i2[ii] = diag_i[ii];
   for (ii = 0; ii < num_nonzeros_diag; ii++) diag_j2[ii] = diag_j[ii];
   hypre_CSRBlockMatrixI(diag) = diag_i2;
   hypre_CSRBlockMatrixJ(diag) = diag_j2;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      for (jj = 0; jj < block_size; jj++)
         for (kk = 0; kk < block_size; kk++)
         {
            if (jj <= kk)
               diag_d2[ii*bnnz+jj*block_size+kk] = diag_d[ii];
            else
               diag_d2[ii*bnnz+jj*block_size+kk] = 0.0;
         }
   }
   hypre_CSRBlockMatrixData(diag) = diag_d2;

   offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
   offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix));
   offd_d = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(par_matrix));
   offd   = hypre_ParCSRBlockMatrixOffd(par_blk_matrix);
   offd_i2 = (int *) malloc((local_size+1) * sizeof(int));
   for (ii = 0; ii <= local_size; ii++) offd_i2[ii] = offd_i[ii];
   hypre_CSRBlockMatrixI(offd) = offd_i2;
   if (num_cols_offd)
   {
      offd_j2 = (int *) malloc(num_nonzeros_offd * sizeof(int));
      for (ii = 0; ii < num_nonzeros_offd; ii++) offd_j2[ii] = offd_j[ii];
      hypre_CSRBlockMatrixJ(offd) = offd_j2;
      offd_d2 = (double *) malloc(num_nonzeros_offd * bnnz * sizeof(double));
      for (ii = 0; ii < num_nonzeros_offd; ii++)
      {
         for (jj = 0; jj < block_size; jj++)
            for (kk = 0; kk < block_size; kk++)
            {
               if (jj <= kk)
                  offd_d2[ii*bnnz+jj*block_size+kk] = offd_d[ii];
               else
                  offd_d2[ii*bnnz+jj*block_size+kk] = 0.0;
            }
      }
      hypre_CSRBlockMatrixData(offd) = offd_d2;
   }
   else
   {
      hypre_CSRBlockMatrixJ(offd) = NULL;
      hypre_CSRBlockMatrixData(offd) = NULL;
   }

   /* build block matrix 2 */
   rstarts = (int *) malloc((nprocs + 1) * sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) rstarts[ii] = row_starts[ii];
   cstarts = (int *) malloc((nprocs + 1) * sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) cstarts[ii] = col_starts[ii];

   par_blk_matrix2 = hypre_ParCSRBlockMatrixCreate(MPI_COMM_WORLD,block_size,
                          global_num_rows, global_num_cols, rstarts,
                          cstarts, num_cols_offd, num_nonzeros_diag,
                          num_nonzeros_offd);
   colMap  = hypre_ParCSRMatrixColMapOffd(par_matrix);
   colMap2 = (int *) malloc(num_cols_offd * sizeof(int));
   for (ii = 0; ii < num_cols_offd; ii++) colMap2[ii] = colMap[ii];
   hypre_ParCSRBlockMatrixColMapOffd(par_blk_matrix2) = colMap2;
   diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(par_matrix));
   diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(par_matrix));
   diag_d = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(par_matrix));
   diag = hypre_ParCSRBlockMatrixDiag(par_blk_matrix2);
   diag_i2 = (int *) malloc((local_size+1) * sizeof(int));
   diag_j2 = (int *) malloc(num_nonzeros_diag * sizeof(int));
   diag_d2 = (double *) malloc(num_nonzeros_diag * bnnz * sizeof(double));
   for (ii = 0; ii <= local_size; ii++) diag_i2[ii] = diag_i[ii];
   for (ii = 0; ii < num_nonzeros_diag; ii++) diag_j2[ii] = diag_j[ii];
   hypre_CSRBlockMatrixI(diag) = diag_i2;
   hypre_CSRBlockMatrixJ(diag) = diag_j2;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      for (jj = 0; jj < block_size; jj++)
         for (kk = 0; kk < block_size; kk++)
         {
            if (jj >= kk)
               diag_d2[ii*bnnz+jj*block_size+kk] = diag_d[ii];
            else
               diag_d2[ii*bnnz+jj*block_size+kk] = 0.0;
         }
   }
   hypre_CSRBlockMatrixData(diag) = diag_d2;

   offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(par_matrix));
   offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(par_matrix));
   offd_d = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(par_matrix));
   offd   = hypre_ParCSRBlockMatrixOffd(par_blk_matrix2);
   offd_i2 = (int *) malloc((local_size+1) * sizeof(int));
   for (ii = 0; ii <= local_size; ii++) offd_i2[ii] = offd_i[ii];
   hypre_CSRBlockMatrixI(offd) = offd_i2;
   if (num_cols_offd)
   {
      offd_j2 = (int *) malloc(num_nonzeros_offd * sizeof(int));
      for (ii = 0; ii < num_nonzeros_offd; ii++) offd_j2[ii] = offd_j[ii];
      hypre_CSRBlockMatrixJ(offd) = offd_j2;
      offd_d2 = (double *) malloc(num_nonzeros_offd * bnnz * sizeof(double));
      for (ii = 0; ii < num_nonzeros_offd; ii++)
      {
         for (jj = 0; jj < block_size; jj++)
            for (kk = 0; kk < block_size; kk++)
            {
               if (jj >= kk)
                  offd_d2[ii*bnnz+jj*block_size+kk] = offd_d[ii];
               else
                  offd_d2[ii*bnnz+jj*block_size+kk] = 0.0;
            }
      }
      hypre_CSRBlockMatrixData(offd) = offd_d2;
   }
   else
   {
      hypre_CSRBlockMatrixJ(offd) = NULL;
      hypre_CSRBlockMatrixData(offd) = NULL;
   }

   /* block matvec */
   col_starts2 = (int *) malloc((nprocs+1)*sizeof(int));
   for (ii = 0; ii <= nprocs; ii++)
      col_starts2[ii] = col_starts[ii] * block_size;
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,global_num_cols*block_size,
                             col_starts2);
   hypre_ParVectorInitialize(x);
   x_local = hypre_ParVectorLocalVector(x);
   data = hypre_VectorData(x_local);
   local_size = col_starts2[mypid+1] - col_starts2[mypid];
   for (ii=0; ii < local_size; ii++) data[ii] = 1.0;
   row_starts2 = (int *) malloc((nprocs+1)*sizeof(int));
   for (ii = 0; ii <= nprocs; ii++)
      row_starts2[ii] = row_starts[ii] * block_size;
   y = hypre_ParVectorCreate(MPI_COMM_WORLD,global_num_rows*block_size,
                             row_starts2);
   hypre_ParVectorSetPartitioningOwner(y,0);
   hypre_ParVectorInitialize(y);
   y_local = hypre_ParVectorLocalVector(y);
   y_data  = hypre_VectorData(y_local);

   hypre_BlockMatvecCommPkgCreate(par_blk_matrix);
   ddata = hypre_ParVectorInnerProd(x, x);
   if (mypid == 0) printf("block x inner product = %e\n", ddata);
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) printf("block y inner product = %e\n", ddata);

   /* RAP */
   hypre_ParCSRBlockMatrixRAP(par_blk_matrix, par_blk_matrix,
                              par_blk_matrix, &rap_matrix);
   for (ii = 0; ii < local_size; ii++) data[ii] = 1.0;
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, x, 0.0, y);
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, y, 0.0, x);
   hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix2, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) printf("(1) A^2 block inner product = %e\n", ddata);
   for (ii = 0; ii < local_size; ii++) data[ii] = 1.0;
   hypre_ParCSRBlockMatrixMatvec (1.0, rap_matrix, x, 0.0, y);
   ddata = hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) printf("(2) A^2 block inner product = %e\n", ddata);
   hypre_ParVectorDestroy(x);
   hypre_ParVectorDestroy(y);

#if 0
   diag_i = hypre_CSRBlockMatrixI(A_diag);
   diag_j = hypre_CSRBlockMatrixJ(A_diag);
   diag_d = hypre_CSRBlockMatrixData(A_diag);
   for (ii = 0; ii < hypre_ParCSRMatrixNumRows(par_matrix); ii++) 
      for (jj = diag_i[ii]; jj < diag_i[ii+1]; jj++) 
         printf("A %4d %4d = %e\n",ii,diag_j[jj],diag_d[jj]);
#endif

#if 0
   diag = hypre_ParCSRBlockMatrixDiag(rap_matrix);
   diag_i = hypre_CSRBlockMatrixI(diag);
   diag_j = hypre_CSRBlockMatrixJ(diag);
   diag_d = hypre_CSRBlockMatrixData(diag);
   printf("RAP block size = %d\n",hypre_ParCSRBlockMatrixBlockSize(rap_matrix));
   printf("RAP num rows   = %d\n",hypre_ParCSRBlockMatrixNumRows(rap_matrix));
   for (ii = 0; ii < hypre_ParCSRBlockMatrixNumRows(rap_matrix); ii++) 
      for (row = 0; row < block_size; row++) 
         for (jj = diag_i[ii]; jj < diag_i[ii+1]; jj++) 
            for (col = 0; col < block_size; col++) 
               printf("RAP %4d %4d = %e\n",ii*block_size+row,
                      diag_j[jj]*block_size+col,diag_d[(jj+row)*block_size+col]);
   offd = hypre_ParCSRBlockMatrixOffd(rap_matrix);
   offd_i = hypre_CSRBlockMatrixI(offd);
   offd_j = hypre_CSRBlockMatrixJ(offd);
   offd_d = hypre_CSRBlockMatrixData(offd);
   if (num_cols_offd)
   {
      for (ii = 0; ii < hypre_ParCSRBlockMatrixNumRows(rap_matrix); ii++) 
         for (row = 0; row < block_size; row++) 
            for (jj = offd_i[ii]; jj < offd_i[ii+1]; jj++) 
               for (col = 0; col < block_size; col++) 
                  printf("RAPOFFD %4d %4d = %e\n",ii*block_size+row,
                         offd_j[jj]*block_size+col,offd_d[(jj+row)*block_size+col]);
   }
#endif

   /* read in the matrix and create a HYPRE_ParCSRMatrix */

#if 0
   fp = fopen("Amat_ee", "r");
   fscanf(fp, "%d %d", &global_num_rows, &num_nonzeros_diag);
   diag_i = (int *) malloc((global_num_rows+1) * sizeof(int));
   diag_j = (int *) malloc(num_nonzeros_diag * sizeof(int));
   diag_d = (double *) malloc(num_nonzeros_diag * sizeof(double));
   row = 0;
   nnz = 0;
   diag_i[0] = 0;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      fscanf(fp, "%d %d %lg", &jj, &col, &ddata);
      if ((jj-1) != row)
      {
         row++;
         diag_i[row] = nnz;
      }
      diag_j[nnz] = col - 1;
      diag_d[nnz++] = ddata;
   }
   diag_i[global_num_rows] = nnz;
   fclose(fp);
   printf("nrows = %d, nnz = %d\n", row+1, nnz);

   row_starts = (int *) malloc(2 * sizeof(int));
   col_starts = (int *) malloc(2 * sizeof(int));
   row_starts[0] = col_starts[0] = 0;
   row_starts[1] = col_starts[1] = global_num_rows;
   num_cols_offd = 0;
   num_nonzeros_offd = 0;
   par_matrix = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD,global_num_rows, 
                      global_num_rows, row_starts, col_starts, num_cols_offd, 
                      num_nonzeros_diag, num_nonzeros_offd);
   A_diag = hypre_ParCSRMatrixDiag(par_matrix);
   hypre_CSRMatrixI(A_diag) = diag_i;
   hypre_CSRMatrixJ(A_diag) = diag_j;
   hypre_CSRMatrixData(A_diag) = diag_d;

   fp = fopen("Gmat", "r");
   fscanf(fp, "%d %d %d", &global_num_rows, &global_num_cols, &num_nonzeros_diag);
   diag_i = (int *) malloc((global_num_rows+1) * sizeof(int));
   diag_j = (int *) malloc(num_nonzeros_diag * sizeof(int));
   diag_d = (double *) malloc(num_nonzeros_diag * sizeof(double));
   row = 0;
   nnz = 0;
   diag_i[0] = 0;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      fscanf(fp, "%d %d %lg", &jj, &col, &ddata);
      if ((jj-1) != row)
      {
         row++;
         diag_i[row] = nnz;
      }
      diag_j[nnz] = col - 1;
      diag_d[nnz++] = ddata;
   }
   diag_i[global_num_rows] = nnz;
   fclose(fp);
   printf("nrows = %d, ncols = %d, nnz = %d\n", row+1, global_num_cols, nnz);

   row_starts = (int *) malloc(2 * sizeof(int));
   col_starts = (int *) malloc(2 * sizeof(int));
   row_starts[0] = col_starts[0] = 0;
   row_starts[1] = global_num_rows;
   col_starts[1] = global_num_cols;
   num_cols_offd = 0;
   num_nonzeros_offd = 0;
   g_matrix = hypre_ParCSRMatrixCreate(MPI_COMM_WORLD,global_num_rows, 
                      global_num_cols, row_starts, col_starts, num_cols_offd, 
                      num_nonzeros_diag, num_nonzeros_offd);
   A_diag = hypre_ParCSRMatrixDiag(g_matrix);
   hypre_CSRMatrixI(A_diag) = diag_i;
   hypre_CSRMatrixJ(A_diag) = diag_j;
   hypre_CSRMatrixData(A_diag) = diag_d;

   hypre_ParCSRMatrixGenSpanningTree(g_matrix, &indices, 0);

   submatrices = (hypre_ParCSRMatrix **) malloc(4 * sizeof(hypre_ParCSRMatrix *));
   hypre_ParCSRMatrixExtractSubmatrices(par_matrix, indices, &submatrices);
#endif

   /* test block tridiagonal solver */

printf("check 1\n");
   HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &gmres_solver);
   HYPRE_GMRESSetKDim(gmres_solver, 10);
   HYPRE_GMRESSetMaxIter(gmres_solver, 1000);
   HYPRE_GMRESSetTol(gmres_solver, 1.0e-6);
   HYPRE_GMRESSetLogging(gmres_solver, 1);
   HYPRE_GMRESSetPrintLevel(gmres_solver, 1);
   HYPRE_BlockTridiagCreate(&precon);
   local_size = col_starts[mypid+1] - col_starts[mypid];
   index_set = (int *) malloc((local_size+1) * sizeof(int));
   jj = 0;
   for (ii = 0; ii < local_size/2; ii++) index_set[jj++] = (ii-1) * 2;
printf("check 1 %d\n", jj);
   HYPRE_BlockTridiagSetIndexSet(precon, jj, index_set);
   HYPRE_GMRESSetPrecond(gmres_solver, 
                         (HYPRE_PtrToSolverFcn) HYPRE_BlockTridiagSolve,
                         (HYPRE_PtrToSolverFcn) HYPRE_BlockTridiagSetup,
                         precon);
   col_starts2 = (int *) malloc((nprocs+1)*sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) col_starts2[ii] = col_starts[ii];
   x = hypre_ParVectorCreate(MPI_COMM_WORLD,global_num_cols,col_starts2);
   hypre_ParVectorInitialize(x);
   x_local = hypre_ParVectorLocalVector(x);
   local_size = col_starts2[mypid+1] - col_starts2[mypid];
   data = hypre_VectorData(x_local);
   for (ii=0; ii < local_size; ii++) data[ii] = 0.0;
   row_starts2 = (int *) malloc((nprocs+1)*sizeof(int));
   for (ii = 0; ii <= nprocs; ii++) row_starts2[ii] = row_starts[ii];
   y = hypre_ParVectorCreate(MPI_COMM_WORLD,global_num_rows,row_starts2);
   hypre_ParVectorInitialize(y);
   y_local = hypre_ParVectorLocalVector(y);
   data = hypre_VectorData(y_local);
   for (ii=0; ii < local_size; ii++) data[ii] = 1.0;

printf("check 1\n");
   HYPRE_GMRESSetup(gmres_solver, (HYPRE_Matrix) par_matrix,
                    (HYPRE_Vector) y,
                    (HYPRE_Vector) x);
printf("check 2\n");
 
   hypre_ParCSRMatrixDestroy(par_matrix);
   hypre_ParVectorDestroy(x);
   hypre_ParVectorDestroy(y);
   hypre_ParCSRBlockMatrixDestroy(par_blk_matrix);
   hypre_ParCSRBlockMatrixDestroy(par_blk_matrix2);
   hypre_ParCSRBlockMatrixDestroy(rap_matrix);

   /* Finalize MPI */
   MPI_Finalize();
   return 0;
}
                                                                                       
/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/
                                                                                       
int MyBuildParLaplacian9pt(HYPRE_ParCSRMatrix  *A_ptr)
{
   int                 nx, ny;
   int                 P, Q;
                                                                                       
   HYPRE_ParCSRMatrix  A;
                                                                                       
   int                 num_procs, myid;
   int                 p, q;
   double             *values;
                                                                                       
   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
                                                                                       
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs );
   MPI_Comm_rank(MPI_COMM_WORLD, &myid );
                                                                                       
   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
                                                                                       
   nx = 100;
   ny = 100;
   P  = 2;
   if (num_procs == 1) P = 1;
   Q  = num_procs/P;
                                                                                       
   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
                                                                                       
   if (myid == 0)
   {
      printf("  Laplacian 9pt:\n");
      printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }
                                                                                       
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/
                                                                                       
   p = myid % P;
   q = ( myid - p)/P;
                                                                                       
   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/
                                                                                       
   values = hypre_CTAlloc(double, 2);
   values[1] = -1.;
   values[0] = 0.;
   if (nx > 1) values[0] += 2.0;
   if (ny > 1) values[0] += 2.0;
   if (nx > 1 && ny > 1) values[0] += 4.0;
   A = (HYPRE_ParCSRMatrix) GenerateLaplacian9pt(MPI_COMM_WORLD,
                                  nx, ny, P, Q, p, q, values);
   hypre_TFree(values);
   *A_ptr = A;
   return (0);
}

