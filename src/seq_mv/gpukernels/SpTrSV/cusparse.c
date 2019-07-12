#include "spkernels.h"
#include "cublas_v2.h"

void
GaussSeidel_cusparse1(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja;
   HYPRE_Real *d_a, *d_a_sorted, *d_b, *d_x;
   double t1, t2, ta;
   HYPRE_Real done = 1.0, dmone = -1.0;

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_a, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_a_sorted, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_b, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));

   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->data, nnz*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);

   cusparseStatus_t status;
   cusparseHandle_t handle=0;
   cusparseMatDescr_t descr_A=0, descr_L=0, descr_U=0;
   cublasHandle_t cublas_handle;

   /* initialize cublas library */
   if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
   {
      printf ("CUBLAS initialization failed\n");
      exit(1);
   }

   /* initialize cusparse library */
   status = cusparseCreate(&handle);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("CUSPARSE Library initialization failed\n");
      exit(1);
   }

   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descr_A) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO) );

   /* include sorting time in the analysis time */
   t1 = wall_timer();

   /* Sort A */
   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;
   HYPRE_Int *P = NULL;
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort_bufferSizeExt(handle, n, n, nnz, d_ia, d_ja, &pBufferSizeInBytes) );
   cudaMalloc((void **)&pBuffer, pBufferSizeInBytes);
   cudaMalloc((void **)&P, sizeof(HYPRE_Int)*nnz);
   HYPRE_CUSPARSE_CALL( cusparseCreateIdentityPermutation(handle, nnz, P) );
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort(handle, n, n, nnz, descr_A, d_ia, d_ja, P, pBuffer) );
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDgthr(handle, nnz, (hypre_double *) d_a, (hypre_double *) d_a_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseSgthr(handle, nnz, (float *) d_a, (float *) d_a_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   cudaFree(pBuffer);
   cudaFree(P);
   cudaFree(d_a);
   d_a = d_a_sorted;

   /* create and setup matrix descriptor for L */
   status = cusparseCreateMatDescr(&descr_L);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor initialization L failed\n");
      exit(1);
   }

   cusparseSolveAnalysisInfo_t info_L = 0;
   cusparseCreateSolveAnalysisInfo(&info_L);
   cusparseSetMatType(descr_L,CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
   cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
   cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

   if (isDoublePrecision)
   {
      status = cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                       descr_L, (hypre_double *) d_a, d_ia, d_ja, info_L);
   }
   else if (isSinglePrecision)
   {
      status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                       descr_L, (float *) d_a, d_ia, d_ja, info_L);
   }

   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("cusparse?csrsv_analysis L failed\n");
      exit(1);
   }

   /* create and setup matrix descriptor for U */
   status = cusparseCreateMatDescr(&descr_U);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor initialization U failed\n");
      exit(1);
   }

   cusparseSolveAnalysisInfo_t info_U = 0;
   cusparseCreateSolveAnalysisInfo(&info_U);
   cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
   cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
   cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

   if (isDoublePrecision)
   {
      status = cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                       descr_U, (hypre_double *) d_a, d_ia, d_ja, info_U);
   }
   else if (isSinglePrecision)
   {
      status = cusparseScsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                       descr_U, (float *) d_a, d_ia, d_ja, info_U);
   }

   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("cusparse?csrsv_analysis U failed\n");
      exit(1);
   }

   // Barrier for GPU calls
   cudaThreadSynchronize();

   ta = wall_timer() - t1;

   /* repeated solves */
   t1 = wall_timer();

   for (int j = 0; j < REPEAT; j++)
   {
      HYPRE_Real *d_y, *d_r;
      cudaMalloc((void **)&d_r, n*sizeof(HYPRE_Real));
      cudaMalloc((void **)&d_y, n*sizeof(HYPRE_Real));

      if (isDoublePrecision)
      {
         // Forward G-S. r = b - A * x
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (hypre_double *) &dmone, descr_A, (hypre_double *) d_a,
                                 d_ia, d_ja, (hypre_double *) d_x,
                                 (hypre_double *) &done, (hypre_double *) d_r) );
         // y = L \ r
         status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, (hypre_double *) &done,
               descr_L, (hypre_double *) d_a, d_ia, d_ja, info_L, (hypre_double *) d_r, (hypre_double *) d_y);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }
         // x = x + y
         cublasDaxpy(cublas_handle, n, (hypre_double *) &done, (hypre_double *) d_y, 1, (hypre_double *) d_x, 1);

         // Backward G-S. r = b - A * x
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (hypre_double *) &dmone, descr_A, (hypre_double *) d_a,
                                 d_ia, d_ja, (hypre_double *) d_x,
                                 (hypre_double *) &done, (hypre_double *) d_r) );
         // y = U \ r
         status = cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, (hypre_double *) &done,
               descr_U, (hypre_double *) d_a, d_ia, d_ja, info_U, (hypre_double *) d_r, (hypre_double *) d_y);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }
         // x = x + y
         cublasDaxpy(cublas_handle, n, (hypre_double *) &done, (hypre_double *) d_y, 1, (hypre_double *) d_x, 1);
      }
      else if (isSinglePrecision)
      {
         // Forward
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (float *) &dmone, descr_A, (float *) d_a,
                                 d_ia, d_ja, (float *) d_x,
                                 (float *) &done, (float *) d_r) );

         status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, (float *) &done,
               descr_L, (float *) d_a, d_ia, d_ja, info_L, (float *) d_r, (float *) d_y);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }

         cublasSaxpy(cublas_handle, n, (float *) &done, (float *) d_y, 1, (float *) d_x, 1);

         // Backward
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (float *) &dmone, descr_A, (float *) d_a,
                                 d_ia, d_ja, (float *) d_x,
                                 (float *) &done, (float *) d_r) );

         status = cusparseScsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, (float *) &done,
               descr_U, (float *) d_a, d_ia, d_ja, info_U, (float *) d_r, (float *) d_y);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }

         cublasSaxpy(cublas_handle, n, (float *) &done, (float *) d_y, 1, (float *) d_x, 1);
      }

      cudaFree(d_y);
      cudaFree(d_r);
   }

   //Barrier for GPU calls
   cudaThreadSynchronize();
   t2 = wall_timer() - t1;
   t2 /= REPEAT;

   if (print)
   {
      printf(" [GPU] CUSPARSE-1 G-S\n");
      printf("  time(s) = %.2e, Gflops = %-5.3f", t2, 4*((nnz)/1e9)/t2);
      printf("  [analysis time %.2e (%.1e x T_sol)] ", ta, ta/t2);
   }

   /*-------- copy x to host mem */
   cudaMemcpy(x, d_x, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_x);

   /* destroy matrix descriptor */
   status = cusparseDestroyMatDescr(descr_L);
   descr_L = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor destruction failed\n");
      exit(1);
   }

   status = cusparseDestroyMatDescr(descr_U);
   descr_U = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor destruction failed\n");
      exit(1);
   }

   status = cusparseDestroySolveAnalysisInfo(info_L);
   info_L = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("analysis info destruction failed\n");
      exit(1);
   }

   status = cusparseDestroySolveAnalysisInfo(info_U);
   info_U = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("analysis info destruction failed\n");
      exit(1);
   }

   /* destroy handle */
   status = cusparseDestroy(handle);
   handle = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("CUSPARSE Library release of resources failed\n");
      exit(1);
   }

   if (CUBLAS_STATUS_SUCCESS != cublasDestroy(cublas_handle))
   {
      printf("CUBLAS Library release of resources failed\n");
      exit(1);
   }
}

void
GaussSeidel_cusparse2(hypre_CSRMatrix *csr, HYPRE_Real *b, HYPRE_Real *x, int REPEAT, bool print)
{
   int n = csr->num_rows;
   int nnz = csr->num_nonzeros;
   int *d_ia, *d_ja;
   HYPRE_Real *d_a, *d_a_sorted, *d_b, *d_x;
   double t1, t2, ta;
   HYPRE_Real done = 1.0, dmone = -1.0;

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   /*------------------- allocate Device Memory */
   cudaMalloc((void **)&d_ia, (n+1)*sizeof(int));
   cudaMalloc((void **)&d_ja, nnz*sizeof(int));
   cudaMalloc((void **)&d_a, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_a_sorted, nnz*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_b, n*sizeof(HYPRE_Real));
   cudaMalloc((void **)&d_x, n*sizeof(HYPRE_Real));

   /*------------------- Memcpy */
   cudaMemcpy(d_ia, csr->i, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, csr->j, nnz*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_a, csr->data, nnz*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);
   cudaMemcpy(d_x, x, n*sizeof(HYPRE_Real), cudaMemcpyHostToDevice);

   cusparseStatus_t status;
   cusparseHandle_t handle=0;
   cusparseMatDescr_t descr_A=0, descr_L=0, descr_U=0;
   cublasHandle_t cublas_handle;

   /* initialize cublas library */
   if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
   {
      printf ("CUBLAS initialization failed\n");
      exit(1);
   }

   /* initialize cusparse library */
   status = cusparseCreate(&handle);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("CUSPARSE Library initialization failed\n");
      exit(1);
   }

   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&descr_A) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(descr_A, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(descr_A, CUSPARSE_INDEX_BASE_ZERO) );

   /* include sorting time in the analysis time */
   t1 = wall_timer();

   /* Sort A */
   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;
   HYPRE_Int *P = NULL;
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort_bufferSizeExt(handle, n, n, nnz, d_ia, d_ja, &pBufferSizeInBytes) );
   cudaMalloc((void **)&pBuffer, pBufferSizeInBytes);
   cudaMalloc((void **)&P, sizeof(HYPRE_Int)*nnz);
   HYPRE_CUSPARSE_CALL( cusparseCreateIdentityPermutation(handle, nnz, P) );
   HYPRE_CUSPARSE_CALL( cusparseXcsrsort(handle, n, n, nnz, descr_A, d_ia, d_ja, P, pBuffer) );
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDgthr(handle, nnz, (hypre_double *) d_a, (hypre_double *) d_a_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseSgthr(handle, nnz, (float *) d_a, (float *) d_a_sorted, P, CUSPARSE_INDEX_BASE_ZERO) );
   }
   cudaFree(pBuffer);
   cudaFree(P);
   cudaFree(d_a);
   d_a = d_a_sorted;

   /* create and setup matrix descriptor for L */
   status = cusparseCreateMatDescr(&descr_L);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor initialization L failed\n");
      exit(1);
   }

   csrsv2Info_t info_L = 0;
   cusparseCreateCsrsv2Info(&info_L);
   cusparseSetMatType(descr_L,CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
   cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
   cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

   int pBufferSize_L;
   void *pBuffer_L = 0;
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
               descr_L, (hypre_double *) d_a, d_ia, d_ja, info_L, &pBufferSize_L) );

      // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
      cudaMalloc((void**)&pBuffer_L, pBufferSize_L);

      status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                        descr_L, (hypre_double *) d_a, d_ia, d_ja, info_L,
                                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                             descr_L, (float *) d_a, d_ia, d_ja, info_L, &pBufferSize_L) );

      // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
      cudaMalloc((void**)&pBuffer_L, pBufferSize_L);

      status = cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                        descr_L, (float *) d_a, d_ia, d_ja, info_L,
                                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);
   }

   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("cusparse?csrsv_analysis L failed\n");
      exit(1);
   }

   /* create and setup matrix descriptor for U */
   status = cusparseCreateMatDescr(&descr_U);
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor initialization U failed\n");
      exit(1);
   }

   csrsv2Info_t info_U = 0;
   cusparseCreateCsrsv2Info(&info_U);
   cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
   cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
   cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
   cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

   int pBufferSize_U;
   void *pBuffer_U = 0;
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
               descr_U, (hypre_double *) d_a, d_ia, d_ja, info_U, &pBufferSize_U) );

      // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
      cudaMalloc((void**)&pBuffer_U, pBufferSize_U);

      status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                        descr_U, (hypre_double *) d_a, d_ia, d_ja, info_U,
                                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
               descr_U, (float *) d_a, d_ia, d_ja, info_U, &pBufferSize_U) );

      // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
      cudaMalloc((void**)&pBuffer_U, pBufferSize_U);

      status = cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz,
                                        descr_U, (float *) d_a, d_ia, d_ja, info_U,
                                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);
   }

   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("cusparse?csrsv_analysis U failed\n");
      exit(1);
   }

   // Barrier for GPU calls
   cudaThreadSynchronize();

   ta = wall_timer() - t1;

   /* repeated solves */
   t1 = wall_timer();

   for (int j = 0; j < REPEAT; j++)
   {
      HYPRE_Real *d_y, *d_r;
      cudaMalloc((void **)&d_r, n*sizeof(HYPRE_Real));
      cudaMalloc((void **)&d_y, n*sizeof(HYPRE_Real));

      if (isDoublePrecision)
      {
         // Forward G-S. r = b - A * x
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (hypre_double *) &dmone, descr_A, (hypre_double *) d_a,
                                 d_ia, d_ja, (hypre_double *) d_x,
                                 (hypre_double *) &done, (hypre_double *) d_r) );
         // y = L \ r
         status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, (hypre_double *) &done,
               descr_L, (hypre_double *) d_a, d_ia, d_ja, info_L, (hypre_double *) d_r, (hypre_double *) d_y,
               CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }
         // x = x + y
         cublasDaxpy(cublas_handle, n, (hypre_double *) &done, (hypre_double *) d_y, 1, (hypre_double *) d_x, 1);

         // Backward G-S. r = b - A * x
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (hypre_double *) &dmone, descr_A, (hypre_double *) d_a,
                                 d_ia, d_ja, (hypre_double *) d_x,
                                 (hypre_double *) &done, (hypre_double *) d_r) );
         // y = U \ r
         status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, (hypre_double *) &done,
               descr_U, (hypre_double *) d_a, d_ia, d_ja, info_U, (hypre_double *) d_r, (hypre_double *) d_y,
               CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }
         // x = x + y
         cublasDaxpy(cublas_handle, n, (hypre_double *) &done, (hypre_double *) d_y, 1, (hypre_double *) d_x, 1);
      }
      else if (isSinglePrecision)
      {
         // Forward
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (float *) &dmone, descr_A, (float *) d_a,
                                 d_ia, d_ja, (float *) d_x,
                                 (float *) &done, (float *) d_r) );

         status = cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, (float *) &done,
               descr_L, (float *) d_a, d_ia, d_ja, info_L, (float *) d_r, (float *) d_y,
               CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_L);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }

         cublasSaxpy(cublas_handle, n, (float *) &done, (float *) d_y, 1, (float *) d_x, 1);

         // Backward
         cudaMemcpy(d_r, d_b, sizeof(HYPRE_Real)*n, cudaMemcpyDeviceToDevice);
         HYPRE_CUSPARSE_CALL( cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 n, n, nnz, (float *) &dmone, descr_A, (float *) d_a,
                                 d_ia, d_ja, (float *) d_x,
                                 (float *) &done, (float *) d_r) );

         status = cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, (float *) &done,
               descr_U, (float *) d_a, d_ia, d_ja, info_U, (float *) d_r, (float *) d_y,
               CUSPARSE_SOLVE_POLICY_USE_LEVEL, pBuffer_U);

         if (status != CUSPARSE_STATUS_SUCCESS)
         {
            printf("cusparse?csrsv_solve L failed\n");
            exit(1);
         }

         cublasSaxpy(cublas_handle, n, (float *) &done, (float *) d_y, 1, (float *) d_x, 1);
      }

      cudaFree(d_y);
      cudaFree(d_r);
   }

   //Barrier for GPU calls
   cudaThreadSynchronize();
   t2 = wall_timer() - t1;
   t2 /= REPEAT;

   if (print)
   {
      printf(" [GPU] CUSPARSE-2 G-S\n");
      printf("  time(s) = %.2e, Gflops = %-5.3f", t2, 4*((nnz)/1e9)/t2);
      printf("  [analysis time %.2e (%.1e x T_sol)] ", ta, ta/t2);
   }

   /*-------- copy x to host mem */
   cudaMemcpy(x, d_x, n*sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_x);
   cudaFree(pBuffer_L);
   cudaFree(pBuffer_U);

   /* destroy matrix descriptor */
   status = cusparseDestroyMatDescr(descr_L);
   descr_L = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor destruction failed\n");
      exit(1);
   }

   status = cusparseDestroyMatDescr(descr_U);
   descr_U = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("Matrix descriptor destruction failed\n");
      exit(1);
   }

   status = cusparseDestroyCsrsv2Info(info_L);
   info_L = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("analysis info destruction failed\n");
      exit(1);
   }

   status = cusparseDestroyCsrsv2Info(info_U);
   info_U = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("analysis info destruction failed\n");
      exit(1);
   }

   /* destroy handle */
   status = cusparseDestroy(handle);
   handle = 0;
   if (status != CUSPARSE_STATUS_SUCCESS)
   {
      printf("CUSPARSE Library release of resources failed\n");
      exit(1);
   }

   if (CUBLAS_STATUS_SUCCESS != cublasDestroy(cublas_handle))
   {
      printf("CUBLAS Library release of resources failed\n");
      exit(1);
   }
}

