#include "spkernels.h"

hypre_double wall_timer()
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  hypre_double t = tim.tv_sec + tim.tv_usec/1e6;

  return(t);
}

/*---------------------------------------------*/
void print_header()
{
   if (sizeof(HYPRE_Real) == sizeof(hypre_double))
   {
      printf("\nTesting SpMV, DOUBLE precision\n");
   }
   else if (sizeof(HYPRE_Real) == sizeof(float))
   {
      printf("\nTesting SpMV, SINGLE precision\n");
   }
}

/*-----------------------------------------*/
HYPRE_Real error_norm(HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int n) {
  HYPRE_Int i;
  HYPRE_Real t, normz, normx;
  normx = normz = 0.0;
  for (i=0; i<n; i++) {
    t = x[i]-y[i];
    normz += t*t;
    normx += x[i]*x[i];
  }
  return (sqrt(normz/normx));
}

/*---------------------------*/
void FreeCOO(struct coo_t *coo)
{
  free(coo->ir);
  free(coo->jc);
  free(coo->val);
}

/**
 * @brief convert csr to csc
 * Assume input csr is 0-based index
 * output csc 0/1 index specified by OUTINDEX      *
 * @param[in] OUTINDEX specifies if CSC should be 0/1 index
 * @param[in] nrow Number of rows
 * @param[in] ncol Number of columns
 * @param[in] job flag
 * @param[in] a Values of input matrix
 * @param[in] ia Input row pointers
 * @param[in] ja Input column indices
 * @param[out] ao Output values
 * @param[out] iao Output row pointers
 * @param[out] jao Output column indices
 */
void csrcsc(HYPRE_Int OUTINDEX, const HYPRE_Int nrow, const HYPRE_Int ncol, HYPRE_Int job,
            HYPRE_Real *a, HYPRE_Int *ja, HYPRE_Int *ia,
            HYPRE_Real *ao, HYPRE_Int *jao, HYPRE_Int *iao) {
  HYPRE_Int i,k;
  for (i=0; i<ncol+1; i++) {
    iao[i] = 0;
  }
  // compute nnz of columns of A
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      iao[ja[k]+1] ++;
    }
  }
  // compute pointers from lengths
  for (i=0; i<ncol; i++) {
    iao[i+1] += iao[i];
  }
  // now do the actual copying
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      HYPRE_Int j = ja[k];
      if (job) {
        ao[iao[j]] = a[k];
      }
      jao[iao[j]++] = i + OUTINDEX;
    }
  }
  /*---- reshift iao and leave */
  for (i=ncol; i>0; i--) {
    iao[i] = iao[i-1] + OUTINDEX;
  }
  iao[0] = OUTINDEX;
}

/**
 * @brief  Sort each row of a csr by increasing column
 * order
 * By Double transposition
 * @param[in] A Matrix to sort
 */
void sortrow(hypre_CSRMatrix *A) {
  /*-------------------------------------------*/
  HYPRE_Int nrows = A->num_rows;
  HYPRE_Int ncols = A->num_cols;
  HYPRE_Int nnz = A->i[nrows];
  // work array
  HYPRE_Real *b;
  HYPRE_Int *jb, *ib;
  b = (HYPRE_Real *) malloc(nnz*sizeof(HYPRE_Real));
  jb = (HYPRE_Int *) malloc(nnz*sizeof(HYPRE_Int));
  ib = (HYPRE_Int *) malloc((ncols+1)*sizeof(HYPRE_Int));
  // Double transposition
  csrcsc(0, nrows, ncols, 1, A->data, A->j, A->i, b, jb, ib);
  csrcsc(0, ncols, nrows, 1, b, jb, ib, A->data, A->j, A->i);
  // free
  free(b);
  free(jb);
  free(ib);
}

void spmv_csr_cpu(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y, int REPEAT)
{
   hypre_Vector *vx = hypre_SeqVectorCreate(csr->num_cols);
   hypre_Vector *vy = hypre_SeqVectorCreate(csr->num_rows);
   hypre_VectorMemoryLocation(vx) = HYPRE_MEMORY_HOST;
   hypre_VectorMemoryLocation(vy) = HYPRE_MEMORY_HOST;
   hypre_VectorOwnsData(vx) = 0;
   hypre_VectorOwnsData(vy) = 0;
   hypre_VectorData(vx) = x;
   hypre_VectorData(vy) = y;

   /*------------- CPU CSR SpMV kernel */
   hypre_double t1, t2;
   t1 = wall_timer();
   for (HYPRE_Int ii=0; ii<REPEAT; ii++)
   {
      hypre_CSRMatrixMatvecOutOfPlaceHost(1.0, csr, vx, 0.0, vy, vy, 0);
   }
   t2 = wall_timer() - t1;
   /*--------------------------------------------------*/
   printf("\n=== [CPU] CSR Kernel ===\n");
   printf("  %.2f ms, %.2f GFLOPS\n",
         t2*1e3/REPEAT, 2*(csr->i[csr->num_rows])/t2/1e9*REPEAT);

   hypre_SeqVectorDestroy(vx);
   hypre_SeqVectorDestroy(vy);
}


void GaussSeidelCPU(int n, int nnz, HYPRE_Real *b, HYPRE_Real *x,
                    hypre_CSRMatrix *csr, int REPEAT, bool print)
{
   int i,k,i1,i2,ii;

   int *ia = csr->i;
   int *ja = csr->j;
   HYPRE_Real *a = csr->data;

   double t1 = wall_timer();

   for (ii = 0; ii < REPEAT; ii++)
   {
      /* Forward G-S */
      for (i = 0; i < n; i++)
      {
         HYPRE_Real diag = 0.0;
         x[i] = b[i];
         i1 = ia[i];
         i2 = ia[i+1];
         for (k = i1; k < i2; k++)
         {
            if (ja[k] != i)
            {
               x[i] -= a[k] * x[ja[k]];
            }
            else
            {
               diag = a[k];
            }
         }
         if (diag != 0.0)
         {
            x[i] /= diag;
         }
      }

      /* Backward G-S */
      for (i = n-1; i >= 0; i--)
      {
         HYPRE_Real diag = 0.0;
         x[i] = b[i];
         i1 = ia[i];
         i2 = ia[i+1];
         for (k = i1; k < i2; k++)
         {
            if (ja[k] != i)
            {
               x[i] -= a[k] * x[ja[k]];
            }
            else
            {
               diag = a[k];
            }
         }
         if (diag != 0.0)
         {
            x[i] /= diag;
         }
      }
   }

   double t2 = wall_timer() - t1;
   if (print)
   {
      printf(" [CPU] G-S \n");
      printf("  time(s) = %.2e, Gflops = %5.3f\n", t2/REPEAT, REPEAT*4*((nnz)/1e9)/t2);
   }
}


__global__ void
hypreCUDAKernel_CSRMoveDiag(HYPRE_Int m, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a)
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= m)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int i, row_start, row_end;

   if (lane < 2)
   {
      i = read_only_load(d_ia + row + lane);
   }

   row_start = __shfl_sync(HYPRE_WARP_FULL_MASK, i, 0);
   row_end   = __shfl_sync(HYPRE_WARP_FULL_MASK, i, 1);

   for (i = row_start + lane; i < row_end; i += HYPRE_WARP_SIZE)
   {
      HYPRE_Int j = d_ja[i];
      if (j == row && i != row_start)
      {
         HYPRE_Complex t = d_a[i];
         d_a[i] = d_a[row_start];
         d_a[row_start] = t;
         d_ja[i] = d_ja[row_start];
         d_ja[row_start] = j;
         break;
      }
   }
}

HYPRE_Int
hypreDevice_CSRMoveDiag(HYPRE_Int m, HYPRE_Int *d_ia, HYPRE_Int *d_ja, HYPRE_Complex *d_a)
{
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(m, "warp", bDim);

   /* trivial case */
   if (m <= 0)
   {
      return hypre_error_flag;
   }

   hypreCUDAKernel_CSRMoveDiag<<<gDim, bDim>>>(m, d_ia, d_ja, d_a);

   return hypre_error_flag;
}

/*-----------------------------------------------*/
void cuda_init(HYPRE_Int argc, char **argv)
{
   HYPRE_Int deviceCount, dev;
   cudaGetDeviceCount(&deviceCount);
   printf("=========================================\n");
   if (deviceCount == 0)
   {
      printf("There is no device supporting CUDA\n");
   }
   for (dev = 0; dev < deviceCount; ++dev)
   {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      if (dev == 0)
      {
         if (deviceProp.major == 9999 && deviceProp.minor == 9999)
         {
            printf("There is no device supporting CUDA.\n");
         }
         else if (deviceCount == 1)
         {
            printf("There is 1 device supporting CUDA\n");
         }
         else
         {
            printf("There are %d devices supporting CUDA\n", deviceCount);
         }
      }
   printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
   printf("  Major revision number:          %d\n",deviceProp.major);
   printf("  Minor revision number:          %d\n",deviceProp.minor);
   printf("  Total amount of global memory:  %.2f GB\n",deviceProp.totalGlobalMem/1e9);
   }
   dev = 0;
   cudaSetDevice(dev);
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, dev);
   printf("\nRunning on Device %d: \"%s\"\n", dev, deviceProp.name);
   printf("=========================================\n");
}

/*---------------------------------------------------*/
void cuda_check_err()
{
   cudaError_t cudaerr = cudaGetLastError() ;
   if (cudaerr != cudaSuccess)
   {
       printf("error: %s\n",cudaGetErrorString(cudaerr));
   }
}

