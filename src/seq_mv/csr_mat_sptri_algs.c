/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

#ifndef CSR_MAT_SPTRI_ALGS_DEBUG
#define CSR_MAT_SPTRI_ALGS_DEBUG
#endif
#undef CSR_MAT_SPTRI_ALGS_DEBUG

enum TRIANGULAR_MATRICES { 
  L_D_U, /* L/U are purely lower/upper triangular without the diagonal diagonal */
  LD_U, /* L is lower triangular with the diagonal, U is pure upper triangular without the diagonal. Useful for forward SOR/GS where the diagonal needs to be embedded in the solve for L */  
  L_DU, /* U is upper triangular with the diagonal, L is pure lower triangular without the diagonal. Useful for backward SOR/GS where the diagonal needs to be embedded in the solve for U */  
  LD_D_DU, /* L/U are lower/upper triangular with the diagonal. Useful for SSOR/SGS where the diagonal needs to be embedded in the forward/backwards solves for L/U */  
};

__global__ void correctForDiagonalKernel(const HYPRE_Int num_rows, const HYPRE_Complex * u_data, const HYPRE_Complex * diag, HYPRE_Complex * result) {  
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row<num_rows) {
    result[row] += diag[row]*u_data[row];
  }
}

void correctForDiagonal(const int num_rows, const HYPRE_Complex * u_data, const HYPRE_Complex * diag, HYPRE_Complex * result) {  
  HYPRE_Int num_threads=128;
  int num_blocks = (num_rows + num_threads - 1)/num_threads;
  if (num_blocks) correctForDiagonalKernel<<<num_blocks,num_threads>>>(num_rows, u_data, diag, result);
  HYPRE_CUDA_CALL(cudaGetLastError());
}


//__launch_bounds__(32)
__global__ void scaleByDiagonalUnsortedKernel(const HYPRE_Int num_rows, const HYPRE_Complex * in,
					      const HYPRE_Complex * invdiag, HYPRE_Complex * out) {  
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row<num_rows) {    
    out[row] = in[row]*invdiag[row];
  }
}

void scaleByDiagonalUnsorted(const int num_rows, const HYPRE_Complex * in,
			     const HYPRE_Complex * invdiag,  HYPRE_Complex * out) {  
  HYPRE_Int num_threads=128;
  int num_blocks = (num_rows + num_threads - 1)/num_threads;
  dim3 gDim(num_blocks,1,1), bDim(num_threads,1,1);
  if (num_blocks) HYPRE_CUDA_LAUNCH(scaleByDiagonalUnsortedKernel, gDim, bDim, num_rows, in, invdiag, out);
}

__global__ void fillDiagonalUnsortedKernel(const HYPRE_Int num_rows, const HYPRE_Complex * data,
					   const HYPRE_Int * i, HYPRE_Complex * invdiag) {  
  int row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row<num_rows) {
    invdiag[row] = 1.0/data[i[row]];
  }
}

void fillDiagonalUnsorted(const int num_rows, const HYPRE_Complex * data,
			  const HYPRE_Int * i, HYPRE_Complex * invdiag) {
  HYPRE_Int num_threads=128;
  int num_blocks = (num_rows + num_threads - 1)/num_threads;
  dim3 gDim(num_blocks,1,1), bDim(num_threads,1,1);
  if (num_blocks) HYPRE_CUDA_LAUNCH(fillDiagonalUnsortedKernel, gDim, bDim, num_rows, data, i, invdiag);
}


//__launch_bounds__(32)
__global__ void spmvLKernel(const HYPRE_Int num_rows, const HYPRE_Complex * data,
			    const HYPRE_Int * i, const HYPRE_Int * j, const HYPRE_Complex * x, const HYPRE_Complex * invdiag,
			    const HYPRE_Complex * r, const HYPRE_Real omega, HYPRE_Complex * inout) {
  
  int row = blockDim.y*blockIdx.x+threadIdx.y;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend;
    if (threadIdx.x==0) {
      rbegin = i[row];
      rend = i[row+1];
    }
    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, blockDim.x);
    rend = __shfl_sync(0xffffffff, rend, 0, blockDim.x);

    HYPRE_Complex sum=0.0;
    for (int t=threadIdx.x; t<rend-rbegin; t+=blockDim.x) {
      /* read the actual column for valid threads/columns */
      HYPRE_Int column = j[rbegin+t];
      HYPRE_Complex value = data[rbegin+t];
      HYPRE_Complex lhs=x[column]*(column<row ? 1.0 : 0.0);
      sum += value*lhs;
    }
    for (int t=blockDim.x/2; t>=1; t/=2)
      sum += __shfl_xor_sync(0xffffffff, sum, t, blockDim.x);

    if (threadIdx.x==0) inout[row] += (r[row] - sum)*invdiag[row];
  }
}

void spmvL(const HYPRE_Int num_rows, const HYPRE_Int num_nonzeros, const HYPRE_Complex * data, const HYPRE_Int * i,
	   const HYPRE_Int * j, const HYPRE_Complex * x, const HYPRE_Complex * invdiag,
	   const HYPRE_Complex * r, const HYPRE_Real omega, HYPRE_Complex * inout) {  
  HYPRE_Int num_threads=128;
  float nnz_per_row = ((1.0*num_nonzeros)/num_rows);
  HYPRE_Int threads_per_row = 1;
  while(threads_per_row < nnz_per_row) threads_per_row*=2;
  if (threads_per_row>32) threads_per_row=32;
  HYPRE_Int num_rows_block = num_threads/threads_per_row;
  int num_blocks = (num_rows + num_rows_block - 1)/num_rows_block;
  dim3 gDim(num_blocks,1,1), bDim(threads_per_row,num_rows_block,1);
  HYPRE_CUDA_LAUNCH(spmvLKernel, gDim, bDim, num_rows, data, i, j, x, invdiag, r, omega, inout);
}


__global__ void spmvKernel(const HYPRE_Int num_rows, const HYPRE_Complex * data,
			   const HYPRE_Int * i, const HYPRE_Int * j, const HYPRE_Complex * x, const HYPRE_Complex * invdiag,
			   const HYPRE_Complex * r, const HYPRE_Real omega, HYPRE_Complex * inout) {
  
  int row = blockDim.y*blockIdx.x+threadIdx.y;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the sub warp */
    int rbegin, rend;
    if (threadIdx.x==0) {
      rbegin = i[row];
      rend = i[row+1];
    }
    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0, blockDim.x);
    rend = __shfl_sync(0xffffffff, rend, 0, blockDim.x);

    HYPRE_Complex sum=0.0;
    for (int t=threadIdx.x; t<rend-rbegin; t+=blockDim.x) {
      /* read the actual column for valid threads/columns */
      HYPRE_Int column = j[rbegin+t];
      HYPRE_Complex value = data[rbegin+t];
      HYPRE_Complex lhs=x[column];
      sum += value*lhs;
    }
    for (int t=blockDim.x/2; t>=1; t/=2)
      sum += __shfl_xor_sync(0xffffffff, sum, t, blockDim.x);
    
    if (threadIdx.x==0) inout[row] += ((1+omega)*r[row] - omega*sum)*invdiag[row];
  }
}

void spmv(const HYPRE_Int num_rows, const HYPRE_Int num_nonzeros, const HYPRE_Complex * data, const HYPRE_Int * i,
	   const HYPRE_Int * j, const HYPRE_Complex * x, const HYPRE_Complex * invdiag,
	  const HYPRE_Complex * r, const HYPRE_Real omega, HYPRE_Complex * inout) {  
  HYPRE_Int num_threads=128;
  float nnz_per_row = ((1.0*num_nonzeros)/num_rows);
  HYPRE_Int threads_per_row = 1;
  while(threads_per_row < nnz_per_row) threads_per_row*=2;
  if (threads_per_row>32) threads_per_row=32;
  HYPRE_Int num_rows_block = num_threads/threads_per_row;
  int num_blocks = (num_rows + num_rows_block - 1)/num_rows_block;
  dim3 gDim(num_blocks,1,1), bDim(threads_per_row,num_rows_block,1);
  HYPRE_CUDA_LAUNCH(spmvKernel, gDim, bDim, num_rows, data, i, j, x, invdiag, r, omega, inout);
}



#define THREADBLOCK_SIZE 256
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

inline __device__ HYPRE_Int scan1Inclusive(HYPRE_Int idata, volatile HYPRE_Int *shmem, cg::thread_block cta) {
    uint pos = 2 * threadIdx.x - (threadIdx.x & (THREADBLOCK_SIZE - 1));
    shmem[pos] = 0;
    pos += THREADBLOCK_SIZE;
    shmem[pos] = idata;

    for (uint offset = 1; offset < THREADBLOCK_SIZE; offset <<= 1) {
        cg::sync(cta);
        HYPRE_Int t = shmem[pos] + shmem[pos - offset];
        cg::sync(cta);
        shmem[pos] = t;
    }
    return shmem[pos];
}

inline __device__ HYPRE_Int scan1Exclusive(HYPRE_Int idata, volatile HYPRE_Int *shmem, cg::thread_block cta) {
    return scan1Inclusive(idata, shmem, cta) - idata;
}
////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(HYPRE_Int * d_Buf, HYPRE_Int * d_Dst, HYPRE_Int * d_Src, uint N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ HYPRE_Int shmem[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * THREADBLOCK_SIZE + threadIdx.x;

    //Load data
    HYPRE_Int idata = 0;
    if (pos<N) idata = d_Src[pos];

    //Calculate exclusive scan
    HYPRE_Int odata = scan1Exclusive(idata, shmem, cta);

    //Write back
    if (pos<N) d_Dst[pos] = odata;
    if (threadIdx.x==THREADBLOCK_SIZE-1 && pos<N) d_Buf[blockIdx.x] = odata+idata;
}


//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(HYPRE_Int *d_Data, HYPRE_Int *d_Buffer, uint N) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    uint pos = blockIdx.x * THREADBLOCK_SIZE + threadIdx.x;
    __shared__ HYPRE_Int buf;
    if (threadIdx.x == 0) buf = d_Buffer[blockIdx.x];
    cg::sync(cta);

    if (pos<N) {
      HYPRE_Int data = d_Data[pos];
      d_Data[pos] = data+buf;
    }
}

void exclusive_scan(HYPRE_Int *d_Dst, HYPRE_Int *d_Src, uint N, HYPRE_MemoryLocation memory_location)
{
  int nBlocks = (N + THREADBLOCK_SIZE - 1) / THREADBLOCK_SIZE;
  HYPRE_Int * d_Src_small, * d_Dst_small;
  d_Src_small = hypre_CTAlloc(HYPRE_Int, nBlocks, memory_location);
  d_Dst_small = hypre_CTAlloc(HYPRE_Int, nBlocks, memory_location);
  HYPRE_CUDA_CALL(cudaMemset(d_Src_small, 0, nBlocks * sizeof(HYPRE_Int)));
  HYPRE_CUDA_CALL(cudaMemset(d_Dst_small, 0, nBlocks * sizeof(HYPRE_Int)));
  
  /* scan the input vector */
  scanExclusiveShared<<<nBlocks, THREADBLOCK_SIZE>>>(d_Src_small, d_Dst, d_Src, N);
  HYPRE_CUDA_CALL(cudaGetLastError());  

  /* recurse and call again */
  if (nBlocks>1) exclusive_scan(d_Dst_small, d_Src_small, nBlocks, memory_location);

  if (nBlocks>1)
    uniformUpdate<<<nBlocks, THREADBLOCK_SIZE>>>(d_Dst, d_Dst_small, N);
  HYPRE_CUDA_CALL(cudaGetLastError());  

  hypre_TFree(d_Src_small, memory_location);
  hypre_TFree(d_Dst_small, memory_location);
  return;
}


template<TRIANGULAR_MATRICES TRI_MAT>
__global__ void fillLUColIndicesAndDataKernel(const HYPRE_Int num_rowsA, const HYPRE_Int num_nonzerosA,
					      const HYPRE_Int * rowsA, const HYPRE_Int * colsA, const HYPRE_Complex * dataA,
					      const HYPRE_Int num_rowsL, const HYPRE_Int num_nonzerosL,
					      const HYPRE_Int * rowsL, HYPRE_Int * colsL, HYPRE_Complex * dataL,
					      const HYPRE_Int num_rowsU, const HYPRE_Int num_nonzerosU,
					      const HYPRE_Int * rowsU, HYPRE_Int * colsU, HYPRE_Complex * dataU,
					      const HYPRE_Int * colIndexForDiagonal, HYPRE_Complex * diagonal) {

  int tid = threadIdx.x%warpSize;
  int rowSmall = threadIdx.x/warpSize;
  int row = (blockDim.x/warpSize)*blockIdx.x + rowSmall;

  if (row<num_rowsA) {
    /* read the row pointers by the first thread in the warp */
    HYPRE_Int Abegin, Aend, Lbegin, Ubegin, diag;
    if (tid==0) {
      Abegin = rowsA[row];
      Aend = rowsA[row+1];
      Lbegin = rowsL[row];
      Ubegin = rowsU[row];
      diag = colIndexForDiagonal[row];
    }
    /* broadcast across the warp */
    Abegin = __shfl_sync(0xffffffff, Abegin, 0);
    Aend = __shfl_sync(0xffffffff, Aend, 0);
    Lbegin = __shfl_sync(0xffffffff, Lbegin, 0);
    Ubegin = __shfl_sync(0xffffffff, Ubegin, 0);
    diag = __shfl_sync(0xffffffff, diag, 0);
    
    /* This compute the loop size to the next multiple of warpSize (32) */
    int roundUpSize = ((Aend-Abegin + warpSize - 1)/warpSize)*warpSize;
    
    for (int t=tid; t<roundUpSize; t+=warpSize) {
      /* read the actual column, value from A and fill either L or U */
      switch (TRI_MAT) {
      case LD_D_DU:
	{
	  if (t<diag) {
	    colsL[Lbegin+t] = colsA[Abegin+t];
	    dataL[Lbegin+t] = dataA[Abegin+t];
	  } else if (t==diag) {
	    colsL[Lbegin+t] = colsA[Abegin+t];
	    dataL[Lbegin+t] = dataA[Abegin+t];
	    colsU[Ubegin+t-(diag)] = colsA[Abegin+t];
	    dataU[Ubegin+t-(diag)] = dataA[Abegin+t];
	    diagonal[row] = dataA[Abegin+t];
	  } else if (t<Aend-Abegin) {
	    colsU[Ubegin+t-(diag)] = colsA[Abegin+t];
	    dataU[Ubegin+t-(diag)] = dataA[Abegin+t];
	  }
	}
	break;
      case L_D_U:
	{
	  if (t<diag) {
	    colsL[Lbegin+t] = colsA[Abegin+t];
	    dataL[Lbegin+t] = dataA[Abegin+t];
	  } else if (t==diag) {
	    diagonal[row] = dataA[Abegin+t];
	  } else if (t<Aend-Abegin) {
	    colsU[Ubegin+t-(diag+1)] = colsA[Abegin+t];
	    dataU[Ubegin+t-(diag+1)] = dataA[Abegin+t];
	  }
	}
	break;
      case LD_U:
	{
	  if (t<diag) {
	    colsL[Lbegin+t] = colsA[Abegin+t];
	    dataL[Lbegin+t] = dataA[Abegin+t];
	  } else if (t==diag) {
	    colsL[Lbegin+t] = colsA[Abegin+t];
	    dataL[Lbegin+t] = dataA[Abegin+t];
	  } else if (t<Aend-Abegin) {
	    colsU[Ubegin+t-(diag+1)] = colsA[Abegin+t];
	    dataU[Ubegin+t-(diag+1)] = dataA[Abegin+t];
	  }
	}
	break;
      case L_DU:
	{
	  if (t<diag) {
	    colsL[Lbegin+t] = colsA[Abegin+t];
	    dataL[Lbegin+t] = dataA[Abegin+t];
	  } else if (t==diag) {
	    colsU[Ubegin+t-(diag)] = colsA[Abegin+t];
	    dataU[Ubegin+t-(diag)] = dataA[Abegin+t];
	  } else if (t<Aend-Abegin) {
	    colsU[Ubegin+t-(diag)] = colsA[Abegin+t];
	    dataU[Ubegin+t-(diag)] = dataA[Abegin+t];
	  }
	}
	break;
      }
    }
  }
}


void fillLUColIndicesAndData(const HYPRE_Int num_rowsA, const HYPRE_Int num_nonzerosA,
			     const HYPRE_Int * rowsA, const HYPRE_Int * colsA, const HYPRE_Complex * dataA,
			     const HYPRE_Int num_rowsL, const HYPRE_Int num_nonzerosL,
			     const HYPRE_Int * rowsL, HYPRE_Int * colsL, HYPRE_Complex * dataL,
			     const HYPRE_Int num_rowsU, const HYPRE_Int num_nonzerosU,
			     const HYPRE_Int * rowsU, HYPRE_Int * colsU, HYPRE_Complex * dataU,
			     const HYPRE_Int * colIndexForDiagonal, HYPRE_Complex * diagonal,
			     TRIANGULAR_MATRICES TRI_MAT) {
  
  HYPRE_Int num_threads=128;
  HYPRE_Int warpSize = 32;
  HYPRE_Int num_rows_block = num_threads/warpSize;
  int num_blocks = (num_rowsA + num_rows_block - 1)/num_rows_block;
  if (num_blocks) {
    if (TRI_MAT==LD_D_DU)
      fillLUColIndicesAndDataKernel<LD_D_DU><<<num_blocks,num_threads>>>(num_rowsA, num_nonzerosA, rowsA, colsA, dataA,
									 num_rowsL, num_nonzerosL, rowsL, colsL, dataL,
									 num_rowsU, num_nonzerosU, rowsU, colsU, dataU,
									 colIndexForDiagonal, diagonal);
    else if (TRI_MAT==L_D_U)
      fillLUColIndicesAndDataKernel<L_D_U><<<num_blocks,num_threads>>>(num_rowsA, num_nonzerosA, rowsA, colsA, dataA,
								       num_rowsL, num_nonzerosL, rowsL, colsL, dataL,
								       num_rowsU, num_nonzerosU, rowsU, colsU, dataU,
								       colIndexForDiagonal, diagonal);
    else if (TRI_MAT==LD_U)
      fillLUColIndicesAndDataKernel<LD_U><<<num_blocks,num_threads>>>(num_rowsA, num_nonzerosA, rowsA, colsA, dataA,
								      num_rowsL, num_nonzerosL, rowsL, colsL, dataL,
								      num_rowsU, num_nonzerosU, rowsU, colsU, dataU,
								      colIndexForDiagonal, diagonal);
    else if (TRI_MAT==L_DU)
      fillLUColIndicesAndDataKernel<L_DU><<<num_blocks,num_threads>>>(num_rowsA, num_nonzerosA, rowsA, colsA, dataA,
								      num_rowsL, num_nonzerosL, rowsL, colsL, dataL,
								      num_rowsU, num_nonzerosU, rowsU, colsU, dataU,
								      colIndexForDiagonal, diagonal);
    
    HYPRE_CUDA_CALL(cudaGetLastError());
  }
}

template<TRIANGULAR_MATRICES TRI_MAT>
__global__ void constructLURowPtrsKernel(const HYPRE_Int num_rows, const HYPRE_Int * rows, const HYPRE_Int * cols,
					 HYPRE_Int * rowsL, HYPRE_Int * rowsU, HYPRE_Int * colIndexForDiagonal) {
  int tid = threadIdx.x%warpSize;
  int rowSmall = threadIdx.x/warpSize;
  int row = (blockDim.x/warpSize)*blockIdx.x + rowSmall;

  if (row<num_rows) {
    /* read the row pointers by the first thread in the warp */
    int rbegin, rend;
    if (tid==0) {
      rbegin = rows[row];
      rend = rows[row+1];
    }
    /* broadcast across the warp */
    rbegin = __shfl_sync(0xffffffff, rbegin, 0);
    rend = __shfl_sync(0xffffffff, rend, 0);

    /* This compute the loop size to the next multiple of warpSize (32) */
    int roundUpSize = ((rend-rbegin + warpSize - 1)/warpSize)*warpSize;
    
    int colIndexForDiag=0;
    for (int t=tid; t<roundUpSize; t+=warpSize) {
      /* make a value for large threads that is guaranteed to be bigger than all others */
      int column = 2*num_rows;

      /* read the actual column for valid threads/columns */
      if (t<rend-rbegin) column = cols[rbegin+t];
      /* make it absolute value so we can search for 0 */
      int val = abs(column-row);

      /* Try to find the location of the diagonal */
      colIndexForDiag = t;
      for (int offset = warpSize/2; offset > 0; offset/=2) {
      	int tmp1 = __shfl_down_sync(0xffffffff, val, offset);
      	int tmp2 = __shfl_down_sync(0xffffffff, colIndexForDiag, offset);
      	if (tmp1 < val) {
      	  val = tmp1;
      	  colIndexForDiag = tmp2;
      	}
      }
      /* broadcast in order to exit successfully for all threads in the warp */
      val = __shfl_sync(0xffffffff, val, 0);
      if (val==0) break;
    }

    if (tid==0) {
      switch (TRI_MAT) {
      case LD_D_DU:
	{
	  rowsL[row]=colIndexForDiag+1;
	  rowsU[row]=rend-rbegin-colIndexForDiag;
	  colIndexForDiagonal[row] = colIndexForDiag;
	}
	break;
      case L_D_U:
	{
	  rowsL[row]=colIndexForDiag;
	  rowsU[row]=rend-rbegin-colIndexForDiag-1;
	  colIndexForDiagonal[row] = colIndexForDiag;
	}
	break;
      case LD_U:
	{
	  rowsL[row]=colIndexForDiag+1;
	  rowsU[row]=rend-rbegin-colIndexForDiag-1;
	  colIndexForDiagonal[row] = colIndexForDiag;
	}
	break;
      case L_DU:
	{
	  rowsL[row]=colIndexForDiag;
	  rowsU[row]=rend-rbegin-colIndexForDiag;
	  colIndexForDiagonal[row] = colIndexForDiag;
	}
	break;
      }
    }
  }
}



void constructLURowPtrs(const int num_rows, const HYPRE_Int * rows, const HYPRE_Int * cols,
			HYPRE_Int * rowsL, HYPRE_Int * rowsU, HYPRE_Int * colIndexForDiagonal,
			HYPRE_Int &nnzLower, HYPRE_Int &nnzUpper, TRIANGULAR_MATRICES TRI_MAT,
			HYPRE_MemoryLocation memory_location) {
  
  HYPRE_Int num_threads=128;
  HYPRE_Int warpSize = 32;
  HYPRE_Int num_rows_block = num_threads/warpSize;
  int num_blocks = (num_rows + num_rows_block - 1)/num_rows_block;
  HYPRE_Int rank;
  hypre_MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (num_blocks) {
    if (TRI_MAT==LD_D_DU)
      constructLURowPtrsKernel<LD_D_DU><<<num_blocks,num_threads>>>(num_rows, rows, cols, rowsL, rowsU, colIndexForDiagonal);
    else if (TRI_MAT==L_D_U)
      constructLURowPtrsKernel<L_D_U><<<num_blocks,num_threads>>>(num_rows, rows, cols, rowsL, rowsU, colIndexForDiagonal);
    else if (TRI_MAT==LD_U)
      constructLURowPtrsKernel<LD_U><<<num_blocks,num_threads>>>(num_rows, rows, cols, rowsL, rowsU, colIndexForDiagonal);
    else if (TRI_MAT==L_DU)
      constructLURowPtrsKernel<L_DU><<<num_blocks,num_threads>>>(num_rows, rows, cols, rowsL, rowsU, colIndexForDiagonal);
    HYPRE_CUDA_CALL(cudaGetLastError());
  }

  /* The following code is necessary because thrust exclusive_scan seems broken and I haven't
     written my own prefix_scan yet. Lame */
  HYPRE_Int * dst;
  dst = hypre_CTAlloc(HYPRE_Int, num_rows+1, memory_location);
  
  HYPRE_CUDA_CALL(cudaMemset(dst,0,(num_rows+1)*sizeof(HYPRE_Int)));
  exclusive_scan(dst, rowsL, num_rows+1, memory_location);
  hypre_TMemcpy(rowsL, dst, HYPRE_Int, num_rows+1, memory_location, memory_location);

  HYPRE_CUDA_CALL(cudaMemset(dst,0,(num_rows+1)*sizeof(HYPRE_Int)));
  exclusive_scan(dst, rowsU, num_rows+1, memory_location);
  hypre_TMemcpy(rowsU, dst, HYPRE_Int, num_rows+1, memory_location, memory_location);

  hypre_TFree(dst, memory_location);

  hypre_TMemcpy(&nnzLower, rowsL+num_rows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);
  hypre_TMemcpy(&nnzUpper, rowsU+num_rows, HYPRE_Int, 1, HYPRE_MEMORY_HOST, memory_location);
}


HYPRE_Int
hypre_CSRMatrixPrintMemoryUsage ( const char * FILENAME, const char * FUNCTIONNAME, HYPRE_Int LINENUMBER, hypre_CSRMatrix * matrix ) {
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  HYPRE_Int rank;
  hypre_MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank==0)
    printf("\t\trank=%d matrix=%p : %s %s %d : free=%1.8g GBs, totalMem=%1.8g GBs\n",
	   rank,matrix,FILENAME,FUNCTIONNAME,LINENUMBER,freeMem/1.e9,totalMem/1.e9);
  return 0;
}


HYPRE_Int
hypre_CSRMatrixDestroyTriMats ( hypre_CSRMatrix *matrix ) {
#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__,matrix);
#endif

  HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

  if (hypre_CSRMatrixLower(matrix)) {
    hypre_CSRMatrixDestroy((hypre_CSRMatrix *)hypre_CSRMatrixLower(matrix));
    hypre_CSRMatrixLower(matrix) = NULL;
  }

  if (hypre_CSRMatrixUpper(matrix)) {
    hypre_CSRMatrixDestroy((hypre_CSRMatrix *)hypre_CSRMatrixUpper(matrix));
    hypre_CSRMatrixUpper(matrix) = NULL;
  }

  /* free the diagonal */
  if (hypre_CSRMatrixDiagonal(matrix)) {
    hypre_TFree(hypre_CSRMatrixDiagonal(matrix), memory_location);
    hypre_CSRMatrixDiagonal(matrix) = NULL;
  }

  /* free the work vector */
  if (hypre_CSRMatrixWorkVector(matrix)) {
    hypre_TFree(hypre_CSRMatrixWorkVector(matrix), memory_location);
    hypre_CSRMatrixWorkVector(matrix) = NULL;
  }

  /* free the work vector2 */
  if (hypre_CSRMatrixWorkVector2(matrix)) {
    hypre_TFree(hypre_CSRMatrixWorkVector2(matrix), memory_location);
    hypre_CSRMatrixWorkVector2(matrix) = NULL;
  }
  return 0;
}


HYPRE_Int
hypre_CSRMatrixDestroyTriMatsSolveDataDevice( hypre_CSRMatrix *matrix )
{
   if (hypre_CSRMatrixCusparseDataLower(matrix)) {
#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
      printf("\t%s %s %d : matrix=%p\n",__FILE__,__FUNCTION__,__LINE__,matrix);
#endif

      HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

      if (hypre_CSRMatrixCusparseDataLower(matrix)) {
         hypre_CudaSpTriMatrixDataDestroy((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(matrix), memory_location);
         hypre_CSRMatrixCusparseDataLower(matrix) = NULL;
      }
      if (hypre_CSRMatrixCusparseDataUpper(matrix)) {
         hypre_CudaSpTriMatrixDataDestroy((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(matrix), memory_location);
         hypre_CSRMatrixCusparseDataUpper(matrix) = NULL;
      }
   }
   return 0;
}


/*****************************************************************************************************/
/* This functions sorts values and column indices in each row in ascending order                     */
/*****************************************************************************************************/
HYPRE_Int
hypre_CSRMatrixSortDeviceMatrixOutOfPlace( HYPRE_Int n, HYPRE_Int nnz, HYPRE_Int * rows,
					   HYPRE_Int * cols, HYPRE_Complex * data,
					   HYPRE_MemoryLocation memory_location ) {
  csru2csrInfo_t info;
  size_t pBufferSizeInBytes=0;
  void * pBuffer = NULL;
  cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
  cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());
  HYPRE_CUSPARSE_CALL(cusparseCreateCsru2csrInfo(&info));
  HYPRE_CUSPARSE_CALL(cusparseDcsru2csr_bufferSizeExt(handle, n, n, nnz, data, rows, cols, info, &pBufferSizeInBytes));
  pBuffer = hypre_CTAlloc(char, pBufferSizeInBytes, memory_location);
  HYPRE_CUSPARSE_CALL(cusparseDcsru2csr(handle, n, n, nnz, descr, data, rows, cols, info, pBuffer));
  hypre_TFree(pBuffer, memory_location);
  HYPRE_CUSPARSE_CALL(cusparseDestroyCsru2csrInfo(info));
  return 0;
}


/************************************************************************************************************/
/* This function creates a pair of triangular matrices whose contents depend on the input parameter TRI_MAT */
/*   TRI_MAT = L_D_U : L/U are purely lower/upper triangular without the diagonal                           */
/*   TRI_MAT = LD_U : L is lower triangular with the diagonal, U is pure upper triangular without the       */
/*     diagonal. Useful for forward SOR/GS where the diagonal needs to be embedded in the solve for L.      */
/*   TRI_MAT = L_DU : U is upper triangular with the diagonal, L is pure lower triangular without the       */
/*     diagonal. Useful for backward SOR/GS where the diagonal needs to be embedded in the solve for U.     */
/*   TRI_MAT = LD_D_DU : L/U are lower/upper triangular with the diagonal. Useful for SSOR/SGS where the    */
/*     diagonal needs to be embedded in the forward/backwards solves for L/U                                */  
/************************************************************************************************************/
HYPRE_Int 
hypre_CSRMatrixCreateTriMatsDevice ( hypre_CSRMatrix *matrix, TRIANGULAR_MATRICES TRI_MAT,
				     HYPRE_Int wv_size, HYPRE_Int wv_size2) {

  if (hypre_CSRMatrixRebuildTriMats(matrix)==0) return 0;

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  printf("\t%s %s %d : matrix=%p\n",__FILE__,__FUNCTION__,__LINE__,matrix);
#endif

  /* Attempt to free a previous version if already built */
  hypre_CSRMatrixDestroyTriMats(matrix);

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  /* Print the memory usage */
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__);

  cudaEvent_t _start, _stop;
  HYPRE_CUDA_CALL( cudaEventCreate(&_start) );
  HYPRE_CUDA_CALL( cudaEventCreate(&_stop) );
  HYPRE_CUDA_CALL( cudaEventRecord(_start) );
#endif

  HYPRE_Int n       = hypre_CSRMatrixNumRows(matrix);
  HYPRE_Int nnz     = hypre_CSRMatrixNumNonzeros(matrix);

  HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

  /* Useful array for knowing where the diagonal element is in the row */
  HYPRE_Int * colIndexForDiagonal = hypre_CTAlloc(HYPRE_Int, n, memory_location);
  HYPRE_CUDA_CALL(cudaMemset(colIndexForDiagonal, 0, n*sizeof(HYPRE_Int)));
  
  /* Create the matrix */
  hypre_CSRMatrix * Lower = (hypre_CSRMatrix *) hypre_CSRMatrixLower(matrix);
  hypre_CSRMatrix * Upper = (hypre_CSRMatrix *) hypre_CSRMatrixUpper(matrix);

  Lower = (hypre_CSRMatrix *) hypre_CSRMatrixCreate(n, n, 0);
  Upper = (hypre_CSRMatrix *) hypre_CSRMatrixCreate(n, n, 0);

  /* In this initialization, only the row pointers array will be constructed for Lower and Upper */
  /* Then, we'll call it again after nnzLower and nnzUpper are computed */
  hypre_CSRMatrixInitialize(Lower);
  hypre_CSRMatrixInitialize(Upper);

  /************************************************************************************************/
  /* It would be nice to sort in place however, I'm not sure if this will break things downstream */
  /************************************************************************************************/

  /* allocate and copy the matrix */
  HYPRE_Int * rows_sorted = hypre_CTAlloc(HYPRE_Int, n+1, memory_location);
  HYPRE_Int * cols_sorted = hypre_CTAlloc(HYPRE_Int, nnz, memory_location);
  HYPRE_Complex * data_sorted = hypre_CTAlloc(HYPRE_Complex, nnz, memory_location);
  hypre_TMemcpy(rows_sorted, hypre_CSRMatrixI(matrix), HYPRE_Int, n+1, memory_location, memory_location);
  hypre_TMemcpy(cols_sorted, hypre_CSRMatrixJ(matrix), HYPRE_Int, nnz, memory_location, memory_location);
  hypre_TMemcpy(data_sorted, hypre_CSRMatrixData(matrix), HYPRE_Complex, nnz, memory_location, memory_location);

  /* Ensure the CSR Matrix is sorted */
  hypre_CSRMatrixSortDeviceMatrixOutOfPlace( n, nnz, rows_sorted, cols_sorted, data_sorted, memory_location );

  /* Construct the row pointer arrays */
  constructLURowPtrs(n, rows_sorted, cols_sorted, Lower->i, Upper->i, colIndexForDiagonal, 
		     Lower->num_nonzeros, Upper->num_nonzeros, TRI_MAT, memory_location);

  /* re initialize the matrix ... this time the colum indices and values arrays are allocated */
  hypre_CSRMatrixInitialize(Lower);
  hypre_CSRMatrixInitialize(Upper);

  /* allocate the diagonal */
  if (!hypre_CSRMatrixDiagonal(matrix) && (TRI_MAT==LD_D_DU || TRI_MAT==L_D_U))
    hypre_CSRMatrixDiagonal(matrix) = hypre_CTAlloc(HYPRE_Complex, n, memory_location);

  /* Fill the column indices and values */
  fillLUColIndicesAndData(n, nnz, rows_sorted, cols_sorted, data_sorted,
			  n, Lower->num_nonzeros, Lower->i, Lower->j, Lower->data,
			  n, Upper->num_nonzeros, Upper->i, Upper->j, Upper->data,
			  colIndexForDiagonal, hypre_CSRMatrixDiagonal(matrix), TRI_MAT);
  
  /* free sorted arrays */
  hypre_TFree(rows_sorted, memory_location);
  hypre_TFree(cols_sorted, memory_location);
  hypre_TFree(data_sorted, memory_location);

  /* free this temporary */
  hypre_TFree(colIndexForDiagonal, memory_location);
  
  /* This seems to be necessary */
  hypre_CSRMatrixLower(matrix) = Lower;
  hypre_CSRMatrixUpper(matrix) = Upper;


  HYPRE_Complex *workvector = hypre_CSRMatrixWorkVector(matrix);
  HYPRE_Complex *workvector2 = hypre_CSRMatrixWorkVector2(matrix);

  /* allocate work  */
  if (!workvector) {
    workvector = hypre_CTAlloc(HYPRE_Complex, wv_size, memory_location);
    hypre_CSRMatrixWorkVector(matrix) = workvector;
  }
  if (!workvector2) {
    workvector2 = hypre_CTAlloc(HYPRE_Complex, wv_size2, memory_location);
    hypre_CSRMatrixWorkVector2(matrix) = workvector2;
  }

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  float ms=0;
  HYPRE_CUDA_CALL( cudaEventRecord(_stop) );
  HYPRE_CUDA_CALL( cudaEventSynchronize(_stop) );
  HYPRE_CUDA_CALL( cudaEventElapsedTime(&ms, _start, _stop) );
  HYPRE_CUDA_CALL( cudaEventDestroy(_start) );
  HYPRE_CUDA_CALL( cudaEventDestroy(_stop) );

  printf("\t%s %s %d : dt=%1.5f seconds\n",__FILE__,__FUNCTION__,__LINE__,ms/1.e3);
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__);
#endif

  /* everything is built for this version of the matrix so set this flag to 0 */
  hypre_CSRMatrixRebuildTriMats(matrix)=0;
  return 0;
}


/************************************************************************************************************/
/* This function creates solve info for the lower/upper triangular matrices. Diagonal MUST be embedded for  */
/* this to work properly.                                                                                   */
/************************************************************************************************************/
HYPRE_Int 
hypre_CSRMatrixCreateTriMatsSolveData ( hypre_CSRMatrix *matrix, HYPRE_Int buildLower, HYPRE_Int buildUpper) {

  if (hypre_CSRMatrixRebuildTriSolves(matrix)==0) return 0;

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  printf("\t%s %s %d : matrix=%p\n",__FILE__,__FUNCTION__,__LINE__,matrix);
#endif

  /* call the method in csr_mat_sptrisolve_device */
  hypre_CSRMatrixDestroyTriMatsSolveDataDevice(matrix);

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  /* print the memory usage */
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__);

  cudaEvent_t _start, _stop;
  HYPRE_CUDA_CALL( cudaEventCreate(&_start) );
  HYPRE_CUDA_CALL( cudaEventCreate(&_stop) );
  HYPRE_CUDA_CALL( cudaEventRecord(_start) );
#endif

  HYPRE_Int n       = hypre_CSRMatrixNumRows(matrix);

  HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);
    
  /* Create the cusparse data mats */
  if (buildLower) {
    hypre_CSRMatrixCusparseDataLower(matrix) = hypre_CudaSpTriMatrixDataCreate();

    /* mat vec with the upper part of the digonal matrix */
    hypre_CSRMatrix * Lower = (hypre_CSRMatrix *) hypre_CSRMatrixLower(matrix);
    csrsv2Info_t lowerSolveInfo = hypre_CudaSpTriMatrixDataSolveInfo((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(matrix));
    cusparseMatDescr_t lowerDescr = hypre_CudaSpTriMatrixDataMatDescr((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(matrix));
 
    /* Create need CUSPARSE data structures for LU operations */
    HYPRE_CUSPARSE_CALL( cusparseSetMatType(lowerDescr, CUSPARSE_MATRIX_TYPE_GENERAL) );
    HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(lowerDescr, CUSPARSE_INDEX_BASE_ZERO) );
    HYPRE_CUSPARSE_CALL( cusparseSetMatDiagType(lowerDescr, CUSPARSE_DIAG_TYPE_NON_UNIT) );	 
    HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(lowerDescr, CUSPARSE_FILL_MODE_LOWER) );

    cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    
    /* allocate the buffer for the analysis of L */
    int bufferSize = 0;  
    HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						   n, Lower->num_nonzeros, lowerDescr, Lower->data,
						   Lower->i, Lower->j, lowerSolveInfo, &bufferSize));
    void * work_buffer = hypre_CTAlloc(char, bufferSize, memory_location);
    
    /* do the analysis of L */
    HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						 n, Lower->num_nonzeros, lowerDescr, Lower->data,
						 Lower->i, Lower->j, lowerSolveInfo,
						 CUSPARSE_SOLVE_POLICY_USE_LEVEL, work_buffer));
    hypre_CudaSpTriMatrixDataWorkBuffer((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(matrix)) = work_buffer;
  }

  if (buildUpper) {
    hypre_CSRMatrixCusparseDataUpper(matrix) = hypre_CudaSpTriMatrixDataCreate();

    /* mat vec with the upper part of the digonal matrix */
    hypre_CSRMatrix * Upper = (hypre_CSRMatrix *) hypre_CSRMatrixUpper(matrix);
    csrsv2Info_t upperSolveInfo = hypre_CudaSpTriMatrixDataSolveInfo((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(matrix));
    cusparseMatDescr_t upperDescr = hypre_CudaSpTriMatrixDataMatDescr((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(matrix));
 
    /* Create need CUSPARSE data structures for LU operations */
    HYPRE_CUSPARSE_CALL( cusparseSetMatType(upperDescr, CUSPARSE_MATRIX_TYPE_GENERAL) );
    HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(upperDescr, CUSPARSE_INDEX_BASE_ZERO) );
    HYPRE_CUSPARSE_CALL( cusparseSetMatDiagType(upperDescr, CUSPARSE_DIAG_TYPE_NON_UNIT) );	 
    HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(upperDescr, CUSPARSE_FILL_MODE_UPPER) );

    cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    
    /* allocate the buffer for the analysis of L */
    int bufferSize = 0;  
    HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						   n, Upper->num_nonzeros, upperDescr, Upper->data,
						   Upper->i, Upper->j, upperSolveInfo, &bufferSize));
    void * work_buffer = hypre_CTAlloc(char, bufferSize, memory_location);
    
    /* do the analysis of L */
    HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						 n, Upper->num_nonzeros, upperDescr, Upper->data,
						 Upper->i, Upper->j, upperSolveInfo,
						 CUSPARSE_SOLVE_POLICY_USE_LEVEL, work_buffer));
    hypre_CudaSpTriMatrixDataWorkBuffer((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(matrix)) = work_buffer; 
  }

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  float ms=0;
  HYPRE_CUDA_CALL( cudaEventRecord(_stop) );
  HYPRE_CUDA_CALL( cudaEventSynchronize(_stop) );
  HYPRE_CUDA_CALL( cudaEventElapsedTime(&ms, _start, _stop) );
  HYPRE_CUDA_CALL( cudaEventDestroy(_start) );
  HYPRE_CUDA_CALL( cudaEventDestroy(_stop) );

  printf("\t%s %s %d : dt=%1.5f seconds\n",__FILE__,__FUNCTION__,__LINE__,ms/1.e3);
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__);
#endif

  /* everything is built for this version of the matrix so set this flag to 0 */
  hypre_CSRMatrixRebuildTriSolves(matrix)=0;

  return 0;
}

HYPRE_Int
hypre_CSRMatrixGaussSeidelDevice ( HYPRE_Real     *Vext_data, hypre_Vector   *f, hypre_Vector   *u,
				   hypre_CSRMatrix *diag, hypre_CSRMatrix *offd ) {

  HYPRE_Real     *u_data  = hypre_VectorData(u);
  HYPRE_Real     *f_data  = hypre_VectorData(f);

  HYPRE_Int num_rows_diag = hypre_CSRMatrixNumRows(diag);
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);

  /* Split diagonal matrix into L and U */
  hypre_CSRMatrixCreateTriMatsDevice(diag, LD_U, num_rows_diag, num_cols_offd);
  hypre_CSRMatrixCreateTriMatsSolveData(diag, 1, 0);
  
  HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(diag);
  
  HYPRE_Complex *workvector = hypre_CSRMatrixWorkVector(diag);
  HYPRE_Complex *workvector2 = hypre_CSRMatrixWorkVector2(diag);
  void * diag_solveworkbuffer = hypre_CudaSpTriMatrixDataWorkBuffer((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(diag));
		 
  hypre_TMemcpy(workvector, f_data, HYPRE_Complex, num_rows_diag, memory_location, hypre_VectorMemoryLocation(f));
  hypre_TMemcpy(workvector2, Vext_data, HYPRE_Complex, num_cols_offd, memory_location, HYPRE_MEMORY_HOST);  
    
  hypre_CSRMatrix * Lower = (hypre_CSRMatrix *) hypre_CSRMatrixLower(diag);
  hypre_CSRMatrix * Upper = (hypre_CSRMatrix *) hypre_CSRMatrixUpper(diag);
  cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
  cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());
  HYPRE_Complex alpha=-1.0, beta=1.0;
  
  /* mat vec with the upper part of the digonal matrix */
  csrsv2Info_t lowerSolveInfo = hypre_CudaSpTriMatrixDataSolveInfo((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(diag));
  cusparseMatDescr_t lowerDescr = hypre_CudaSpTriMatrixDataMatDescr((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(diag));

  /* Do the forward operation : Ly = b - Ux */

  /* Firstm compute b - Ux */
  HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				      Upper->num_rows, Upper->num_cols, Upper->num_nonzeros,
				      &alpha, descr, Upper->data, Upper->i, Upper->j,
				      u_data, &beta, workvector) );

  /* mat vec with the off digaonal matrix */
  if (num_cols_offd)
    HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  					offd->num_rows, offd->num_cols, offd->num_nonzeros,
  					&alpha, descr, offd->data, offd->i, offd->j,
  					workvector2, &beta, workvector) );
  
  /* solve with the lower part of the diagonal matrix. The diagonal vector is built in */
  HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					     Lower->num_rows, Lower->num_nonzeros,
					     &beta, lowerDescr, Lower->data,
					     Lower->i, Lower->j, lowerSolveInfo,
					     workvector, u_data, 
					     CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     diag_solveworkbuffer) );

  return 0;
}

HYPRE_Int
hypre_CSRMatrixSymmetricGaussSeidelDevice ( HYPRE_Real     *Vext_data, hypre_Vector   *f, hypre_Vector   *u,
					    hypre_CSRMatrix *diag, hypre_CSRMatrix *offd ) {

  HYPRE_Real     *u_data  = hypre_VectorData(u);
  HYPRE_Real     *f_data  = hypre_VectorData(f);

  HYPRE_Int num_rows_diag = hypre_CSRMatrixNumRows(diag);
  HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(offd);

  /* Split diagonal matrix into L and U */
  hypre_CSRMatrixCreateTriMatsDevice(diag, LD_D_DU, num_rows_diag, num_cols_offd);
  hypre_CSRMatrixCreateTriMatsSolveData(diag, 1, 1);

  HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(diag);
  
  HYPRE_Complex *workvector = hypre_CSRMatrixWorkVector(diag);
  HYPRE_Complex *workvector2 = hypre_CSRMatrixWorkVector2(diag);
  void * diag_solveworkbuffer_lower = hypre_CudaSpTriMatrixDataWorkBuffer((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(diag));
  void * diag_solveworkbuffer_upper = hypre_CudaSpTriMatrixDataWorkBuffer((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(diag));

  hypre_TMemcpy(workvector, f_data, HYPRE_Complex, num_rows_diag, memory_location, hypre_VectorMemoryLocation(f));
  hypre_TMemcpy(workvector2, Vext_data, HYPRE_Complex, num_cols_offd, memory_location, HYPRE_MEMORY_HOST);  
    
  hypre_CSRMatrix * Lower = (hypre_CSRMatrix *) hypre_CSRMatrixLower(diag);
  hypre_CSRMatrix * Upper = (hypre_CSRMatrix *) hypre_CSRMatrixUpper(diag);
  cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
  cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());
  HYPRE_Complex alpha=-1.0, beta=1.0;
  
  /* mat vec with the upper part of the digonal matrix */
  csrsv2Info_t lowerSolveInfo = hypre_CudaSpTriMatrixDataSolveInfo((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(diag));
  csrsv2Info_t upperSolveInfo = hypre_CudaSpTriMatrixDataSolveInfo((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(diag));
  cusparseMatDescr_t lowerDescr = hypre_CudaSpTriMatrixDataMatDescr((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataLower(diag));
  cusparseMatDescr_t upperDescr = hypre_CudaSpTriMatrixDataMatDescr((hypre_CudaSpTriMatrixData *)hypre_CSRMatrixCusparseDataUpper(diag));

  /* Do the forward operation : Ly = b - Ux */

  /* First, compute b - Ux. U includes the diagonal so we need to readd that contribution afterwards */
  HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				      Upper->num_rows, Upper->num_cols, Upper->num_nonzeros,
				      &alpha, upperDescr, Upper->data, Upper->i, Upper->j,
				      u_data, &beta, workvector) );

  /* Need to subtract out the diagonal matrix diagonal times u_data because L/U have the diagonal built in for the solves */
  correctForDiagonal(Upper->num_rows, u_data, hypre_CSRMatrixDiagonal(diag), workvector);

  /* mat vec with the off digaonal matrix */
  if (num_cols_offd)
    HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  					offd->num_rows, offd->num_cols, offd->num_nonzeros,
  					&alpha, descr, offd->data, offd->i, offd->j,
  					workvector2, &beta, workvector) );
  
  /* solve with the lower part of the diagonal matrix. The diagonal vector is built in */
  HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					     Lower->num_rows, Lower->num_nonzeros,
					     &beta, lowerDescr, Lower->data,
					     Lower->i, Lower->j, lowerSolveInfo,
					     workvector, u_data, 
					     CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     diag_solveworkbuffer_lower) );


  /* Do the reverse operation : Uy = b - Lx */

  /* First, compute b - Lx. L includes the diagonal so we need to readd that contribution afterwards */
  HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				      Lower->num_rows, Lower->num_cols, Lower->num_nonzeros,
				      &alpha, lowerDescr, Lower->data, Lower->i, Lower->j,
				      u_data, &beta, workvector) );

  /* Need to subtract out the diagonal matrix diagonal times u_data because L/U have the diagonal built in for the solves */
  correctForDiagonal(Lower->num_rows, u_data, hypre_CSRMatrixDiagonal(diag), workvector);

  /* mat vec with the off digaonal matrix */
  if (num_cols_offd)
    HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
  					offd->num_rows, offd->num_cols, offd->num_nonzeros,
  					&alpha, descr, offd->data, offd->i, offd->j,
  					workvector2, &beta, workvector) );
  
  /* solve with the upper part of the diagonal matrix. The diagonal vector is built in */
  HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					     Upper->num_rows, Upper->num_nonzeros,
					     &beta, upperDescr, Upper->data,
					     Upper->i, Upper->j, upperSolveInfo,
					     workvector, u_data, 
					     CUSPARSE_SOLVE_POLICY_USE_LEVEL,
					     diag_solveworkbuffer_upper) );
  return 0;
}

HYPRE_Int
hypre_CSRMatrixTwoStageGaussSeidelDevice (  hypre_Vector   *r, hypre_Vector   *u, hypre_CSRMatrix *diag, hypre_CSRMatrix *offd, HYPRE_Real omega, HYPRE_Int choice) {

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  /* Print the memory usage */
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__);

  cudaEvent_t _start, _stop;
  HYPRE_CUDA_CALL( cudaEventCreate(&_start) );
  HYPRE_CUDA_CALL( cudaEventCreate(&_stop) );
  HYPRE_CUDA_CALL( cudaEventRecord(_start) );
#endif

  HYPRE_Real     *u_data  = hypre_VectorData(u);
  HYPRE_Real     *r_data  = hypre_VectorData(r);

  HYPRE_Int num_rows_diag = hypre_CSRMatrixNumRows(diag);
  HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(diag);
  HYPRE_Complex *workvector = hypre_CSRMatrixWorkVector(diag);
  HYPRE_Complex *workvector2 = hypre_CSRMatrixWorkVector2(diag);
  HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(diag);

  if (hypre_CSRMatrixRebuildTriMats(diag)==1) {
    /* free the work vector */
    if (workvector) {
      hypre_TFree(workvector, memory_location);
      hypre_CSRMatrixWorkVector(diag) = NULL;
    }
    workvector = hypre_CTAlloc(HYPRE_Complex, num_rows_diag, memory_location);
    hypre_CSRMatrixWorkVector(diag) = workvector;

    /* free the work vector */
    if (workvector2) {
      hypre_TFree(workvector2, memory_location);
      hypre_CSRMatrixWorkVector2(diag) = NULL;
    }
    workvector2 = hypre_CTAlloc(HYPRE_Complex, num_rows_diag, memory_location);
    hypre_CSRMatrixWorkVector2(diag) = workvector2;
    fillDiagonalUnsorted(num_rows_diag, hypre_CSRMatrixData(diag), hypre_CSRMatrixI(diag), workvector2);
  }

  /* Need to subtract out the diagonal matrix diagonal times u_data because L/U have the diagonal built in for the solves */
  scaleByDiagonalUnsorted(num_rows_diag, r_data, workvector2, workvector);
  if (choice==0) {
    /* spmv with the full matrix */
    spmv(num_rows_diag, nnz_diag, hypre_CSRMatrixData(diag), hypre_CSRMatrixI(diag),
	 hypre_CSRMatrixJ(diag), workvector, workvector2, r_data, omega, u_data);
  } else if (choice==1) {
    /* spmv with L */
    spmvL(num_rows_diag, nnz_diag, hypre_CSRMatrixData(diag), hypre_CSRMatrixI(diag),
	  hypre_CSRMatrixJ(diag), workvector, workvector2, r_data, omega, u_data);
  }

  /* set this flag so work vectors are NOT reallocated */
  hypre_CSRMatrixRebuildTriMats(diag)=0;

#if defined(CSR_MAT_SPTRI_ALGS_DEBUG)
  float ms=0;
  HYPRE_CUDA_CALL( cudaEventRecord(_stop) );
  HYPRE_CUDA_CALL( cudaEventSynchronize(_stop) );
  HYPRE_CUDA_CALL( cudaEventElapsedTime(&ms, _start, _stop) );
  HYPRE_CUDA_CALL( cudaEventDestroy(_start) );
  HYPRE_CUDA_CALL( cudaEventDestroy(_stop) );

  printf("\t%s %s %d : dt=%1.5f seconds\n",__FILE__,__FUNCTION__,__LINE__,ms/1.e3);
  hypre_CSRMatrixPrintMemoryUsage(__FILE__,__FUNCTION__,__LINE__);
#endif

  return 0;
}

#endif
