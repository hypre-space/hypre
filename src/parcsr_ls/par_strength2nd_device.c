/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "HYPRE.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_IJ_mv.h"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "HYPRE_utilities.h"

#if defined(HYPRE_USING_CUDA)

#if 1
//-----------------------------------------------------------------------
HYPRE_Int
hypre_BoomerAMGCreate2ndSDevice( hypre_ParCSRMatrix  *S,
                                 HYPRE_Int           *CF_marker_host,
                                 HYPRE_Int            num_paths,
                                 HYPRE_BigInt        *coarse_row_starts,
                                 hypre_ParCSRMatrix **S2_ptr)
{
   HYPRE_Int           S_nr_local = hypre_ParCSRMatrixNumRows(S);
   hypre_CSRMatrix    *S_diag     = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix    *S_offd     = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int           S_diag_nnz = hypre_CSRMatrixNumNonzeros(S_diag);
   HYPRE_Int           S_offd_nnz = hypre_CSRMatrixNumNonzeros(S_offd);
   hypre_ParCSRMatrix *SI         = hypre_CTAlloc(hypre_ParCSRMatrix, 1, HYPRE_MEMORY_HOST);
   hypre_CSRMatrix    *Id, *SI_diag;
   hypre_ParCSRMatrix *S_XC, *S_CX, *S2;
   HYPRE_Int          *CF_marker, *new_end;
   HYPRE_Complex       coeff = 2.0;

   /*
   MPI_Comm comm = hypre_ParCSRMatrixComm(S);
   HYPRE_Int num_proc, myid;
   hypre_MPI_Comm_size(comm, &num_proc);
   hypre_MPI_Comm_rank(comm, &myid);
   */

   CF_marker = hypre_TAlloc(HYPRE_Int, S_nr_local, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(CF_marker, CF_marker_host, HYPRE_Int, S_nr_local, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* 1. Create new matrix with added diagonal */
   hypre_NvtxPushRangeColor("Setup", 1);

   /* give S data arrays */
   hypre_CSRMatrixData(S_diag) = hypre_TAlloc(HYPRE_Complex, S_diag_nnz, HYPRE_MEMORY_DEVICE );
   HYPRE_THRUST_CALL( fill,
                      hypre_CSRMatrixData(S_diag),
                      hypre_CSRMatrixData(S_diag) + S_diag_nnz,
                      1.0 );

   hypre_CSRMatrixData(S_offd) = hypre_TAlloc(HYPRE_Complex, S_offd_nnz, HYPRE_MEMORY_DEVICE );
   HYPRE_THRUST_CALL( fill,
                      hypre_CSRMatrixData(S_offd),
                      hypre_CSRMatrixData(S_offd) + S_offd_nnz,
                      1.0 );

   hypre_MatvecCommPkgCreate(S);

   /* S(C, :) and S(:, C) */
   hypre_ParCSRMatrixGenerate1DCFDevice(S, CF_marker_host, coarse_row_starts, NULL, &S_CX, &S_XC);

   hypre_assert(S_nr_local == hypre_ParCSRMatrixNumCols(S_CX));

   /* add coeff*I to S_CX */
   Id = hypre_CSRMatrixCreate( hypre_ParCSRMatrixNumRows(S_CX),
                               hypre_ParCSRMatrixNumCols(S_CX),
                               hypre_ParCSRMatrixNumRows(S_CX) );

   hypre_CSRMatrixInitialize_v2(Id, 0, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( sequence,
                      hypre_CSRMatrixI(Id),
                      hypre_CSRMatrixI(Id) + hypre_ParCSRMatrixNumRows(S_CX) + 1,
                      0  );

   new_end = HYPRE_THRUST_CALL( copy_if,
                                thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(hypre_ParCSRMatrixNumCols(S_CX)),
                                CF_marker,
                                hypre_CSRMatrixJ(Id),
                                is_nonnegative<HYPRE_Int>()  );

   hypre_assert(new_end - hypre_CSRMatrixJ(Id) == hypre_ParCSRMatrixNumRows(S_CX));

   hypre_TFree(CF_marker, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( fill,
                      hypre_CSRMatrixData(Id),
                      hypre_CSRMatrixData(Id) + hypre_ParCSRMatrixNumRows(S_CX),
                      coeff );

   SI_diag = hypre_CSRMatrixAddDevice(hypre_ParCSRMatrixDiag(S_CX), Id);

   hypre_CSRMatrixDestroy(Id);

   /* global nnz has changed, but we do not care about it */
   /*
   hypre_ParCSRMatrixSetNumNonzeros(S_CX);
   hypre_ParCSRMatrixDNumNonzeros(S_CX) = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(S_CX);
   */

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(S_CX));
   hypre_ParCSRMatrixDiag(S_CX) = SI_diag;

   hypre_NvtxPopRange();

   /* 2. Perform matrix-matrix multiplication */
   hypre_NvtxPushRangeColor("Matrix-matrix mult", 3);

   S2 = hypre_ParCSRMatMatDevice(S_CX, S_XC);

   hypre_ParCSRMatrixDestroy(S_CX);
   hypre_ParCSRMatrixDestroy(S_XC);

   hypre_NvtxPopRange();

   // Clean up matrix before returning it.
   if (num_paths == 2)
   {
      // If num_paths = 2, prune elements < 2.
      hypre_ParCSRMatrixDropSmallEntriesDevice(S2, 1.5, 0, 0);
   }

   hypre_TFree(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S2)), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(S2)), HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixRemoveDiagonalDevice(hypre_ParCSRMatrixDiag(S2));

   /* global nnz has changed, but we do not care about it */

   hypre_MatvecCommPkgCreate(S2);

   *S2_ptr = S2;

   return 0;
}

#else

//-----------------------------------------------------------------------
HYPRE_Int hypre_BoomerAMGExtractCCDev( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int           *CF_marker,
                                       HYPRE_BigInt        *coarse_row_starts,
                                       hypre_ParCSRMatrix **ACC_ptr);

void truncate( hypre_ParCSRMatrix* S, HYPRE_Complex trunc_level );
void add_diagonal( hypre_ParCSRMatrix* S );
void add_unit_data( hypre_ParCSRMatrix* S );
void remove_diagonal( hypre_ParCSRMatrix* S );

__global__ void hypre_CUDAKernel_ScaleDiag( HYPRE_Int nr_of_rows, HYPRE_Int* SI_diag_i,
                        HYPRE_Complex* SI_diag_a, HYPRE_Complex coeff );

//-----------------------------------------------------------------------
HYPRE_Int hypre_BoomerAMGCreate2ndSDevice( hypre_ParCSRMatrix  *S,
                                           HYPRE_Int           *CF_marker,
                                           HYPRE_Int            num_paths,
                                           HYPRE_BigInt        *coarse_row_starts,
                                           hypre_ParCSRMatrix **C_ptr)
{
   HYPRE_Int nr_of_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(S));
   HYPRE_Complex coeff=1;

   MPI_Comm comm = hypre_ParCSRMatrixComm(S);
   HYPRE_Int num_proc, myid;
   hypre_MPI_Comm_size(comm, &num_proc);
   hypre_MPI_Comm_rank(comm, &myid);

   HYPRE_Int *CF_marker_d = hypre_TAlloc(HYPRE_Int, nr_of_rows, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(CF_marker_d, CF_marker, HYPRE_Int, nr_of_rows, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   // 1. Create new matrix with added diagonal, S+coeff*I, for coeff=1 or 2.
   hypre_NvtxPushRangeColor("Setup", 1);
   add_unit_data( S );
   hypre_ParCSRMatrix* SI = hypre_ParCSRMatrixClone( S, 1 );
   add_diagonal( SI );

   // 2. Add a data array to the strength matrix, needed in order
   //    to use matrix-matrix multiplication.
   add_unit_data( SI );
   hypre_NvtxPopRange();

   // 2b. Scale diagonal by coeff
   if( coeff != 1 )
   {
      HYPRE_Int* SI_diag_i=hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(SI));
      HYPRE_Complex* SI_diag_a=hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(SI));
      dim3 bDim   = hypre_GetDefaultCUDABlockDimension();
      dim3 gDim   = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "thread",   bDim);
      hypre_CUDAKernel_ScaleDiag<<<gDim,bDim>>>( nr_of_rows, SI_diag_i, SI_diag_a, coeff );
      HYPRE_CUDA_CALL(cudaDeviceSynchronize());
   }

   // 3. Perform matrix-matrix multiplication
   //   HYPRE_CUDA_CALL(cudaDeviceSynchronize());
   //   hypre_MPI_Barrier(comm);
   hypre_NvtxPushRangeColor("Matrix-matrix mult", 3);
   hypre_ParCSRMatrix *SIS = hypre_ParCSRMatMatDevice( SI, S );
   hypre_NvtxPopRange();

   // 4. Extract CC parts of the matrices
   hypre_ParCSRMatrix *SCC;
   HYPRE_CUDA_CALL(cudaDeviceSynchronize());
   hypre_MPI_Barrier(comm);
   hypre_NvtxPushRangeColor("ExtractCCDev", 4);
   hypre_BoomerAMGExtractCCDev( SIS, CF_marker_d, coarse_row_starts, &SCC );
   hypre_NvtxPopRange();
   if( num_paths == 2 )
   {
      // 5. If num_paths =2, prune elements < 2.
      truncate(SCC,1.5);
   }
   // 6. Clean up matrix before returning it.
   remove_diagonal( SCC );
   hypre_TFree( hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(SCC)),hypre_ParCSRMatrixMemoryLocation(SCC));
   hypre_TFree( hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(SCC)),hypre_ParCSRMatrixMemoryLocation(SCC));

   hypre_TFree( CF_marker_d, HYPRE_MEMORY_DEVICE);

   *C_ptr = SCC;
   return 0;
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_AddDiagI( HYPRE_Int nr_of_rows, HYPRE_Int* SI_diag_i );
__global__ void hypre_CUDAKernel_AddDiagJ( HYPRE_Int nr_of_rows, HYPRE_Int* SI_diag_i, 
                            HYPRE_Int* SI_diag_j, HYPRE_Int* SI_diag_jnew );

//-----------------------------------------------------------------------
void add_diagonal( hypre_ParCSRMatrix* SI )
{
   // It is assumed that the input SI is a strenght matrix, i.e., it has no diagonal
   // elements and no data array. This function computes SI := SI+I, where the
   // resulting matrix is still without a data array.
   HYPRE_MemoryLocation ml = hypre_ParCSRMatrixMemoryLocation(SI);
   HYPRE_Int nr_of_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(SI));
   HYPRE_Int* SI_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(SI));
   HYPRE_Int nnz        = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(SI));
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(nr_of_rows+1, "thread",   bDim);
   hypre_CUDAKernel_AddDiagI<<<gDim,bDim>>>( nr_of_rows, SI_diag_i );
   HYPRE_Int* SI_diag_j     = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(SI));
   HYPRE_Int* SI_diag_j_new = hypre_CTAlloc( HYPRE_Int, nnz+nr_of_rows, ml);
   cudaDeviceSynchronize();
   dim3 gDim2=hypre_GetDefaultCUDAGridDimension(nr_of_rows, "thread",   bDim);
   hypre_CUDAKernel_AddDiagJ<<<gDim2,bDim>>>( nr_of_rows, SI_diag_i, SI_diag_j, SI_diag_j_new );
   cudaDeviceSynchronize();
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(SI)) = SI_diag_j_new;
   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(SI)) = nnz+nr_of_rows;
   hypre_TFree(SI_diag_j,ml);
   hypre_MatvecCommPkgCreate(SI);
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_MarkDiagonal( HYPRE_Int nr_of_rows, HYPRE_Int* S_diag_i,
                               HYPRE_Int* S_diag_j, HYPRE_Int* dm, HYPRE_Int* dmj );
__global__ void hypre_CUDAKernel_MarkDiagonal_w( HYPRE_Int nr_of_rows, HYPRE_Int* S_diag_i,
                                 HYPRE_Int* S_diag_j, HYPRE_Int* dm, HYPRE_Int* dmj );
struct is_false
{
   __host__ __device__
   bool operator()(const HYPRE_Int x)
   {
      return x==0;
   }
};

//-----------------------------------------------------------------------
void remove_diagonal( hypre_ParCSRMatrix* S )
{
   HYPRE_MemoryLocation ml = hypre_ParCSRMatrixMemoryLocation(S);
   HYPRE_Int nr_of_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(S));
   HYPRE_Int* S_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(S));
   HYPRE_Int* S_diag_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(S));
   HYPRE_Int nnz        = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(S));
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "warp",   bDim);
   HYPRE_Int* dm =hypre_CTAlloc(HYPRE_Int, nr_of_rows+1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int* dmj=hypre_CTAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   hypre_CUDAKernel_MarkDiagonal_w<<<gDim,bDim>>>( nr_of_rows, S_diag_i, S_diag_j, dm, dmj );
   thrust::device_ptr<HYPRE_Int>  dmp(dm), dmjp(dmj);
   HYPRE_Int ndiag=thrust::reduce(dmp,dmp+nr_of_rows);
   HYPRE_Int* S_diag_j_new = hypre_CTAlloc( HYPRE_Int, nnz-ndiag, ml);
   thrust::device_ptr<HYPRE_Int> S_diag_jp(S_diag_j), S_diag_j_newp(S_diag_j_new);
   thrust::copy_if(S_diag_jp,S_diag_jp+nnz,dmjp,S_diag_j_newp,is_false());
   hypre_TFree(S_diag_j,ml);
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(S)) = S_diag_j_new;
   HYPRE_Complex* S_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S));
   if( S_diag_a != NULL )
   {
      HYPRE_Complex* S_diag_a_new = hypre_CTAlloc( HYPRE_Complex, nnz-ndiag, ml);
      thrust::device_ptr<HYPRE_Complex> S_diag_ap(S_diag_a), S_diag_a_newp(S_diag_a_new);
      thrust::copy_if(S_diag_ap,S_diag_ap+nnz,dmjp,S_diag_a_newp,is_false());
      hypre_TFree(S_diag_a,ml);
      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S)) = S_diag_a_new;
   }
   thrust::exclusive_scan(dmp,dmp+nr_of_rows+1,dmp);
   thrust::minus<HYPRE_Int> op;
   thrust::device_ptr<HYPRE_Int> S_diag_ip(S_diag_i);
   thrust::transform( S_diag_ip, S_diag_ip+nr_of_rows+1, dmp, S_diag_ip, op );
   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(S)) = nnz-ndiag;
   hypre_MatvecCommPkgCreate(S);

   hypre_TFree( dm, HYPRE_MEMORY_DEVICE);
   hypre_TFree( dmj, HYPRE_MEMORY_DEVICE);
}

//-----------------------------------------------------------------------
void add_unit_data( hypre_ParCSRMatrix* S )
{
   HYPRE_MemoryLocation ml = hypre_ParCSRMatrixMemoryLocation(S);
   HYPRE_Int nnzd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(S));
   HYPRE_Int nnzo = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(S));
   // In case data exists, remove old arrays, might have wrong size.
   hypre_TFree(hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S)),ml);
   hypre_TFree(hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(S)),ml);
   // Create new data arrays
   HYPRE_Complex* S_diag_a = hypre_CTAlloc( HYPRE_Complex, nnzd, ml );
   HYPRE_Complex* S_offd_a = hypre_CTAlloc( HYPRE_Complex, nnzo, ml );
   thrust::device_ptr<HYPRE_Complex>  S_diag_a_dev(S_diag_a);
   thrust::fill(S_diag_a_dev,S_diag_a_dev+nnzd,1.0);
   thrust::device_ptr<HYPRE_Complex>  S_offd_a_dev(S_offd_a);
   thrust::fill(S_offd_a_dev,S_offd_a_dev+nnzo,1.0);
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S)) = S_diag_a;
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(S)) = S_offd_a;
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_MarkTrunc_w( HYPRE_Int nr_of_rows, HYPRE_Int* S_diag_i,
                              HYPRE_Complex* S_diag_a, HYPRE_Complex trunc_level,
                              HYPRE_Int* dm, HYPRE_Int* dmj );
//-----------------------------------------------------------------------
void truncate( hypre_ParCSRMatrix* S, HYPRE_Complex trunc_level )
{
   HYPRE_MemoryLocation ml = hypre_ParCSRMatrixMemoryLocation(S);
   HYPRE_Int nr_of_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(S));
   HYPRE_Int*     S_diag_i = hypre_CSRMatrixI(   hypre_ParCSRMatrixDiag(S));
   HYPRE_Int*     S_diag_j = hypre_CSRMatrixJ(   hypre_ParCSRMatrixDiag(S));
   HYPRE_Complex* S_diag_a = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S));
   HYPRE_Int nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(S));
   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(nr_of_rows, "warp",   bDim);
   HYPRE_Int* dm =hypre_CTAlloc(HYPRE_Int, nr_of_rows+1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int* dmj=hypre_CTAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   hypre_CUDAKernel_MarkTrunc_w<<<gDim,bDim>>>( nr_of_rows, S_diag_i, S_diag_a, trunc_level, dm, dmj );
   thrust::device_ptr<HYPRE_Int>  dmp(dm), dmjp(dmj);
   HYPRE_Int no_remove=thrust::reduce(dmp,dmp+nr_of_rows);

   // New col index array
   HYPRE_Int* S_diag_j_new = hypre_CTAlloc( HYPRE_Int, nnz-no_remove, ml);
   thrust::device_ptr<HYPRE_Int> S_diag_jp(S_diag_j), S_diag_j_newp(S_diag_j_new);
   thrust::copy_if(S_diag_jp,S_diag_jp+nnz,dmjp,S_diag_j_newp,is_false());
   hypre_TFree(S_diag_j,ml);
   hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(S)) = S_diag_j_new;

   // New data array
   HYPRE_Complex* S_diag_a_new = hypre_CTAlloc( HYPRE_Complex, nnz-no_remove, ml);
   thrust::device_ptr<HYPRE_Complex> S_diag_ap(S_diag_a), S_diag_a_newp(S_diag_a_new);
   thrust::copy_if(S_diag_ap,S_diag_ap+nnz,dmjp,S_diag_a_newp,is_false());
   hypre_TFree(S_diag_a,ml);
   hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(S)) = S_diag_a_new;

   thrust::exclusive_scan(dmp,dmp+nr_of_rows+1,dmp);
   thrust::minus<HYPRE_Int> op;
   thrust::device_ptr<HYPRE_Int> S_diag_ip(S_diag_i);
   thrust::transform( S_diag_ip, S_diag_ip+nr_of_rows+1, dmp, S_diag_ip, op );
   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(S)) = nnz-no_remove;

   // Off diagonal part
   hypre_TFree( dmj, HYPRE_MEMORY_DEVICE);

   HYPRE_Int*     S_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(S));
   HYPRE_Int*     S_offd_j = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(S));
   HYPRE_Complex* S_offd_a = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(S));
   nnz = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(S));
   dmj=hypre_CTAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   hypre_CUDAKernel_MarkTrunc_w<<<gDim,bDim>>>( nr_of_rows, S_offd_i, S_offd_a, trunc_level, dm, dmj );
   no_remove=thrust::reduce(dmp,dmp+nr_of_rows);

   // New col index array
   HYPRE_Int* S_offd_j_new = hypre_CTAlloc( HYPRE_Int, nnz-no_remove, ml);
   thrust::device_ptr<HYPRE_Int> S_offd_jp(S_offd_j), S_offd_j_newp(S_offd_j_new);
   thrust::device_ptr<HYPRE_Int> dmjp2(dmj);
   thrust::copy_if(S_offd_jp,S_offd_jp+nnz,dmjp2,S_offd_j_newp,is_false());
   hypre_TFree(S_offd_j,ml);
   hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(S)) = S_offd_j_new;
   // New data array
   HYPRE_Complex* S_offd_a_new = hypre_CTAlloc( HYPRE_Complex, nnz-no_remove, ml);
   thrust::device_ptr<HYPRE_Complex> S_offd_ap(S_offd_a), S_offd_a_newp(S_offd_a_new);
   thrust::copy_if(S_offd_ap,S_offd_ap+nnz,dmjp2,S_offd_a_newp,is_false());
   hypre_TFree(S_offd_a,ml);
   hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(S)) = S_offd_a_new;

   thrust::exclusive_scan(dmp,dmp+nr_of_rows+1,dmp);
   //   thrust::minus<HYPRE_Int> op;
   thrust::device_ptr<HYPRE_Int> S_offd_ip(S_offd_i);
   thrust::transform( S_offd_ip, S_offd_ip+nr_of_rows+1, dmp, S_offd_ip, op );
   hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(S)) = nnz-no_remove;


// Should compress local index in offd, but the routine will likely work without doing it
   hypre_TFree( dm, HYPRE_MEMORY_DEVICE);
   hypre_TFree( dmj, HYPRE_MEMORY_DEVICE);

   hypre_MatvecCommPkgCreate(S);
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_ScaleDiag( HYPRE_Int nr_of_rows, HYPRE_Int* SI_diag_i,
                        HYPRE_Complex* SI_diag_a, HYPRE_Complex coeff )
{
   HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>();
   if( row < nr_of_rows )
      SI_diag_a[SI_diag_i[row]] *= coeff;
 // Assuming each row has a diagonal element, that occurs first in the row.
}

//-----------------------------------------------------------------------
__global__  void hypre_CUDAKernel_AddDiagI( HYPRE_Int nr_of_rows, HYPRE_Int* SI_diag_i )
{
   HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>();
   if( row <= nr_of_rows )
      SI_diag_i[row] += row;
  //Adding one element per row, assumes that no row has a diagonal element initially.
}

//-----------------------------------------------------------------------
__global__  void hypre_CUDAKernel_AddDiagJ( HYPRE_Int nr_of_rows, HYPRE_Int* SI_diag_i,
                        HYPRE_Int* SI_diag_j, HYPRE_Int* SI_diag_jnew )
{
   HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>(), ib, ie;
   if( row >= nr_of_rows )
      return;
   ib = SI_diag_i[row];
   ie = SI_diag_i[row+1];
   SI_diag_jnew[ib] = row;
   for( HYPRE_Int ind=ib+1; ind < ie; ind++ )
      SI_diag_jnew[ind] = SI_diag_j[ind-row-1];
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_MarkDiagonal( HYPRE_Int nr_of_rows, HYPRE_Int* S_diag_i,
                               HYPRE_Int* S_diag_j, HYPRE_Int* dm, HYPRE_Int* dmj )
{
   HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>();//, ib, ie;
   if( row >= nr_of_rows )
      return;
   dm[row]=0;
   for( HYPRE_Int ind=S_diag_i[row]; ind < S_diag_i[row+1]; ind++ )
   {
      dmj[ind]=0;
      if( S_diag_j[ind]==row )
      {
         dm[row]=1;
         dmj[ind]=1;
      }
   }
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_MarkDiagonal_w( HYPRE_Int nr_of_rows, HYPRE_Int* S_diag_i,
                                 HYPRE_Int* S_diag_j, HYPRE_Int* dm, HYPRE_Int* dmj )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>(), ib, ie, sj, hd;
   if( row >= nr_of_rows )
   {
      return;
   }
   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   if (lane < 2)
   {
      ib = read_only_load(S_diag_i + row + lane);
   }
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);
   hd = 0;
   for (HYPRE_Int ind = ib + lane; __any_sync(HYPRE_WARP_FULL_MASK, ind < ie); ind += HYPRE_WARP_SIZE)
   {
      if( ind < ie )
      {
         sj = read_only_load(S_diag_j + ind);
         hd= (sj==row||hd==1);
         dmj[ind] = sj==row;
      }
   }
   dm[row]=warp_allreduce_max(hd);
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_MarkTrunc_w( HYPRE_Int nr_of_rows, HYPRE_Int* S_diag_i,
                                   HYPRE_Complex* S_diag_a,
                                   HYPRE_Complex trunc_level,
                                   HYPRE_Int* dm, HYPRE_Int* dmj )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>(), ib, ie, hd;
   HYPRE_Complex sa;
   if( row >= nr_of_rows )
   {
      return;
   }
   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   if (lane < 2)
   {
      ib = read_only_load(S_diag_i + row + lane);
   }
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);
   hd = 0;
   for (HYPRE_Int ind = ib + lane; __any_sync(HYPRE_WARP_FULL_MASK, ind < ie); ind += HYPRE_WARP_SIZE)
   {
      if( ind < ie )
      {
         sa = read_only_load(S_diag_a + ind);
         hd += (sa <= trunc_level);
         dmj[ind] = sa <= trunc_level; // 1--> remove, 0--> keep
      }
   }
   dm[row]=warp_allreduce_sum(hd); // Number of elements to remove in `row'
}

//-----------------------------------------------------------------------

__global__ void hypre_CUDAKernel_GetCfarray( HYPRE_Int nrofrows, HYPRE_Int nnz,  HYPRE_Int* CF_marker,
                                             HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j, HYPRE_Int* CF_array );
__global__ void hypre_CUDAKernel_GetCfarrayOffd( HYPRE_Int nrofrows, HYPRE_Int nnz,  HYPRE_Int* CF_marker,
                                                 HYPRE_Int* CF_marker_offd, HYPRE_Int* A_offd_i,
                                                 HYPRE_Int* A_offd_j, HYPRE_Int* CF_array );
__global__ void hypre_CUDAKernel_GetIcarray( HYPRE_Int nrofrows,  HYPRE_Int* A_diag_i, HYPRE_Int* CF_marker,
                                             HYPRE_Int* CF_array, HYPRE_Int* ACC_tmp );
__global__ void hypre_CUDAKernel_TransformColindDiag( HYPRE_Int nnz, HYPRE_Int* fine_to_coarse,
                                                      HYPRE_Int* A_diag_j );
void restrict_colmap_to_cpts( HYPRE_Int ACC_offd_size, HYPRE_Int* ACC_offd_j,
                              HYPRE_Int nr_cols_A, HYPRE_Int* col_map_A,
                              HYPRE_Int* nr_cols_ACC, HYPRE_Int** col_map_ACC );
void CoarseToFine_comm( MPI_Comm comm, HYPRE_Int nr_cols_P, HYPRE_Int* col_map_P,
                        HYPRE_Int* fine_to_coarse, HYPRE_Int first_diagonal,
                        HYPRE_Int nr_of_rows, HYPRE_Int* num_cpts, HYPRE_Int global_nr_of_rows );
void getcfmoffd( hypre_ParCSRMatrix* A, HYPRE_Int* CF_marker, HYPRE_Int** CF_marker_offd );


HYPRE_Int hypre_BoomerAMGExtractCCDev( hypre_ParCSRMatrix  *A,
                                       HYPRE_Int           *CF_marker,
                                       HYPRE_BigInt        *coarse_row_starts,
                                       hypre_ParCSRMatrix **ACC_ptr)
{
   hypre_CSRMatrix    *A_diag     = hypre_ParCSRMatrixDiag(A);
   HYPRE_Complex      *A_diag_a   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int          *A_diag_i   = hypre_CSRMatrixI(A_diag);
   HYPRE_Int          *A_diag_j   = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int           A_diag_nnz = hypre_CSRMatrixNumNonzeros(A_diag);

   hypre_CSRMatrix    *A_offd     = hypre_ParCSRMatrixOffd(A);
   HYPRE_Complex      *A_offd_a   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int          *A_offd_i   = hypre_CSRMatrixI(A_offd);
   HYPRE_Int          *A_offd_j   = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int           A_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);

   // 0. Information on coarse dimensions
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int num_procs, my_id;
   HYPRE_Int my_first_cpt, my_last_cpt, global_num_coarse;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   my_first_cpt = coarse_row_starts[0];
   my_last_cpt  = coarse_row_starts[1]-1;
   if (my_id == (num_procs -1)) global_num_coarse = coarse_row_starts[1];
   hypre_MPI_Bcast(&global_num_coarse, 1, HYPRE_MPI_BIG_INT, num_procs-1, comm);
#else
   my_first_cpt = coarse_row_starts[my_id];
   my_last_cpt  = coarse_row_starts[my_id+1]-1;
   global_num_coarse = coarse_row_starts[num_procs];
#endif

   // 1. diagonal part of ACC
   HYPRE_Int nrofrows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int *CF_array= hypre_CTAlloc(HYPRE_Int,A_diag_nnz,HYPRE_MEMORY_DEVICE);
   thrust::device_ptr<HYPRE_Int> CF_array_dev(CF_array);
   thrust::fill(CF_array_dev,CF_array_dev+A_diag_nnz,0);
   dim3 bDim   = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim   = hypre_GetDefaultCUDAGridDimension( nrofrows, "thread",   bDim);
   dim3 gDimw  = hypre_GetDefaultCUDAGridDimension( nrofrows, "warp",   bDim);
   hypre_CUDAKernel_GetCfarray<<<gDim,bDim>>>( nrofrows, A_diag_nnz, CF_marker, A_diag_i, A_diag_j, CF_array );

   HYPRE_Int *ACC_tmp = hypre_CTAlloc( HYPRE_Int, nrofrows, HYPRE_MEMORY_DEVICE );
   hypre_CUDAKernel_GetIcarray<<<gDimw,bDim>>>( nrofrows, A_diag_i, CF_marker, CF_array, ACC_tmp );

   HYPRE_Int nrofcoarserows = my_last_cpt-my_first_cpt+1;
   //   HYPRE_Int nrofcoarserows = thrust::count_if( CF_marker, CF_marker+nrofrows, is_positive() );

   HYPRE_Int *ACC_diag_i = hypre_CTAlloc( HYPRE_Int, nrofcoarserows+1, HYPRE_MEMORY_DEVICE);
   thrust::device_ptr<HYPRE_Int> ACC_tmp_dev(ACC_tmp), CF_marker_dev(CF_marker), ACC_diag_i_dev(ACC_diag_i);
   thrust::copy_if( ACC_tmp_dev, ACC_tmp_dev+nrofrows, CF_marker_dev, ACC_diag_i_dev, is_positive<HYPRE_Int>() );
   thrust::exclusive_scan( ACC_diag_i_dev, ACC_diag_i_dev+nrofcoarserows+1, ACC_diag_i_dev );
   HYPRE_Int ACC_diag_nnz = ACC_diag_i[nrofcoarserows]; // Should do this on device

   HYPRE_Int *ACC_diag_j    = NULL;
   HYPRE_Complex *ACC_diag_a= NULL;
   if( ACC_diag_nnz > 0 )
   {
      ACC_diag_j = hypre_CTAlloc(HYPRE_Int, ACC_diag_nnz, HYPRE_MEMORY_DEVICE);

      thrust::device_ptr<HYPRE_Int> A_diag_j_dev(A_diag_j), ACC_diag_j_dev(ACC_diag_j);
      thrust::device_ptr<HYPRE_Int> CF_array_dev(CF_array);
      thrust::copy_if( A_diag_j_dev, A_diag_j_dev+A_diag_nnz, CF_array_dev, ACC_diag_j_dev, is_positive<HYPRE_Int>() );
      ACC_diag_a = hypre_CTAlloc( HYPRE_Complex, ACC_diag_nnz, HYPRE_MEMORY_DEVICE);
      thrust::device_ptr<HYPRE_Complex> A_diag_a_dev(A_diag_a), ACC_diag_a_dev(ACC_diag_a);
      thrust::copy_if( A_diag_a_dev, A_diag_a_dev+A_diag_nnz, CF_array_dev, ACC_diag_a_dev, is_positive<HYPRE_Int>() );
   }
   hypre_TFree(CF_array,HYPRE_MEMORY_DEVICE);

   // 2. off-diagonal part of ACC
   HYPRE_Int* CF_marker_offd;
   getcfmoffd( A, CF_marker, &CF_marker_offd );

   HYPRE_Int *CF_array_offd = hypre_CTAlloc(HYPRE_Int,A_offd_nnz,HYPRE_MEMORY_DEVICE);
   hypre_CUDAKernel_GetCfarrayOffd<<<gDim,bDim>>>( nrofrows, A_offd_nnz, CF_marker, CF_marker_offd, A_offd_i, A_offd_j, CF_array_offd );

   hypre_CUDAKernel_GetIcarray<<<gDimw,bDim>>>( nrofrows, A_offd_i, CF_marker, CF_array_offd, ACC_tmp );

   HYPRE_Int *ACC_offd_i = hypre_CTAlloc(HYPRE_Int,nrofcoarserows+1,HYPRE_MEMORY_DEVICE);
   thrust::device_ptr<HYPRE_Int> ACC_offd_i_dev(ACC_offd_i);
   thrust::copy_if( ACC_tmp_dev, ACC_tmp_dev+nrofrows, CF_marker_dev, ACC_offd_i_dev, is_positive<HYPRE_Int>() );
   thrust::exclusive_scan( ACC_offd_i_dev, ACC_offd_i_dev+nrofcoarserows+1, ACC_offd_i_dev );
   HYPRE_Int ACC_offd_nnz = ACC_offd_i[nrofcoarserows];

   HYPRE_Int *ACC_offd_j    = NULL;
   HYPRE_Complex *ACC_offd_a= NULL;
   if( ACC_offd_nnz > 0 )
   {
      ACC_offd_j= hypre_CTAlloc(HYPRE_Int, ACC_offd_nnz, HYPRE_MEMORY_DEVICE);
      thrust::device_ptr<HYPRE_Int> ACC_offd_j_dev(ACC_offd_j), CF_array_offd_dev(CF_array_offd);
      thrust::device_ptr<HYPRE_Int> A_offd_j_dev(A_offd_j);
      thrust::copy_if( A_offd_j_dev, A_offd_j_dev+A_offd_nnz, CF_array_offd_dev, ACC_offd_j_dev, is_positive<HYPRE_Int>() );
      ACC_offd_a = hypre_CTAlloc( HYPRE_Complex, ACC_offd_nnz, HYPRE_MEMORY_DEVICE);
      thrust::device_ptr<HYPRE_Complex> A_offd_a_dev(A_offd_a), ACC_offd_a_dev(ACC_offd_a);
      thrust::copy_if( A_offd_a_dev, A_offd_a_dev+A_offd_nnz, CF_array_offd_dev, ACC_offd_a_dev, is_positive<HYPRE_Int>() );
   }
   hypre_TFree(ACC_tmp,HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_array_offd,HYPRE_MEMORY_DEVICE);
   // 3. Transform diagonal submatrix column index (F,C) to (C)
   HYPRE_Int* fine_to_coarse = hypre_CTAlloc(HYPRE_Int,nrofrows, HYPRE_MEMORY_DEVICE);
   thrust::device_ptr<HYPRE_Int> fine_to_coarse_dev(fine_to_coarse);
   thrust::transform( CF_marker_dev, CF_marker_dev+nrofrows, fine_to_coarse_dev, is_positive<HYPRE_Int>());
   thrust::exclusive_scan( fine_to_coarse_dev, fine_to_coarse_dev+nrofrows, fine_to_coarse_dev );
   if( ACC_diag_nnz > 0 )
   {
      gDim   = hypre_GetDefaultCUDAGridDimension( ACC_diag_nnz, "thread",   bDim);
      hypre_CUDAKernel_TransformColindDiag<<<gDim,bDim>>>( ACC_diag_nnz, fine_to_coarse, ACC_diag_j );
      cudaDeviceSynchronize();
   }

   // 4. Transform off-diagonal submatrix column index (F,C) to (C)
   HYPRE_Int* col_map_A_dev = hypre_ParCSRMatrixDeviceColMapOffd(A);
   //   HYPRE_Int* col_map_A     = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int nr_cols_A  = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_Int nr_cols_ACC=0;
   HYPRE_Int* col_map_ACC=NULL, *col_map_ACC_h=NULL;
   if( nr_cols_A > 0 )
   {
      restrict_colmap_to_cpts( ACC_offd_nnz, ACC_offd_j, nr_cols_A, col_map_A_dev,
                               &nr_cols_ACC, &col_map_ACC );
      //      cudaDeviceSynchronize();
      HYPRE_Int  first_diagonal = hypre_ParCSRMatrixFirstColDiag(A);
      HYPRE_Int  global_nr_of_rows = hypre_ParCSRMatrixGlobalNumRows(A);
      col_map_ACC_h = hypre_CTAlloc( HYPRE_Int, nr_cols_ACC, HYPRE_MEMORY_HOST);
      hypre_TMemcpy( col_map_ACC_h, col_map_ACC, HYPRE_Int, nr_cols_ACC, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      CoarseToFine_comm( comm, nr_cols_ACC, col_map_ACC_h, fine_to_coarse, first_diagonal, nrofrows,
                         coarse_row_starts, global_nr_of_rows );
      hypre_TMemcpy( col_map_ACC, col_map_ACC_h, HYPRE_Int, nr_cols_ACC, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
   }
   hypre_TFree(fine_to_coarse,HYPRE_MEMORY_DEVICE);
   // 5. Create ParCSRMatrix structure and return it
   hypre_ParCSRMatrix* ACC = hypre_ParCSRMatrixCreate(comm, global_num_coarse,
         global_num_coarse, coarse_row_starts,
         coarse_row_starts, nr_cols_ACC, ACC_diag_i[nrofcoarserows], ACC_offd_i[nrofcoarserows]);

   hypre_ParCSRMatrixOwnsRowStarts(ACC) = 0;

   hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(ACC)) = ACC_diag_i;
   if (ACC_diag_i[nrofcoarserows])
   {
      hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(ACC))    = ACC_diag_j;
      hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(ACC)) = ACC_diag_a;
   }

   hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(ACC)) = ACC_offd_i;
   if (nr_cols_ACC)
   {
      if (ACC_offd_i[nrofcoarserows])
      {
         hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(ACC))    = ACC_offd_j;
         hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(ACC)) = ACC_offd_a;
      }
      hypre_ParCSRMatrixColMapOffd(ACC)       = col_map_ACC_h;
      hypre_ParCSRMatrixDeviceColMapOffd(ACC) = col_map_ACC;
   }
   hypre_ParCSRMatrixCommPkg(ACC) = NULL;
   *ACC_ptr = ACC;
   return 0;
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_GetCfarray( HYPRE_Int nrofrows, HYPRE_Int nnz,  HYPRE_Int* CF_marker,
                             HYPRE_Int* A_diag_i, HYPRE_Int* A_diag_j, HYPRE_Int* CF_array )
{
   HYPRE_Int i=hypre_cuda_get_grid_thread_id<1,1>();
   if( i < nrofrows )
      for( HYPRE_Int ind=A_diag_i[i] ; ind < A_diag_i[i+1] ; ind++ )
         CF_array[ind] = CF_marker[i]>0 && CF_marker[A_diag_j[ind]]>0;
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_GetCfarrayOffd( HYPRE_Int nrofrows, HYPRE_Int nnz,  HYPRE_Int* CF_marker,
                                  HYPRE_Int* CF_marker_offd, HYPRE_Int* A_offd_i,
                                  HYPRE_Int* A_offd_j, HYPRE_Int* CF_array )
{
   HYPRE_Int i=hypre_cuda_get_grid_thread_id<1,1>();
   if( i < nrofrows )
      for( HYPRE_Int ind=A_offd_i[i] ; ind < A_offd_i[i+1] ; ind++ )
         CF_array[ind] = CF_marker[i]>0 && CF_marker_offd[A_offd_j[ind]]>0;
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_GetIcarray( HYPRE_Int nrofrows, HYPRE_Int* A_i,
                             HYPRE_Int* CF_marker, HYPRE_Int* CF_array, HYPRE_Int* ACC_tmp )
{
   HYPRE_Int i = hypre_cuda_get_grid_warp_id<1,1>();
   if( i >= nrofrows )
   {
      return;
   }
   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int ib, ie;
   ACC_tmp[i] = 0;
   if (lane == 0)
   {
      ib = read_only_load(CF_marker + i);
   }
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);
   if (ib <= 0)
   {
      return;
   }
   HYPRE_Int ncl=0;
   if (lane < 2)
   {
      ib = read_only_load(A_i + i + lane);
   }
   ie = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 1);
   ib = __shfl_sync(HYPRE_WARP_FULL_MASK, ib, 0);
   for (HYPRE_Int ind = ib + lane; __any_sync(HYPRE_WARP_FULL_MASK, ind < ie); ind += HYPRE_WARP_SIZE)
   {
      if( ind < ie )
         ncl += CF_array[ind]==1;

   }
   ncl=warp_reduce_sum(ncl);
   if( lane == 0 )
      ACC_tmp[i]= ncl;
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_TransformColindDiag( HYPRE_Int nnz, HYPRE_Int* fine_to_coarse,
                                       HYPRE_Int* A_diag_j )
{
   HYPRE_Int i=hypre_cuda_get_grid_thread_id<1,1>();
   if( i < nnz )
      A_diag_j[i] = fine_to_coarse[A_diag_j[i]];
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_MarkCpts( HYPRE_Int P_offd_size, HYPRE_Int *P_offd_j,
                           HYPRE_Int* fine_to_coarse )
{
   HYPRE_Int myid=hypre_cuda_get_grid_thread_id<1,1>();
   if( myid < P_offd_size )
         atomicOr(&fine_to_coarse[P_offd_j[myid]],1);
}

//-----------------------------------------------------------------------
__global__ void hypre_CUDAKernel_RemapOffd( HYPRE_Int P_offd_size, HYPRE_Int *P_offd_j,
                            HYPRE_Int* fine_to_coarse )
{
   HYPRE_Int myid = hypre_cuda_get_grid_thread_id<1,1>();
   if( myid < P_offd_size )
      P_offd_j[myid] = fine_to_coarse[P_offd_j[myid]];
}

//-----------------------------------------------------------------------
void restrict_colmap_to_cpts( HYPRE_Int ACC_offd_size, HYPRE_Int* ACC_offd_j,
                              HYPRE_Int nr_cols_A, HYPRE_Int* col_map_A,
                              HYPRE_Int* nr_cols_ACC, HYPRE_Int** col_map_ACC )
{
   /* Remove F-pts and unused values from column map colmap_P. Re-enumerate P_off_j to the
      new, compressed, local index space */
   HYPRE_Int *new_col_map;
   HYPRE_Int* fine_to_coarse;
   HYPRE_Int ncpts=0;
   if( ACC_offd_size > 0 )
   {
      dim3 block = hypre_GetDefaultCUDABlockDimension();
      dim3 grid  = hypre_GetDefaultCUDAGridDimension( ACC_offd_size, "thread",   block);
      fine_to_coarse = hypre_CTAlloc(HYPRE_Int, nr_cols_A+1, HYPRE_MEMORY_DEVICE);
      grid   = hypre_GetDefaultCUDAGridDimension( ACC_offd_size, "thread",   block);
      thrust::device_ptr<HYPRE_Int> fine_to_coarse_dev(fine_to_coarse);
      thrust::fill(fine_to_coarse_dev,fine_to_coarse_dev+nr_cols_A+1,0);
      hypre_CUDAKernel_MarkCpts<<<grid,block>>>( ACC_offd_size, ACC_offd_j, fine_to_coarse );
      ncpts = thrust::reduce(fine_to_coarse_dev,fine_to_coarse_dev+nr_cols_A);

   /* Construct new column map */
      new_col_map = hypre_CTAlloc( HYPRE_Int, ncpts, HYPRE_MEMORY_DEVICE );
      thrust::device_ptr<HYPRE_Int> new_col_map_dev(new_col_map), col_map_A_dev(col_map_A);
      thrust::copy_if( col_map_A_dev, col_map_A_dev+nr_cols_A, fine_to_coarse_dev,
                    new_col_map_dev, is_positive<HYPRE_Int>() );

   /* Map A_offd_j to new local indices */
      thrust::exclusive_scan( fine_to_coarse_dev, &fine_to_coarse_dev[nr_cols_A+1], fine_to_coarse_dev );
      hypre_CUDAKernel_RemapOffd<<<grid,block>>>( ACC_offd_size, ACC_offd_j, fine_to_coarse );
      cudaDeviceSynchronize();
   /* Give back memory and return result */
      //      hypre_TFree(*col_map_P,HYPRE_MEMORY_SHARED);
      hypre_TFree(fine_to_coarse,HYPRE_MEMORY_DEVICE);
      *nr_cols_ACC = ncpts;
      *col_map_ACC = new_col_map;
   }
}


/*-----------------------------------------------------------------------*/
HYPRE_Int getSendInfo( HYPRE_Int nr_cols_P, HYPRE_Int* col_map_P,
                 HYPRE_Int* num_cpts, HYPRE_Int num_proc,
                 HYPRE_Int global_nr_of_rows,
                 HYPRE_Int** nrec, HYPRE_Int** recprocs )
{
   /* Collect information about owners of coarse points */
   /* Input: nr_cols_P - size of col_map_P              */
   /*        col_map_P - Global index of columns of P in the total array enumeration 0..n-1 */
   /*        nr_of_rows - Number of rows in proc               */
   /*        num_cpts - first fine grid point index for each processor */
   /*        num_proc - Number of  processors */
   /* Output: nrec - nrec[p] is the number of elements to receive from processor p */
   /*         recprocs - recprocs[i] is the processor id of the value of element i of col_map_P */

   HYPRE_Int i, toolow, toohigh, fail, proc;
   fail = 0;
   for( i=0 ; i < nr_cols_P ; i++ )
   {
      /* Initial guess */
      toolow=toohigh=0;
      proc = (HYPRE_Int)trunc(((HYPRE_Real)col_map_P[i]/global_nr_of_rows)*num_proc);
      if( num_cpts[proc] <= col_map_P[i] )
      {
         if( col_map_P[i] < num_cpts[proc+1] )
         {
            (*nrec)[proc]++;
            (*recprocs)[i]=proc;
         }
         else
            toolow=1;
      }
      else
         toohigh = 1;
      /* Adjust if guess not correct */
      if( toohigh )
      {
         while( proc >= 0 && (num_cpts[proc] > col_map_P[i]) )
            proc--;
         if( proc < 0 )
            fail = 1;
         else
         {
            (*nrec)[proc]++;
            (*recprocs)[i]=proc;
         }
      }
      if( toolow )
      {
         while( (proc <= num_proc) && num_cpts[proc] < col_map_P[i] )
            proc++;
         if( proc > num_proc )
            fail = 2;
         else
         {
            (*nrec)[proc]++;
            (*recprocs)[i]=proc;
         }
      }
   }
   return fail;
}

/*-----------------------------------------------------------------------*/
void CoarseToFine_comm( MPI_Comm comm, HYPRE_Int nr_cols_P, HYPRE_Int* col_map_P,
                        HYPRE_Int* fine_to_coarse, HYPRE_Int first_diagonal,
                        HYPRE_Int nr_of_rows, HYPRE_Int* num_cpts, HYPRE_Int global_nr_of_rows )
{
   /* Translate the indices of the column map col_map_P to a coarse point enumeration */
   /* It is assumed that col_map_P holds indices in the range 0..n-1, where n is the gobal number */
   /* of columns. Furthermore, nc of the n colums correspond to coarse point indices.  */
   /* The elements in col_map_P are assumed to be coarse point indices. This routine transforms */
   /* them from the 0..n-1 enumeration to a 0..nc-1 enumeration */

   HYPRE_Int** buf, **sbuf;
   HYPRE_Int* el, *nrec, *nsend, *recprocs;
   HYPRE_Int num_proc, p, i, fail, tag, tag2, tag3;
   hypre_MPI_Request* req, *req2, *req3;
   hypre_MPI_Status status;
   HYPRE_Int* first_fine_pr, *fine_to_coarse_h;
   HYPRE_Int first_coarse, myid;

   hypre_MPI_Comm_size(comm, &num_proc);
   hypre_MPI_Comm_rank(comm, &myid);
   fine_to_coarse_h = hypre_CTAlloc(HYPRE_Int, nr_of_rows, HYPRE_MEMORY_HOST );
   hypre_TMemcpy( fine_to_coarse_h, fine_to_coarse, HYPRE_Int, nr_of_rows, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   nrec     = hypre_CTAlloc(HYPRE_Int, num_proc, HYPRE_MEMORY_HOST ); /* receive this number of elements from proc p */
   recprocs = hypre_CTAlloc(HYPRE_Int, nr_cols_P, HYPRE_MEMORY_HOST );
   nsend    = hypre_CTAlloc(HYPRE_Int, num_proc, HYPRE_MEMORY_HOST ); /* send this number of elements to proc p      */
   el       = hypre_CTAlloc(HYPRE_Int, num_proc, HYPRE_MEMORY_HOST );

   first_fine_pr = hypre_CTAlloc(HYPRE_Int, num_proc+1, HYPRE_MEMORY_HOST );
   hypre_MPI_Allgather( &first_diagonal, 1, HYPRE_MPI_INT, first_fine_pr, 1, HYPRE_MPI_INT, comm );
   first_fine_pr[num_proc]=global_nr_of_rows;

   fail = getSendInfo( nr_cols_P, col_map_P, first_fine_pr, num_proc, global_nr_of_rows, &nrec, &recprocs );
   hypre_TFree(first_fine_pr,HYPRE_MEMORY_HOST);

#ifdef HYPRE_NO_GLOBAL_PARTITION
   first_coarse = num_cpts[0];
#else
   first_coarse = num_cpts[myid];
#endif
   if( fail == 0 )
   {
      req   = hypre_CTAlloc(hypre_MPI_Request,num_proc,HYPRE_MEMORY_HOST);
      req2  = hypre_CTAlloc(hypre_MPI_Request,num_proc,HYPRE_MEMORY_HOST);
      req3  = hypre_CTAlloc(hypre_MPI_Request,num_proc,HYPRE_MEMORY_HOST);

      tag = 305;
      tag2= 306;
      tag3= 307;

      /* Receive wanted number of elements */
      for( p=0 ; p < num_proc ; p++ )
         hypre_MPI_Irecv( &nsend[p], 1, HYPRE_MPI_INT, p, tag, comm, &req[p] );

      buf  = hypre_CTAlloc( HYPRE_Int*, num_proc, HYPRE_MEMORY_HOST);
      sbuf = hypre_CTAlloc( HYPRE_Int*, num_proc, HYPRE_MEMORY_HOST);

      /* Send wanted number of elements */
      for( p=0 ; p < num_proc ; p++ )
      {
         hypre_MPI_Isend(&nrec[p], 1, HYPRE_MPI_INT,  p, tag, comm, &req2[p] );
         el[p]  = 0;
         buf[p] = hypre_CTAlloc( HYPRE_Int, nrec[p], HYPRE_MEMORY_HOST);
      }
      /* Pack the indices for which transformation is wanted */
      for( i=0 ; i < nr_cols_P ; i++ )
      {
         p=recprocs[i];
         buf[p][el[p]++] = col_map_P[i];
      }
      /* finish size receive, and post receive for the data */
      for( p=0 ; p < num_proc ; p++ )
      {
         hypre_MPI_Wait( &req[p], &status );
         sbuf[p] = hypre_CTAlloc(HYPRE_Int, nsend[p], HYPRE_MEMORY_HOST );
         hypre_MPI_Irecv( sbuf[p], nsend[p], HYPRE_MPI_INT, p, tag2, comm, &req[p] );
      }
      /* Send the packed indices */
      for( p=0 ; p < num_proc ; p++ )
         hypre_MPI_Isend( buf[p], nrec[p], HYPRE_MPI_INT, p, tag2, comm, &req2[p] );

      //      /* Post receive for last send */
      //      for( p=0 ; p < num_proc ; p++ )
      //         hypre_MPI_Irecv( buf[p], nrec[p], HYPRE_MPI_INT, p, tag3, comm, &req3[p] );

      /* Transform indices and send them back to their destinations */
      for( p=0 ; p < num_proc ; p++ )
      {
         hypre_MPI_Wait( &req[p], &status );
         for( i=0 ; i < nsend[p]; i++ )
            sbuf[p][i] = fine_to_coarse_h[sbuf[p][i]-first_diagonal]+first_coarse;
         hypre_MPI_Isend( sbuf[p], nsend[p], HYPRE_MPI_INT, p, tag3, comm, &req2[p] );
      }
      /* Post receive for last send */
      for( p=0 ; p < num_proc ; p++ )
         hypre_MPI_Irecv( buf[p], nrec[p], HYPRE_MPI_INT, p, tag3, comm, &req3[p] );

      /* Wait for last irecv */
      for( p=0 ; p < num_proc ; p++ )
      {
         hypre_MPI_Wait( &req3[p], &status );
         el[p] = 0;
      }
      /* store the transformed indices */
      for( i=0 ; i < nr_cols_P ; i++ )
         col_map_P[i] = buf[recprocs[i]][el[recprocs[i]]++];

      hypre_TFree(req,  HYPRE_MEMORY_HOST);
      hypre_TFree(req2, HYPRE_MEMORY_HOST);
      hypre_TFree(req3, HYPRE_MEMORY_HOST);
      for( p=0 ; p < num_proc ; p++ )
      {
         hypre_TFree( buf[p], HYPRE_MEMORY_HOST);
         hypre_TFree(sbuf[p], HYPRE_MEMORY_HOST);
      }
      hypre_TFree( buf, HYPRE_MEMORY_HOST);
      hypre_TFree(sbuf, HYPRE_MEMORY_HOST);
   }
   else
      printf("ERROR: fail = %d in CoarseToFine, proc %d \n",fail,myid);
   hypre_TFree(fine_to_coarse_h, HYPRE_MEMORY_HOST);
   hypre_TFree(nrec,    HYPRE_MEMORY_HOST);
   hypre_TFree(nsend,   HYPRE_MEMORY_HOST);
   hypre_TFree(recprocs,HYPRE_MEMORY_HOST);
   hypre_TFree(el,      HYPRE_MEMORY_HOST);
}

//-----------------------------------------------------------------------
void getcfmoffd( hypre_ParCSRMatrix* A, HYPRE_Int* CF_marker, HYPRE_Int** CF_marker_offd )
{
   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int index, num_sends, *int_buf_data, i, j, start;
   HYPRE_Int num_cols_A_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));

   if (num_cols_A_offd > 0)
      *CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_A_offd, HYPRE_MEMORY_DEVICE);
   else
   {
      *CF_marker_offd = NULL;
      return;
   }
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, 
						num_sends), HYPRE_MEMORY_HOST);
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
		int_buf_data[index++] 
		 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }
   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, 
	*CF_marker_offd);
   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(int_buf_data,HYPRE_MEMORY_HOST);
}

#endif

#endif /* #if defined(HYPRE_USING_CUDA) */
