/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

/* struct componentwise_multiply_functor */
/* { */

/*    __host__ __device__ */
/*    float operator()(const HYPRE_Complex& x, const HYPRE_Complex& y) const { */ 
/*       return x * y; */
/*    } */
/* }; */

/* void thrust_componentwise_multiply(HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Int size) */
/* { */
/*     // y <- x * y */
/*     thrust::transform(thrust::device, x, x + size, y, y, componentwise_multiply_functor()); */
/* } */





__global__
void VecScaleKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex *__restrict__ v, const HYPRE_Complex * __restrict__ l1_norm, hypre_int num_rows)
{
   HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<num_rows)
   {
      u[i]+=v[i]/l1_norm[i];
   }
}

void VecScale(HYPRE_Complex *u, HYPRE_Complex *v, HYPRE_Complex *l1_norm, hypre_int num_rows)
{
   const HYPRE_Int tpb=64;
   HYPRE_Int num_blocks=num_rows/tpb+1;
   hypre_MemPrefetch(l1_norm, sizeof(HYPRE_Complex)*num_rows, HYPRE_MEMORY_DEVICE);
   VecScaleKernel<<<num_blocks,tpb,0,hypre_HandleCudaComputeStream(hypre_handle())>>>(u,v,l1_norm,num_rows);
   hypre_SyncCudaComputeStream(hypre_handle());
}

__global__
void VecScaleMaskedKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex *__restrict__ v, const HYPRE_Complex * __restrict__ l1_norm, const HYPRE_Int *mask, hypre_int mask_size)
{
   HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<mask_size)
   {
      u[mask[i]]+=v[mask[i]]/l1_norm[mask[i]];
   }
}

void VecScaleMasked(HYPRE_Complex *u, HYPRE_Complex *v, HYPRE_Complex *l1_norm, HYPRE_Int *mask, hypre_int mask_size)
{
   const HYPRE_Int tpb=64;
   HYPRE_Int num_blocks=mask_size/tpb+1;
   hypre_MemPrefetch(l1_norm, sizeof(HYPRE_Complex)*mask_size, HYPRE_MEMORY_DEVICE);
   VecScaleMaskedKernel<<<num_blocks,tpb,0,hypre_HandleCudaComputeStream(hypre_handle())>>>(u,v,l1_norm,mask,mask_size);
   hypre_SyncCudaComputeStream(hypre_handle());
}



HYPRE_Int
hypre_BoomerAMGDD_FAC_Jacobi_device( hypre_AMGDDCompGrid *compGrid, hypre_AMGDDCompGridMatrix *A, hypre_AMGDDCompGridVector *f, hypre_AMGDDCompGridVector *u )
{
   HYPRE_Int i,j; 
   HYPRE_Real relax_weight = hypre_AMGDDCompGridRelaxWeight(compGrid);

   // Calculate l1_norms if necessary (right now, I'm just using this vector for the diagonal of A and doing straight ahead Jacobi)
   if (!hypre_AMGDDCompGridL1Norms(compGrid))
   {
      HYPRE_Int total_real_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) + hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
      hypre_AMGDDCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, total_real_nodes, hypre_AMGDDCompGridMemoryLocation(compGrid));
      hypre_CSRMatrix *diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_AMGDDCompGridL1Norms(compGrid)[i] = hypre_CSRMatrixData(diag)[j];
         }
      }
      diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] = hypre_CSRMatrixData(diag)[j];
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }

   hypre_AMGDDCompGridVectorCopy(f, hypre_AMGDDCompGridTemp2(compGrid));

   hypre_AMGDDCompGridMatvec(-relax_weight, A, u, relax_weight, hypre_AMGDDCompGridTemp2(compGrid));

   VecScale(hypre_VectorData(hypre_AMGDDCompGridVectorOwned(u)),
            hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid))),
            hypre_AMGDDCompGridL1Norms(compGrid),
            hypre_AMGDDCompGridNumOwnedNodes(compGrid));
   VecScale(hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(u)),
            hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid))),
            &(hypre_AMGDDCompGridL1Norms(compGrid)[ hypre_AMGDDCompGridNumOwnedNodes(compGrid) ]),
            hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));

   return 0;
}

HYPRE_Int
FAC_CFL1Jacobi_device( hypre_AMGDDCompGrid *compGrid, HYPRE_Int relax_set )
{
   HYPRE_Real relax_weight = hypre_AMGDDCompGridRelaxWeight(compGrid);

   // Get cusparse handle and setup bsr matrix
   static cusparseHandle_t handle;
   static cusparseMatDescr_t descr;
   static HYPRE_Int FirstCall=1;

   if (FirstCall)
   {
      handle = hypre_HandleCusparseHandle(hypre_handle());

      cusparseStatus_t status= cusparseCreateMatDescr(&descr);
      if (status != CUSPARSE_STATUS_SUCCESS) {
         hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: Matrix descriptor initialization failed\n");
         return hypre_error_flag;
      }

      cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

      FirstCall=0;
   }
   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }
   hypre_AMGDDCompGridVectorCopy(hypre_AMGDDCompGridF(compGrid), hypre_AMGDDCompGridTemp2(compGrid));
   double alpha = -relax_weight;
   double beta = relax_weight;

   HYPRE_Complex *owned_u = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(compGrid)));
   HYPRE_Complex *nonowned_u = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridU(compGrid)));
   HYPRE_Complex *owned_tmp = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid)));
   HYPRE_Complex *nonowned_tmp = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid)));

   if (relax_set)
   {
      hypre_CSRMatrix *mat = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                owned_tmp);

      mat = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                owned_tmp);

      mat = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridNonOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                nonowned_tmp);

      mat = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridNonOwnedCMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                nonowned_tmp);

      cudaDeviceSynchronize();

      VecScaleMasked(owned_u,owned_tmp,hypre_AMGDDCompGridL1Norms(compGrid),hypre_AMGDDCompGridOwnedCMask(compGrid),hypre_AMGDDCompGridNumOwnedCPoints(compGrid));
      VecScaleMasked(nonowned_u,nonowned_tmp,&(hypre_AMGDDCompGridL1Norms(compGrid)[hypre_AMGDDCompGridNumOwnedNodes(compGrid)]),hypre_AMGDDCompGridNonOwnedCMask(compGrid),hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid));
      
   }
   else
   {
      hypre_CSRMatrix *mat = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumOwnedNodes(compGrid) - hypre_AMGDDCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                owned_tmp);

      mat = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumOwnedNodes(compGrid) - hypre_AMGDDCompGridNumOwnedCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                owned_tmp);

      mat = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) - hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridNonOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                nonowned_u,
                &beta,
                nonowned_tmp);

      mat = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      cusparseDbsrxmv(handle,
                CUSPARSE_DIRECTION_ROW,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) - hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid),
                hypre_CSRMatrixNumRows(mat),
                hypre_CSRMatrixNumCols(mat),
                hypre_CSRMatrixNumNonzeros(mat),
                &alpha,
                descr,
                hypre_CSRMatrixData(mat),
                hypre_AMGDDCompGridNonOwnedFMask(compGrid),
                hypre_CSRMatrixI(mat),
                &(hypre_CSRMatrixI(mat)[1]),
                hypre_CSRMatrixJ(mat),
                1,
                owned_u,
                &beta,
                nonowned_tmp);

      cudaDeviceSynchronize();

      VecScaleMasked(owned_u,owned_tmp,hypre_AMGDDCompGridL1Norms(compGrid),hypre_AMGDDCompGridOwnedFMask(compGrid),hypre_AMGDDCompGridNumOwnedNodes(compGrid) - hypre_AMGDDCompGridNumOwnedCPoints(compGrid));
      VecScaleMasked(nonowned_u,nonowned_tmp,&(hypre_AMGDDCompGridL1Norms(compGrid)[hypre_AMGDDCompGridNumOwnedNodes(compGrid)]),hypre_AMGDDCompGridNonOwnedFMask(compGrid),hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) - hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid));
      
   }

   return 0;
}


#endif
