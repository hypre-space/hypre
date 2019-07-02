#ifdef HYPRE_USING_GPU
#include <cuda_runtime_api.h>
HYPRE_Int VecScaleScalar(HYPRE_Complex *u, const HYPRE_Complex alpha,  HYPRE_Int num_rows,cudaStream_t s);
void DiagScaleVector(HYPRE_Complex *x, HYPRE_Complex *y, HYPRE_Complex *A_data, HYPRE_Int *A_i, HYPRE_Int num_rows, cudaStream_t s);
void VecCopy(HYPRE_Complex* tgt, const HYPRE_Complex* src, HYPRE_Int size,cudaStream_t s);
void VecSet(HYPRE_Complex* tgt, HYPRE_Int size, HYPRE_Complex value, cudaStream_t s);
void VecScale(HYPRE_Complex *u, HYPRE_Complex *v, HYPRE_Complex *l1_norm, HYPRE_Int num_rows,cudaStream_t s);
void VecScaleSplit(HYPRE_Complex *u, HYPRE_Complex *v, HYPRE_Complex *l1_norm, HYPRE_Int num_rows,cudaStream_t s);
void CudaCompileFlagCheck();
void BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt* src, HYPRE_Int size, cudaStream_t s);
#endif
