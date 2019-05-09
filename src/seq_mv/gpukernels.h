#ifdef HYPRE_USING_GPU
#include <cuda_runtime_api.h>
HYPRE_Int VecScaleScalar(HYPRE_Real *u, const HYPRE_Real alpha,  HYPRE_Int num_rows,cudaStream_t s);
void DiagScaleVector(HYPRE_Real *x, HYPRE_Real *y, HYPRE_Real *A_data, HYPRE_Int *A_i, HYPRE_Int num_rows, cudaStream_t s);
void VecCopy(HYPRE_Real* tgt, const HYPRE_Real* src, HYPRE_Int size,cudaStream_t s);
void VecSet(HYPRE_Real* tgt, HYPRE_Int size, HYPRE_Real value, cudaStream_t s);
void VecScale(HYPRE_Real *u, HYPRE_Real *v, HYPRE_Real *l1_norm, HYPRE_Int num_rows,cudaStream_t s);
void VecScaleSplit(HYPRE_Real *u, HYPRE_Real *v, HYPRE_Real *l1_norm, HYPRE_Int num_rows,cudaStream_t s);
void CudaCompileFlagCheck();
void BigToSmallCopy(HYPRE_Int *tgt, const HYPRE_BigInt* src, HYPRE_Int size, cudaStream_t s);
#endif
