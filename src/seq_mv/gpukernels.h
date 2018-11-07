#ifdef HYPRE_USING_GPU
#include <cuda_runtime_api.h>
int VecScaleScalar(double *u, const double alpha,  int num_rows,cudaStream_t s);
void DiagScaleVector(double *x, double *y, double *A_data, int *A_i, int num_rows, cudaStream_t s);
void VecCopy(double* tgt, const double* src, int size,cudaStream_t s);
void VecSet(double* tgt, int size, double value, cudaStream_t s);
void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void VecScaleSplit(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void CudaCompileFlagCheck();
void BigToSmallCopy(hypre_int *tgt, const HYPRE_Int* src, hypre_int size, cudaStream_t s);
#endif
