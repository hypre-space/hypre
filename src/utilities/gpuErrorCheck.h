#ifdef HYPRE_USE_GPU
#ifndef __cusparseErrorCheck__
#define __cusparseErrorCheck__
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
inline const char *cusparseErrorCheck(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:
	    return "Some new error";
    }

    return "Congrats::Undefined ERRROR";
}
inline const char *cublasErrorCheck(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
	    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
	    return "Some new error";
    }

    return "Congrats::This cannot possibly be happening";
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
     fprintf(stderr,"CUDA ERROR ( Code = %d) in line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUDA ERROR : %s \n", cudaGetErrorString(code));
     exit(2);
   }
}
#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
     fprintf(stderr,"CUSPARSE ERROR  ( Code = %d) IN CUDA CALL line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUSPARSE ERROR : %s \n", cusparseErrorCheck(code));
   }
}
#define cublasErrchk(ans){ cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
     fprintf(stderr,"CUBLAS ERROR  ( Code = %d) IN CUDA CALL line %d of file %s\n",code,line,file);
     fprintf(stderr,"CUBLAS ERROR : %s \n", cublasErrorCheck(code));
   }
}
//int PointerType(const void *ptr);
void cudaSafeFree(void *ptr);
void PrintPointerAttributes(void *ptr);
//size_t mempush(void* ptr, size_t size,int purge);
//int memloc(void *ptr, int device);
#endif
#endif
