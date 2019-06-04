#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "_hypre_utilities.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

/* Global struct that holds device,library handles etc */
struct hypre__global_struct hypre__global_handle = {0, 0, 1, 0};

/* Returns the same cusparse handle with every call */
cusparseHandle_t getCusparseHandle(){
  cusparseStatus_t status;
  static cusparseHandle_t handle;
  static hypre_int firstcall=1;
  if (firstcall){
    firstcall=0;
    status= cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: CUSPARSE Library initialization failed\n");
      handle=0;
      exit(2);
    }
    HYPRE_CUSPARSE_CALL(cusparseSetStream(handle,HYPRE_STREAM(4)));
  } else return handle;
  return handle;
}


nvtxDomainHandle_t getdomain(hypre_int i){
    static hypre_int firstcall=1;
    const hypre_int MAXDOMAINS=1;
    static nvtxDomainHandle_t h[MAXDOMAINS];
    if (firstcall){
      h[0]= nvtxDomainCreateA("HYPRE_A");
      firstcall=0;
    }
    if (i<MAXDOMAINS) return h[i];
   // fprintf(stderr,"ERROR in getdomain in utilities/gpuMem.c %d  is greater than MAXDOMAINS = %d \n Returning default domain",i,MAXDOMAINS);
    return NULL;
  }

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */

