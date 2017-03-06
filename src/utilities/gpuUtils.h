#ifdef HYPRE_USE_GPU
#include "nvToolsExt.h"
size_t mempush(const void *ptr, size_t size,int purge);
int memloc(const void *ptr, int device);
cudaStream_t getstream(int i);
nvtxDomainHandle_t getdomain(int i);
#endif
