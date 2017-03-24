#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#include "gpuErrorCheck.h"
#define CUDAMEMATTACHTYPE cudaMemAttachGlobal
//#define CUDAMEMATTACHTYPE cudaMemAttachHost
#define HYPRE_GPU_USE_PINNED 1
#define HYPRE_USE_MANAGED_SCALABLE 1
#endif
