#if defined(HYPRE_USE_GPU) || defined(HYPRE_USE_MANAGED)
#include "gpuErrorCheck.h"
#define CUDAMEMATTACHTYPE cudaMemAttachGlobal
//#define CUDAMEMATTACHTYPE cudaMemAttachHost
#endif
