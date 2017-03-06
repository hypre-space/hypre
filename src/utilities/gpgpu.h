#ifdef HYPRE_USE_GPU
#include "gpuErrorCheck.h"
#define CUDAMEMATTACHTYPE cudaMemAttachGlobal
//#define CUDAMEMATTACHTYPE cudaMemAttachHost
#include "gpuUtils.h"
#endif
