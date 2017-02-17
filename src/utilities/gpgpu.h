#ifdef HYPRE_USE_GPU
#include "gpuErrorCheck.h"
#define CUDAMEMATTACHTYPE cudaMemAttachGlobal
#include "gpuUtils.h"
#endif
