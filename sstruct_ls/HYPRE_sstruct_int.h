#ifndef HYPRE_PARCSR_INTERFACE_INTERPRETER
#define HYPRE_PARCSR_INTERFACE_INTERPRETER

#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "sstruct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

int
hypre_SStructPVectorSetRandomValues( hypre_SStructPVector *pvector, int seed);

int
hypre_SStructVectorSetRandomValues( hypre_SStructVector *vector, int seed);

int
hypre_SStructSetRandomValues( void *v, int seed);

int
HYPRE_SStructSetupInterpreter( mv_InterfaceInterpreter *i );

int
HYPRE_SStructSetupMatvec(HYPRE_MatvecFunctions * mv);

#ifdef __cplusplus
}
#endif

#endif
