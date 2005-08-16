#ifndef HYPRE_PARCSR_INTERFACE_INTERPRETER
#define HYPRE_PARCSR_INTERFACE_INTERPRETER

#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "struct_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

int
hypre_StructVectorSetRandomValues( hypre_StructVector *vector, int seed);

int
hypre_StructSetRandomValues( void *v, int seed);

int
HYPRE_StructSetupInterpreter( mv_InterfaceInterpreter *i );

int
HYPRE_StructSetupMatvec(HYPRE_MatvecFunctions * mv);


#ifdef __cplusplus
}
#endif

#endif
