#ifndef HYPRE_PARCSR_INTERFACE_INTERPRETER
#define HYPRE_PARCSR_INTERFACE_INTERPRETER

#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"


#ifdef __cplusplus
extern "C" {
#endif

int
hypre_ParCSRMultiVectorPrint( void* x_, const char* fileName );

void*
hypre_ParCSRMultiVectorRead( MPI_Comm comm, void* ii_, const char* fileName );

int
HYPRE_ParCSRSetupInterpreter( mv_InterfaceInterpreter *i );

int
HYPRE_ParCSRSetupMatvec(HYPRE_MatvecFunctions * mv);

#ifdef __cplusplus
}
#endif

#endif
