#ifndef HYPRE_PARCSR_INTERFACE_INTERPRETER
#define HYPRE_PARCSR_INTERFACE_INTERPRETER

#include <HYPRE_config.h>

#include "parcsr_ls.h"

#include "HYPRE_interpreter.h"

#ifdef __cplusplus
extern "C" {
#endif

int
HYPRE_ParCSRSetupInterpreter( HYPRE_InterfaceInterpreter *i );

#ifdef __cplusplus
}
#endif

#endif
