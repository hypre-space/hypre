
#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "general.h"

/* Solver structure from PETSc */
#include "sles.h"

#include "../HYPRE.h"

/* Prototypes for DistributedMatrix */
#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"
