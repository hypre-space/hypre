
#include "general.h"

#include "../utilities/memory.h"

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "mpi.h"

#include "HYPRE.h"

#include "./struct_solver.h"

#include "./hypre_protos.h"
#include "./internal_protos.h"
