#include <HYPRE_config.h>

#include "general.h"

#include "utilities.h"

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "HYPRE.h"

#include "HYPRE_CI_struct_linear_solvers_types.h"

#include "HYPRE_CI_struct_matrix_vector_types.h"
#include "HYPRE_CI_struct_matrix_vector_protos.h"

#ifdef PETSC_AVAILABLE
#include "HYPRE_PETScSolverParILUT_types.h"
#include "HYPRE_PETScSolverParILUT_protos.h"
#endif

#include "struct_solver.h"

#include "hypre_protos.h"
#include "internal_protos.h"
