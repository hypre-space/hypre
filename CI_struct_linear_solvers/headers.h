#include <../HYPRE_config.h>

#include "../utilities/general.h"

#include "../utilities/utilities.h"

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "../HYPRE.h"

#include "./HYPRE_CI_struct_linear_solvers_types.h"

#include "../CI_struct_matrix_vector/HYPRE_CI_struct_matrix_vector_types.h"
#include "../CI_struct_matrix_vector/HYPRE_CI_struct_matrix_vector_protos.h"

#ifdef PETSC_AVAILABLE
#include "../PETSc_linear_solvers/ParILUT/HYPRE_PETScSolverParILUT_types.h"
#include "../PETSc_linear_solvers/ParILUT/HYPRE_PETScSolverParILUT_protos.h"
#endif

#include "./struct_solver.h"

#include "./hypre_protos.h"
#include "./internal_protos.h"
