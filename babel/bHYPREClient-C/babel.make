IORHDRS = bHYPRE_BoomerAMG_IOR.h bHYPRE_CoefficientAccess_IOR.h               \
  bHYPRE_GMRES_IOR.h bHYPRE_IJBuildMatrix_IOR.h bHYPRE_IJBuildVector_IOR.h    \
  bHYPRE_IJParCSRMatrix_IOR.h bHYPRE_IJParCSRVector_IOR.h bHYPRE_IOR.h        \
  bHYPRE_Operator_IOR.h bHYPRE_PCG_IOR.h bHYPRE_ParCSRDiagScale_IOR.h         \
  bHYPRE_Pilut_IOR.h bHYPRE_PreconditionedSolver_IOR.h                        \
  bHYPRE_ProblemDefinition_IOR.h bHYPRE_SStructBuildMatrix_IOR.h              \
  bHYPRE_SStructBuildVector_IOR.h bHYPRE_SStructGraph_IOR.h                   \
  bHYPRE_SStructGrid_IOR.h bHYPRE_SStructMatrix_IOR.h                         \
  bHYPRE_SStructParCSRMatrix_IOR.h bHYPRE_SStructParCSRVector_IOR.h           \
  bHYPRE_SStructStencil_IOR.h bHYPRE_SStructVariable_IOR.h                    \
  bHYPRE_SStructVector_IOR.h bHYPRE_Solver_IOR.h                              \
  bHYPRE_StructBuildMatrix_IOR.h bHYPRE_StructBuildVector_IOR.h               \
  bHYPRE_StructGrid_IOR.h bHYPRE_StructMatrix_IOR.h                           \
  bHYPRE_StructStencil_IOR.h bHYPRE_StructVector_IOR.h bHYPRE_Vector_IOR.h
IORSRCS = bHYPRE_BoomerAMG_IOR.c bHYPRE_CoefficientAccess_IOR.c               \
  bHYPRE_GMRES_IOR.c bHYPRE_IJBuildMatrix_IOR.c bHYPRE_IJBuildVector_IOR.c    \
  bHYPRE_IJParCSRMatrix_IOR.c bHYPRE_IJParCSRVector_IOR.c                     \
  bHYPRE_Operator_IOR.c bHYPRE_PCG_IOR.c bHYPRE_ParCSRDiagScale_IOR.c         \
  bHYPRE_Pilut_IOR.c bHYPRE_PreconditionedSolver_IOR.c                        \
  bHYPRE_ProblemDefinition_IOR.c bHYPRE_SStructBuildMatrix_IOR.c              \
  bHYPRE_SStructBuildVector_IOR.c bHYPRE_SStructGraph_IOR.c                   \
  bHYPRE_SStructGrid_IOR.c bHYPRE_SStructMatrix_IOR.c                         \
  bHYPRE_SStructParCSRMatrix_IOR.c bHYPRE_SStructParCSRVector_IOR.c           \
  bHYPRE_SStructStencil_IOR.c                     \
  bHYPRE_SStructVector_IOR.c bHYPRE_Solver_IOR.c                              \
  bHYPRE_StructBuildMatrix_IOR.c bHYPRE_StructBuildVector_IOR.c               \
  bHYPRE_StructGrid_IOR.c bHYPRE_StructMatrix_IOR.c                           \
  bHYPRE_StructStencil_IOR.c bHYPRE_StructVector_IOR.c bHYPRE_Vector_IOR.c
STUBHDRS = bHYPRE.h bHYPRE_BoomerAMG.h bHYPRE_CoefficientAccess.h             \
  bHYPRE_GMRES.h bHYPRE_IJBuildMatrix.h bHYPRE_IJBuildVector.h                \
  bHYPRE_IJParCSRMatrix.h bHYPRE_IJParCSRVector.h bHYPRE_Operator.h           \
  bHYPRE_PCG.h bHYPRE_ParCSRDiagScale.h bHYPRE_Pilut.h                        \
  bHYPRE_PreconditionedSolver.h bHYPRE_ProblemDefinition.h                    \
  bHYPRE_SStructBuildMatrix.h bHYPRE_SStructBuildVector.h                     \
  bHYPRE_SStructGraph.h bHYPRE_SStructGrid.h bHYPRE_SStructMatrix.h           \
  bHYPRE_SStructParCSRMatrix.h bHYPRE_SStructParCSRVector.h                   \
  bHYPRE_SStructStencil.h bHYPRE_SStructVariable.h bHYPRE_SStructVector.h     \
  bHYPRE_Solver.h bHYPRE_StructBuildMatrix.h bHYPRE_StructBuildVector.h       \
  bHYPRE_StructGrid.h bHYPRE_StructMatrix.h bHYPRE_StructStencil.h            \
  bHYPRE_StructVector.h bHYPRE_Vector.h
STUBSRCS = bHYPRE_BoomerAMG_Stub.c bHYPRE_CoefficientAccess_Stub.c            \
  bHYPRE_GMRES_Stub.c bHYPRE_IJBuildMatrix_Stub.c bHYPRE_IJBuildVector_Stub.c \
  bHYPRE_IJParCSRMatrix_Stub.c bHYPRE_IJParCSRVector_Stub.c                   \
  bHYPRE_Operator_Stub.c bHYPRE_PCG_Stub.c bHYPRE_ParCSRDiagScale_Stub.c      \
  bHYPRE_Pilut_Stub.c bHYPRE_PreconditionedSolver_Stub.c                      \
  bHYPRE_ProblemDefinition_Stub.c bHYPRE_SStructBuildMatrix_Stub.c            \
  bHYPRE_SStructBuildVector_Stub.c bHYPRE_SStructGraph_Stub.c                 \
  bHYPRE_SStructGrid_Stub.c bHYPRE_SStructMatrix_Stub.c                       \
  bHYPRE_SStructParCSRMatrix_Stub.c bHYPRE_SStructParCSRVector_Stub.c         \
  bHYPRE_SStructStencil_Stub.c bHYPRE_SStructVariable_Stub.c                  \
  bHYPRE_SStructVector_Stub.c bHYPRE_Solver_Stub.c                            \
  bHYPRE_StructBuildMatrix_Stub.c bHYPRE_StructBuildVector_Stub.c             \
  bHYPRE_StructGrid_Stub.c bHYPRE_StructMatrix_Stub.c                         \
  bHYPRE_StructStencil_Stub.c bHYPRE_StructVector_Stub.c bHYPRE_Vector_Stub.c
