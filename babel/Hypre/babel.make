IMPLHDRS = Hypre_BoomerAMG_Impl.h Hypre_GMRES_Impl.h                          \
  Hypre_IJParCSRMatrix_Impl.h Hypre_IJParCSRVector_Impl.h Hypre_PCG_Impl.h    \
  Hypre_ParCSRDiagScale_Impl.h Hypre_Pilut_Impl.h Hypre_SStructGraph_Impl.h   \
  Hypre_SStructGrid_Impl.h Hypre_SStructMatrix_Impl.h                         \
  Hypre_SStructParCSRMatrix_Impl.h Hypre_SStructParCSRVector_Impl.h           \
  Hypre_SStructStencil_Impl.h Hypre_SStructVector_Impl.h                      \
  Hypre_StructGrid_Impl.h Hypre_StructMatrix_Impl.h                           \
  Hypre_StructStencil_Impl.h Hypre_StructVector_Impl.h
IMPLSRCS = Hypre_BoomerAMG_Impl.c Hypre_GMRES_Impl.c                          \
  Hypre_IJParCSRMatrix_Impl.c Hypre_IJParCSRVector_Impl.c Hypre_PCG_Impl.c    \
  Hypre_ParCSRDiagScale_Impl.c Hypre_Pilut_Impl.c Hypre_SStructGraph_Impl.c   \
  Hypre_SStructGrid_Impl.c Hypre_SStructMatrix_Impl.c                         \
  Hypre_SStructParCSRMatrix_Impl.c Hypre_SStructParCSRVector_Impl.c           \
  Hypre_SStructStencil_Impl.c Hypre_SStructVector_Impl.c                      \
  Hypre_StructGrid_Impl.c Hypre_StructMatrix_Impl.c                           \
  Hypre_StructStencil_Impl.c Hypre_StructVector_Impl.c
IORHDRS = Hypre_BoomerAMG_IOR.h Hypre_CoefficientAccess_IOR.h                 \
  Hypre_GMRES_IOR.h Hypre_IJBuildMatrix_IOR.h Hypre_IJBuildVector_IOR.h       \
  Hypre_IJParCSRMatrix_IOR.h Hypre_IJParCSRVector_IOR.h Hypre_IOR.h           \
  Hypre_Operator_IOR.h Hypre_PCG_IOR.h Hypre_ParCSRDiagScale_IOR.h            \
  Hypre_Pilut_IOR.h Hypre_PreconditionedSolver_IOR.h                          \
  Hypre_ProblemDefinition_IOR.h Hypre_SStructBuildMatrix_IOR.h                \
  Hypre_SStructBuildVector_IOR.h Hypre_SStructGraph_IOR.h                     \
  Hypre_SStructGrid_IOR.h Hypre_SStructMatrix_IOR.h                           \
  Hypre_SStructParCSRMatrix_IOR.h Hypre_SStructParCSRVector_IOR.h             \
  Hypre_SStructStencil_IOR.h Hypre_SStructVariable_IOR.h                      \
  Hypre_SStructVector_IOR.h Hypre_Solver_IOR.h Hypre_StructBuildMatrix_IOR.h  \
  Hypre_StructBuildVector_IOR.h Hypre_StructGrid_IOR.h                        \
  Hypre_StructMatrix_IOR.h Hypre_StructStencil_IOR.h Hypre_StructVector_IOR.h \
  Hypre_Vector_IOR.h
IORSRCS = Hypre_BoomerAMG_IOR.c Hypre_CoefficientAccess_IOR.c                 \
  Hypre_GMRES_IOR.c Hypre_IJBuildMatrix_IOR.c Hypre_IJBuildVector_IOR.c       \
  Hypre_IJParCSRMatrix_IOR.c Hypre_IJParCSRVector_IOR.c Hypre_Operator_IOR.c  \
  Hypre_PCG_IOR.c Hypre_ParCSRDiagScale_IOR.c Hypre_Pilut_IOR.c               \
  Hypre_PreconditionedSolver_IOR.c Hypre_ProblemDefinition_IOR.c              \
  Hypre_SStructBuildMatrix_IOR.c Hypre_SStructBuildVector_IOR.c               \
  Hypre_SStructGraph_IOR.c Hypre_SStructGrid_IOR.c Hypre_SStructMatrix_IOR.c  \
  Hypre_SStructParCSRMatrix_IOR.c Hypre_SStructParCSRVector_IOR.c             \
  Hypre_SStructStencil_IOR.c                       \
  Hypre_SStructVector_IOR.c Hypre_Solver_IOR.c Hypre_StructBuildMatrix_IOR.c  \
  Hypre_StructBuildVector_IOR.c Hypre_StructGrid_IOR.c                        \
  Hypre_StructMatrix_IOR.c Hypre_StructStencil_IOR.c Hypre_StructVector_IOR.c \
  Hypre_Vector_IOR.c
SKELSRCS = Hypre_BoomerAMG_Skel.c Hypre_GMRES_Skel.c                          \
  Hypre_IJParCSRMatrix_Skel.c Hypre_IJParCSRVector_Skel.c Hypre_PCG_Skel.c    \
  Hypre_ParCSRDiagScale_Skel.c Hypre_Pilut_Skel.c Hypre_SStructGraph_Skel.c   \
  Hypre_SStructGrid_Skel.c Hypre_SStructMatrix_Skel.c                         \
  Hypre_SStructParCSRMatrix_Skel.c Hypre_SStructParCSRVector_Skel.c           \
  Hypre_SStructStencil_Skel.c Hypre_SStructVector_Skel.c                      \
  Hypre_StructGrid_Skel.c Hypre_StructMatrix_Skel.c                           \
  Hypre_StructStencil_Skel.c Hypre_StructVector_Skel.c
STUBHDRS = Hypre.h Hypre_BoomerAMG.h Hypre_CoefficientAccess.h Hypre_GMRES.h  \
  Hypre_IJBuildMatrix.h Hypre_IJBuildVector.h Hypre_IJParCSRMatrix.h          \
  Hypre_IJParCSRVector.h Hypre_Operator.h Hypre_PCG.h Hypre_ParCSRDiagScale.h \
  Hypre_Pilut.h Hypre_PreconditionedSolver.h Hypre_ProblemDefinition.h        \
  Hypre_SStructBuildMatrix.h Hypre_SStructBuildVector.h Hypre_SStructGraph.h  \
  Hypre_SStructGrid.h Hypre_SStructMatrix.h Hypre_SStructParCSRMatrix.h       \
  Hypre_SStructParCSRVector.h Hypre_SStructStencil.h Hypre_SStructVariable.h  \
  Hypre_SStructVector.h Hypre_Solver.h Hypre_StructBuildMatrix.h              \
  Hypre_StructBuildVector.h Hypre_StructGrid.h Hypre_StructMatrix.h           \
  Hypre_StructStencil.h Hypre_StructVector.h Hypre_Vector.h
STUBSRCS = Hypre_BoomerAMG_Stub.c Hypre_CoefficientAccess_Stub.c              \
  Hypre_GMRES_Stub.c Hypre_IJBuildMatrix_Stub.c Hypre_IJBuildVector_Stub.c    \
  Hypre_IJParCSRMatrix_Stub.c Hypre_IJParCSRVector_Stub.c                     \
  Hypre_Operator_Stub.c Hypre_PCG_Stub.c Hypre_ParCSRDiagScale_Stub.c         \
  Hypre_Pilut_Stub.c Hypre_PreconditionedSolver_Stub.c                        \
  Hypre_ProblemDefinition_Stub.c Hypre_SStructBuildMatrix_Stub.c              \
  Hypre_SStructBuildVector_Stub.c Hypre_SStructGraph_Stub.c                   \
  Hypre_SStructGrid_Stub.c Hypre_SStructMatrix_Stub.c                         \
  Hypre_SStructParCSRMatrix_Stub.c Hypre_SStructParCSRVector_Stub.c           \
  Hypre_SStructStencil_Stub.c Hypre_SStructVariable_Stub.c                    \
  Hypre_SStructVector_Stub.c Hypre_Solver_Stub.c                              \
  Hypre_StructBuildMatrix_Stub.c Hypre_StructBuildVector_Stub.c               \
  Hypre_StructGrid_Stub.c Hypre_StructMatrix_Stub.c                           \
  Hypre_StructStencil_Stub.c Hypre_StructVector_Stub.c Hypre_Vector_Stub.c
