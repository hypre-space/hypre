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
STUBDOCS = SIDL_BaseClass.fif SIDL_BaseException.fif SIDL_BaseInterface.fif   \
  SIDL_ClassInfo.fif SIDL_ClassInfoI.fif SIDL_DLL.fif SIDL_Loader.fif         \
  bHYPRE_BoomerAMG.fif bHYPRE_CoefficientAccess.fif bHYPRE_GMRES.fif          \
  bHYPRE_IJBuildMatrix.fif bHYPRE_IJBuildVector.fif bHYPRE_IJParCSRMatrix.fif \
  bHYPRE_IJParCSRVector.fif bHYPRE_Operator.fif bHYPRE_PCG.fif                \
  bHYPRE_ParCSRDiagScale.fif bHYPRE_Pilut.fif bHYPRE_PreconditionedSolver.fif \
  bHYPRE_ProblemDefinition.fif bHYPRE_SStructBuildMatrix.fif                  \
  bHYPRE_SStructBuildVector.fif bHYPRE_SStructGraph.fif                       \
  bHYPRE_SStructGrid.fif bHYPRE_SStructMatrix.fif                             \
  bHYPRE_SStructParCSRMatrix.fif bHYPRE_SStructParCSRVector.fif               \
  bHYPRE_SStructStencil.fif bHYPRE_SStructVector.fif bHYPRE_Solver.fif        \
  bHYPRE_StructBuildMatrix.fif bHYPRE_StructBuildVector.fif                   \
  bHYPRE_StructGrid.fif bHYPRE_StructMatrix.fif bHYPRE_StructStencil.fif      \
  bHYPRE_StructVector.fif bHYPRE_Vector.fif
STUBFORTRANINC = bHYPRE_SStructVariable.inc
STUBSRCS = SIDL_BaseClass_fStub.c SIDL_BaseException_fStub.c                  \
  SIDL_BaseInterface_fStub.c SIDL_ClassInfoI_fStub.c SIDL_ClassInfo_fStub.c   \
  SIDL_DLL_fStub.c SIDL_Loader_fStub.c bHYPRE_BoomerAMG_fStub.c               \
  bHYPRE_CoefficientAccess_fStub.c bHYPRE_GMRES_fStub.c                       \
  bHYPRE_IJBuildMatrix_fStub.c bHYPRE_IJBuildVector_fStub.c                   \
  bHYPRE_IJParCSRMatrix_fStub.c bHYPRE_IJParCSRVector_fStub.c                 \
  bHYPRE_Operator_fStub.c bHYPRE_PCG_fStub.c bHYPRE_ParCSRDiagScale_fStub.c   \
  bHYPRE_Pilut_fStub.c bHYPRE_PreconditionedSolver_fStub.c                    \
  bHYPRE_ProblemDefinition_fStub.c bHYPRE_SStructBuildMatrix_fStub.c          \
  bHYPRE_SStructBuildVector_fStub.c bHYPRE_SStructGraph_fStub.c               \
  bHYPRE_SStructGrid_fStub.c bHYPRE_SStructMatrix_fStub.c                     \
  bHYPRE_SStructParCSRMatrix_fStub.c bHYPRE_SStructParCSRVector_fStub.c       \
  bHYPRE_SStructStencil_fStub.c bHYPRE_SStructVariable_fStub.c                \
  bHYPRE_SStructVector_fStub.c bHYPRE_Solver_fStub.c                          \
  bHYPRE_StructBuildMatrix_fStub.c bHYPRE_StructBuildVector_fStub.c           \
  bHYPRE_StructGrid_fStub.c bHYPRE_StructMatrix_fStub.c                       \
  bHYPRE_StructStencil_fStub.c bHYPRE_StructVector_fStub.c                    \
  bHYPRE_Vector_fStub.c
