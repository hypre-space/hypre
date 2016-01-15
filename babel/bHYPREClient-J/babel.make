IORHDRS = bHYPRE_BiCGSTAB_IOR.h bHYPRE_BoomerAMG_IOR.h bHYPRE_CGNR_IOR.h      \
  bHYPRE_CoefficientAccess_IOR.h bHYPRE_ErrorCode_IOR.h                       \
  bHYPRE_ErrorHandler_IOR.h bHYPRE_Euclid_IOR.h bHYPRE_GMRES_IOR.h            \
  bHYPRE_HGMRES_IOR.h bHYPRE_HPCG_IOR.h bHYPRE_Hybrid_IOR.h                   \
  bHYPRE_IJMatrixView_IOR.h bHYPRE_IJParCSRMatrix_IOR.h                       \
  bHYPRE_IJParCSRVector_IOR.h bHYPRE_IJVectorView_IOR.h bHYPRE_IOR.h          \
  bHYPRE_IdentitySolver_IOR.h bHYPRE_MPICommunicator_IOR.h                    \
  bHYPRE_MatrixVectorView_IOR.h bHYPRE_Operator_IOR.h bHYPRE_PCG_IOR.h        \
  bHYPRE_ParCSRDiagScale_IOR.h bHYPRE_ParaSails_IOR.h bHYPRE_Pilut_IOR.h      \
  bHYPRE_PreconditionedSolver_IOR.h bHYPRE_ProblemDefinition_IOR.h            \
  bHYPRE_SStructDiagScale_IOR.h bHYPRE_SStructGraph_IOR.h                     \
  bHYPRE_SStructGrid_IOR.h bHYPRE_SStructMatrixVectorView_IOR.h               \
  bHYPRE_SStructMatrixView_IOR.h bHYPRE_SStructMatrix_IOR.h                   \
  bHYPRE_SStructParCSRMatrix_IOR.h bHYPRE_SStructParCSRVector_IOR.h           \
  bHYPRE_SStructSplit_IOR.h bHYPRE_SStructStencil_IOR.h                       \
  bHYPRE_SStructVariable_IOR.h bHYPRE_SStructVectorView_IOR.h                 \
  bHYPRE_SStructVector_IOR.h bHYPRE_Schwarz_IOR.h bHYPRE_Solver_IOR.h         \
  bHYPRE_StructDiagScale_IOR.h bHYPRE_StructGrid_IOR.h                        \
  bHYPRE_StructJacobi_IOR.h bHYPRE_StructMatrixView_IOR.h                     \
  bHYPRE_StructMatrix_IOR.h bHYPRE_StructPFMG_IOR.h bHYPRE_StructSMG_IOR.h    \
  bHYPRE_StructStencil_IOR.h bHYPRE_StructVectorView_IOR.h                    \
  bHYPRE_StructVector_IOR.h bHYPRE_Vector_IOR.h
STUBHDRS = bHYPRE_BiCGSTAB_jniStub.h bHYPRE_BoomerAMG_jniStub.h               \
  bHYPRE_CGNR_jniStub.h bHYPRE_CoefficientAccess_jniStub.h                    \
  bHYPRE_ErrorHandler_jniStub.h bHYPRE_Euclid_jniStub.h                       \
  bHYPRE_GMRES_jniStub.h bHYPRE_HGMRES_jniStub.h bHYPRE_HPCG_jniStub.h        \
  bHYPRE_Hybrid_jniStub.h bHYPRE_IJMatrixView_jniStub.h                       \
  bHYPRE_IJParCSRMatrix_jniStub.h bHYPRE_IJParCSRVector_jniStub.h             \
  bHYPRE_IJVectorView_jniStub.h bHYPRE_IdentitySolver_jniStub.h               \
  bHYPRE_MPICommunicator_jniStub.h bHYPRE_MatrixVectorView_jniStub.h          \
  bHYPRE_Operator_jniStub.h bHYPRE_PCG_jniStub.h                              \
  bHYPRE_ParCSRDiagScale_jniStub.h bHYPRE_ParaSails_jniStub.h                 \
  bHYPRE_Pilut_jniStub.h bHYPRE_PreconditionedSolver_jniStub.h                \
  bHYPRE_ProblemDefinition_jniStub.h bHYPRE_SStructDiagScale_jniStub.h        \
  bHYPRE_SStructGraph_jniStub.h bHYPRE_SStructGrid_jniStub.h                  \
  bHYPRE_SStructMatrixVectorView_jniStub.h bHYPRE_SStructMatrixView_jniStub.h \
  bHYPRE_SStructMatrix_jniStub.h bHYPRE_SStructParCSRMatrix_jniStub.h         \
  bHYPRE_SStructParCSRVector_jniStub.h bHYPRE_SStructSplit_jniStub.h          \
  bHYPRE_SStructStencil_jniStub.h bHYPRE_SStructVectorView_jniStub.h          \
  bHYPRE_SStructVector_jniStub.h bHYPRE_Schwarz_jniStub.h                     \
  bHYPRE_Solver_jniStub.h bHYPRE_StructDiagScale_jniStub.h                    \
  bHYPRE_StructGrid_jniStub.h bHYPRE_StructJacobi_jniStub.h                   \
  bHYPRE_StructMatrixView_jniStub.h bHYPRE_StructMatrix_jniStub.h             \
  bHYPRE_StructPFMG_jniStub.h bHYPRE_StructSMG_jniStub.h                      \
  bHYPRE_StructStencil_jniStub.h bHYPRE_StructVectorView_jniStub.h            \
  bHYPRE_StructVector_jniStub.h bHYPRE_Vector_jniStub.h
STUBSRCS = bHYPRE_BiCGSTAB_jniStub.c bHYPRE_BoomerAMG_jniStub.c               \
  bHYPRE_CGNR_jniStub.c bHYPRE_CoefficientAccess_jniStub.c                    \
  bHYPRE_ErrorHandler_jniStub.c bHYPRE_Euclid_jniStub.c                       \
  bHYPRE_GMRES_jniStub.c bHYPRE_HGMRES_jniStub.c bHYPRE_HPCG_jniStub.c        \
  bHYPRE_Hybrid_jniStub.c bHYPRE_IJMatrixView_jniStub.c                       \
  bHYPRE_IJParCSRMatrix_jniStub.c bHYPRE_IJParCSRVector_jniStub.c             \
  bHYPRE_IJVectorView_jniStub.c bHYPRE_IdentitySolver_jniStub.c               \
  bHYPRE_MPICommunicator_jniStub.c bHYPRE_MatrixVectorView_jniStub.c          \
  bHYPRE_Operator_jniStub.c bHYPRE_PCG_jniStub.c                              \
  bHYPRE_ParCSRDiagScale_jniStub.c bHYPRE_ParaSails_jniStub.c                 \
  bHYPRE_Pilut_jniStub.c bHYPRE_PreconditionedSolver_jniStub.c                \
  bHYPRE_ProblemDefinition_jniStub.c bHYPRE_SStructDiagScale_jniStub.c        \
  bHYPRE_SStructGraph_jniStub.c bHYPRE_SStructGrid_jniStub.c                  \
  bHYPRE_SStructMatrixVectorView_jniStub.c bHYPRE_SStructMatrixView_jniStub.c \
  bHYPRE_SStructMatrix_jniStub.c bHYPRE_SStructParCSRMatrix_jniStub.c         \
  bHYPRE_SStructParCSRVector_jniStub.c bHYPRE_SStructSplit_jniStub.c          \
  bHYPRE_SStructStencil_jniStub.c bHYPRE_SStructVectorView_jniStub.c          \
  bHYPRE_SStructVector_jniStub.c bHYPRE_Schwarz_jniStub.c                     \
  bHYPRE_Solver_jniStub.c bHYPRE_StructDiagScale_jniStub.c                    \
  bHYPRE_StructGrid_jniStub.c bHYPRE_StructJacobi_jniStub.c                   \
  bHYPRE_StructMatrixView_jniStub.c bHYPRE_StructMatrix_jniStub.c             \
  bHYPRE_StructPFMG_jniStub.c bHYPRE_StructSMG_jniStub.c                      \
  bHYPRE_StructStencil_jniStub.c bHYPRE_StructVectorView_jniStub.c            \
  bHYPRE_StructVector_jniStub.c bHYPRE_Vector_jniStub.c
