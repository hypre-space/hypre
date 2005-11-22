IMPLHDRS = bHYPRE_BiCGSTAB_Impl.h bHYPRE_BoomerAMG_Impl.h bHYPRE_CGNR_Impl.h  \
  bHYPRE_Euclid_Impl.h bHYPRE_GMRES_Impl.h bHYPRE_HGMRES_Impl.h               \
  bHYPRE_HPCG_Impl.h bHYPRE_IJParCSRMatrix_Impl.h                             \
  bHYPRE_IJParCSRVector_Impl.h bHYPRE_IdentitySolver_Impl.h                   \
  bHYPRE_MPICommunicator_Impl.h bHYPRE_PCG_Impl.h                             \
  bHYPRE_ParCSRDiagScale_Impl.h bHYPRE_ParaSails_Impl.h bHYPRE_Pilut_Impl.h   \
  bHYPRE_SStructDiagScale_Impl.h bHYPRE_SStructGraph_Impl.h                   \
  bHYPRE_SStructGrid_Impl.h bHYPRE_SStructMatrix_Impl.h                       \
  bHYPRE_SStructParCSRMatrix_Impl.h bHYPRE_SStructParCSRVector_Impl.h         \
  bHYPRE_SStructStencil_Impl.h bHYPRE_SStructVector_Impl.h                    \
  bHYPRE_StructDiagScale_Impl.h bHYPRE_StructGrid_Impl.h                      \
  bHYPRE_StructMatrix_Impl.h bHYPRE_StructPFMG_Impl.h bHYPRE_StructSMG_Impl.h \
  bHYPRE_StructStencil_Impl.h bHYPRE_StructVector_Impl.h
IMPLSRCS = bHYPRE_BiCGSTAB_Impl.c bHYPRE_BoomerAMG_Impl.c bHYPRE_CGNR_Impl.c  \
  bHYPRE_Euclid_Impl.c bHYPRE_GMRES_Impl.c bHYPRE_HGMRES_Impl.c               \
  bHYPRE_HPCG_Impl.c bHYPRE_IJParCSRMatrix_Impl.c                             \
  bHYPRE_IJParCSRVector_Impl.c bHYPRE_IdentitySolver_Impl.c                   \
  bHYPRE_MPICommunicator_Impl.c bHYPRE_PCG_Impl.c                             \
  bHYPRE_ParCSRDiagScale_Impl.c bHYPRE_ParaSails_Impl.c bHYPRE_Pilut_Impl.c   \
  bHYPRE_SStructDiagScale_Impl.c bHYPRE_SStructGraph_Impl.c                   \
  bHYPRE_SStructGrid_Impl.c bHYPRE_SStructMatrix_Impl.c                       \
  bHYPRE_SStructParCSRMatrix_Impl.c bHYPRE_SStructParCSRVector_Impl.c         \
  bHYPRE_SStructStencil_Impl.c bHYPRE_SStructVector_Impl.c                    \
  bHYPRE_StructDiagScale_Impl.c bHYPRE_StructGrid_Impl.c                      \
  bHYPRE_StructMatrix_Impl.c bHYPRE_StructPFMG_Impl.c bHYPRE_StructSMG_Impl.c \
  bHYPRE_StructStencil_Impl.c bHYPRE_StructVector_Impl.c
IORHDRS = bHYPRE_BiCGSTAB_IOR.h bHYPRE_BoomerAMG_IOR.h bHYPRE_CGNR_IOR.h      \
  bHYPRE_CoefficientAccess_IOR.h bHYPRE_Euclid_IOR.h bHYPRE_GMRES_IOR.h       \
  bHYPRE_HGMRES_IOR.h bHYPRE_HPCG_IOR.h bHYPRE_IJMatrixView_IOR.h             \
  bHYPRE_IJParCSRMatrix_IOR.h bHYPRE_IJParCSRVector_IOR.h                     \
  bHYPRE_IJVectorView_IOR.h bHYPRE_IOR.h bHYPRE_IdentitySolver_IOR.h          \
  bHYPRE_MPICommunicator_IOR.h bHYPRE_MatrixVectorView_IOR.h                  \
  bHYPRE_Operator_IOR.h bHYPRE_PCG_IOR.h bHYPRE_ParCSRDiagScale_IOR.h         \
  bHYPRE_ParaSails_IOR.h bHYPRE_Pilut_IOR.h bHYPRE_PreconditionedSolver_IOR.h \
  bHYPRE_ProblemDefinition_IOR.h bHYPRE_SStructDiagScale_IOR.h                \
  bHYPRE_SStructGraph_IOR.h bHYPRE_SStructGrid_IOR.h                          \
  bHYPRE_SStructMatrixView_IOR.h bHYPRE_SStructMatrix_IOR.h                   \
  bHYPRE_SStructParCSRMatrix_IOR.h bHYPRE_SStructParCSRVector_IOR.h           \
  bHYPRE_SStructStencil_IOR.h bHYPRE_SStructVariable_IOR.h                    \
  bHYPRE_SStructVectorView_IOR.h bHYPRE_SStructVector_IOR.h                   \
  bHYPRE_SStruct_MatrixVectorView_IOR.h bHYPRE_Solver_IOR.h                   \
  bHYPRE_StructDiagScale_IOR.h bHYPRE_StructGrid_IOR.h                        \
  bHYPRE_StructMatrixView_IOR.h bHYPRE_StructMatrix_IOR.h                     \
  bHYPRE_StructPFMG_IOR.h bHYPRE_StructSMG_IOR.h bHYPRE_StructStencil_IOR.h   \
  bHYPRE_StructVectorView_IOR.h bHYPRE_StructVector_IOR.h bHYPRE_Vector_IOR.h
IORSRCS = bHYPRE_BiCGSTAB_IOR.c bHYPRE_BoomerAMG_IOR.c bHYPRE_CGNR_IOR.c      \
  bHYPRE_Euclid_IOR.c bHYPRE_GMRES_IOR.c bHYPRE_HGMRES_IOR.c                  \
  bHYPRE_HPCG_IOR.c bHYPRE_IJParCSRMatrix_IOR.c bHYPRE_IJParCSRVector_IOR.c   \
  bHYPRE_IdentitySolver_IOR.c bHYPRE_MPICommunicator_IOR.c bHYPRE_PCG_IOR.c   \
  bHYPRE_ParCSRDiagScale_IOR.c bHYPRE_ParaSails_IOR.c bHYPRE_Pilut_IOR.c      \
  bHYPRE_SStructDiagScale_IOR.c bHYPRE_SStructGraph_IOR.c                     \
  bHYPRE_SStructGrid_IOR.c bHYPRE_SStructMatrix_IOR.c                         \
  bHYPRE_SStructParCSRMatrix_IOR.c bHYPRE_SStructParCSRVector_IOR.c           \
  bHYPRE_SStructStencil_IOR.c bHYPRE_SStructVector_IOR.c                      \
  bHYPRE_StructDiagScale_IOR.c bHYPRE_StructGrid_IOR.c                        \
  bHYPRE_StructMatrix_IOR.c bHYPRE_StructPFMG_IOR.c bHYPRE_StructSMG_IOR.c    \
  bHYPRE_StructStencil_IOR.c bHYPRE_StructVector_IOR.c
SKELSRCS = bHYPRE_BiCGSTAB_Skel.c bHYPRE_BoomerAMG_Skel.c bHYPRE_CGNR_Skel.c  \
  bHYPRE_Euclid_Skel.c bHYPRE_GMRES_Skel.c bHYPRE_HGMRES_Skel.c               \
  bHYPRE_HPCG_Skel.c bHYPRE_IJParCSRMatrix_Skel.c                             \
  bHYPRE_IJParCSRVector_Skel.c bHYPRE_IdentitySolver_Skel.c                   \
  bHYPRE_MPICommunicator_Skel.c bHYPRE_PCG_Skel.c                             \
  bHYPRE_ParCSRDiagScale_Skel.c bHYPRE_ParaSails_Skel.c bHYPRE_Pilut_Skel.c   \
  bHYPRE_SStructDiagScale_Skel.c bHYPRE_SStructGraph_Skel.c                   \
  bHYPRE_SStructGrid_Skel.c bHYPRE_SStructMatrix_Skel.c                       \
  bHYPRE_SStructParCSRMatrix_Skel.c bHYPRE_SStructParCSRVector_Skel.c         \
  bHYPRE_SStructStencil_Skel.c bHYPRE_SStructVector_Skel.c                    \
  bHYPRE_StructDiagScale_Skel.c bHYPRE_StructGrid_Skel.c                      \
  bHYPRE_StructMatrix_Skel.c bHYPRE_StructPFMG_Skel.c bHYPRE_StructSMG_Skel.c \
  bHYPRE_StructStencil_Skel.c bHYPRE_StructVector_Skel.c
STUBHDRS = bHYPRE.h bHYPRE_BiCGSTAB.h bHYPRE_BoomerAMG.h bHYPRE_CGNR.h        \
  bHYPRE_CoefficientAccess.h bHYPRE_Euclid.h bHYPRE_GMRES.h bHYPRE_HGMRES.h   \
  bHYPRE_HPCG.h bHYPRE_IJMatrixView.h bHYPRE_IJParCSRMatrix.h                 \
  bHYPRE_IJParCSRVector.h bHYPRE_IJVectorView.h bHYPRE_IdentitySolver.h       \
  bHYPRE_MPICommunicator.h bHYPRE_MatrixVectorView.h bHYPRE_Operator.h        \
  bHYPRE_PCG.h bHYPRE_ParCSRDiagScale.h bHYPRE_ParaSails.h bHYPRE_Pilut.h     \
  bHYPRE_PreconditionedSolver.h bHYPRE_ProblemDefinition.h                    \
  bHYPRE_SStructDiagScale.h bHYPRE_SStructGraph.h bHYPRE_SStructGrid.h        \
  bHYPRE_SStructMatrix.h bHYPRE_SStructMatrixView.h                           \
  bHYPRE_SStructParCSRMatrix.h bHYPRE_SStructParCSRVector.h                   \
  bHYPRE_SStructStencil.h bHYPRE_SStructVariable.h bHYPRE_SStructVector.h     \
  bHYPRE_SStructVectorView.h bHYPRE_SStruct_MatrixVectorView.h                \
  bHYPRE_Solver.h bHYPRE_StructDiagScale.h bHYPRE_StructGrid.h                \
  bHYPRE_StructMatrix.h bHYPRE_StructMatrixView.h bHYPRE_StructPFMG.h         \
  bHYPRE_StructSMG.h bHYPRE_StructStencil.h bHYPRE_StructVector.h             \
  bHYPRE_StructVectorView.h bHYPRE_Vector.h
STUBSRCS = bHYPRE_BiCGSTAB_Stub.c bHYPRE_BoomerAMG_Stub.c bHYPRE_CGNR_Stub.c  \
  bHYPRE_CoefficientAccess_Stub.c bHYPRE_Euclid_Stub.c bHYPRE_GMRES_Stub.c    \
  bHYPRE_HGMRES_Stub.c bHYPRE_HPCG_Stub.c bHYPRE_IJMatrixView_Stub.c          \
  bHYPRE_IJParCSRMatrix_Stub.c bHYPRE_IJParCSRVector_Stub.c                   \
  bHYPRE_IJVectorView_Stub.c bHYPRE_IdentitySolver_Stub.c                     \
  bHYPRE_MPICommunicator_Stub.c bHYPRE_MatrixVectorView_Stub.c                \
  bHYPRE_Operator_Stub.c bHYPRE_PCG_Stub.c bHYPRE_ParCSRDiagScale_Stub.c      \
  bHYPRE_ParaSails_Stub.c bHYPRE_Pilut_Stub.c                                 \
  bHYPRE_PreconditionedSolver_Stub.c bHYPRE_ProblemDefinition_Stub.c          \
  bHYPRE_SStructDiagScale_Stub.c bHYPRE_SStructGraph_Stub.c                   \
  bHYPRE_SStructGrid_Stub.c bHYPRE_SStructMatrixView_Stub.c                   \
  bHYPRE_SStructMatrix_Stub.c bHYPRE_SStructParCSRMatrix_Stub.c               \
  bHYPRE_SStructParCSRVector_Stub.c bHYPRE_SStructStencil_Stub.c              \
  bHYPRE_SStructVariable_Stub.c bHYPRE_SStructVectorView_Stub.c               \
  bHYPRE_SStructVector_Stub.c bHYPRE_SStruct_MatrixVectorView_Stub.c          \
  bHYPRE_Solver_Stub.c bHYPRE_StructDiagScale_Stub.c bHYPRE_StructGrid_Stub.c \
  bHYPRE_StructMatrixView_Stub.c bHYPRE_StructMatrix_Stub.c                   \
  bHYPRE_StructPFMG_Stub.c bHYPRE_StructSMG_Stub.c                            \
  bHYPRE_StructStencil_Stub.c bHYPRE_StructVectorView_Stub.c                  \
  bHYPRE_StructVector_Stub.c bHYPRE_Vector_Stub.c
