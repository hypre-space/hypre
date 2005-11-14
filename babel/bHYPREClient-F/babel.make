IORHDRS = bHYPRE_BoomerAMG_IOR.h bHYPRE_CoefficientAccess_IOR.h               \
  bHYPRE_GMRES_IOR.h bHYPRE_HPCG_IOR.h bHYPRE_IJMatrixView_IOR.h              \
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
STUBDOCS = bHYPRE_BoomerAMG.fif bHYPRE_CoefficientAccess.fif bHYPRE_GMRES.fif \
  bHYPRE_HPCG.fif bHYPRE_IJMatrixView.fif bHYPRE_IJParCSRMatrix.fif           \
  bHYPRE_IJParCSRVector.fif bHYPRE_IJVectorView.fif bHYPRE_IdentitySolver.fif \
  bHYPRE_MPICommunicator.fif bHYPRE_MatrixVectorView.fif bHYPRE_Operator.fif  \
  bHYPRE_PCG.fif bHYPRE_ParCSRDiagScale.fif bHYPRE_ParaSails.fif              \
  bHYPRE_Pilut.fif bHYPRE_PreconditionedSolver.fif                            \
  bHYPRE_ProblemDefinition.fif bHYPRE_SStructDiagScale.fif                    \
  bHYPRE_SStructGraph.fif bHYPRE_SStructGrid.fif bHYPRE_SStructMatrix.fif     \
  bHYPRE_SStructMatrixView.fif bHYPRE_SStructParCSRMatrix.fif                 \
  bHYPRE_SStructParCSRVector.fif bHYPRE_SStructStencil.fif                    \
  bHYPRE_SStructVector.fif bHYPRE_SStructVectorView.fif                       \
  bHYPRE_SStruct_MatrixVectorView.fif bHYPRE_Solver.fif                       \
  bHYPRE_StructDiagScale.fif bHYPRE_StructGrid.fif bHYPRE_StructMatrix.fif    \
  bHYPRE_StructMatrixView.fif bHYPRE_StructPFMG.fif bHYPRE_StructSMG.fif      \
  bHYPRE_StructStencil.fif bHYPRE_StructVector.fif                            \
  bHYPRE_StructVectorView.fif bHYPRE_Vector.fif sidl_BaseClass.fif            \
  sidl_BaseException.fif sidl_BaseInterface.fif sidl_ClassInfo.fif            \
  sidl_ClassInfoI.fif sidl_DFinder.fif sidl_DLL.fif sidl_Finder.fif           \
  sidl_InvViolation.fif sidl_Loader.fif sidl_PostViolation.fif                \
  sidl_PreViolation.fif sidl_SIDLException.fif sidl_io_Deserializer.fif       \
  sidl_io_IOException.fif sidl_io_Serializeable.fif sidl_io_Serializer.fif    \
  sidl_rmi_ConnectRegistry.fif sidl_rmi_InstanceHandle.fif                    \
  sidl_rmi_InstanceRegistry.fif sidl_rmi_Invocation.fif                       \
  sidl_rmi_NetworkException.fif sidl_rmi_ProtocolFactory.fif                  \
  sidl_rmi_Response.fif
STUBFORTRANINC = bHYPRE_SStructVariable.inc sidl_Resolve.inc sidl_Scope.inc
STUBSRCS = bHYPRE_BoomerAMG_fStub.c bHYPRE_CoefficientAccess_fStub.c          \
  bHYPRE_GMRES_fStub.c bHYPRE_HPCG_fStub.c bHYPRE_IJMatrixView_fStub.c        \
  bHYPRE_IJParCSRMatrix_fStub.c bHYPRE_IJParCSRVector_fStub.c                 \
  bHYPRE_IJVectorView_fStub.c bHYPRE_IdentitySolver_fStub.c                   \
  bHYPRE_MPICommunicator_fStub.c bHYPRE_MatrixVectorView_fStub.c              \
  bHYPRE_Operator_fStub.c bHYPRE_PCG_fStub.c bHYPRE_ParCSRDiagScale_fStub.c   \
  bHYPRE_ParaSails_fStub.c bHYPRE_Pilut_fStub.c                               \
  bHYPRE_PreconditionedSolver_fStub.c bHYPRE_ProblemDefinition_fStub.c        \
  bHYPRE_SStructDiagScale_fStub.c bHYPRE_SStructGraph_fStub.c                 \
  bHYPRE_SStructGrid_fStub.c bHYPRE_SStructMatrixView_fStub.c                 \
  bHYPRE_SStructMatrix_fStub.c bHYPRE_SStructParCSRMatrix_fStub.c             \
  bHYPRE_SStructParCSRVector_fStub.c bHYPRE_SStructStencil_fStub.c            \
  bHYPRE_SStructVariable_fStub.c bHYPRE_SStructVectorView_fStub.c             \
  bHYPRE_SStructVector_fStub.c bHYPRE_SStruct_MatrixVectorView_fStub.c        \
  bHYPRE_Solver_fStub.c bHYPRE_StructDiagScale_fStub.c                        \
  bHYPRE_StructGrid_fStub.c bHYPRE_StructMatrixView_fStub.c                   \
  bHYPRE_StructMatrix_fStub.c bHYPRE_StructPFMG_fStub.c                       \
  bHYPRE_StructSMG_fStub.c bHYPRE_StructStencil_fStub.c                       \
  bHYPRE_StructVectorView_fStub.c bHYPRE_StructVector_fStub.c                 \
  bHYPRE_Vector_fStub.c sidl_BaseClass_fStub.c sidl_BaseException_fStub.c     \
  sidl_BaseInterface_fStub.c sidl_ClassInfoI_fStub.c sidl_ClassInfo_fStub.c   \
  sidl_DFinder_fStub.c sidl_DLL_fStub.c sidl_Finder_fStub.c                   \
  sidl_InvViolation_fStub.c sidl_Loader_fStub.c sidl_PostViolation_fStub.c    \
  sidl_PreViolation_fStub.c sidl_Resolve_fStub.c sidl_SIDLException_fStub.c   \
  sidl_Scope_fStub.c sidl_array_fStub.c sidl_bool_fStub.c sidl_char_fStub.c   \
  sidl_dcomplex_fStub.c sidl_double_fStub.c sidl_fcomplex_fStub.c             \
  sidl_float_fStub.c sidl_int_fStub.c sidl_io_Deserializer_fStub.c            \
  sidl_io_IOException_fStub.c sidl_io_Serializeable_fStub.c                   \
  sidl_io_Serializer_fStub.c sidl_long_fStub.c sidl_opaque_fStub.c            \
  sidl_rmi_ConnectRegistry_fStub.c sidl_rmi_InstanceHandle_fStub.c            \
  sidl_rmi_InstanceRegistry_fStub.c sidl_rmi_Invocation_fStub.c               \
  sidl_rmi_NetworkException_fStub.c sidl_rmi_ProtocolFactory_fStub.c          \
  sidl_rmi_Response_fStub.c sidl_string_fStub.c
