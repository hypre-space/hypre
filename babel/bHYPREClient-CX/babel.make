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
STUBHDRS = bHYPRE.hxx bHYPRE_BiCGSTAB.hxx bHYPRE_BoomerAMG.hxx                \
  bHYPRE_CGNR.hxx bHYPRE_CoefficientAccess.hxx bHYPRE_ErrorCode.hxx           \
  bHYPRE_ErrorHandler.hxx bHYPRE_Euclid.hxx bHYPRE_GMRES.hxx                  \
  bHYPRE_HGMRES.hxx bHYPRE_HPCG.hxx bHYPRE_Hybrid.hxx bHYPRE_IJMatrixView.hxx \
  bHYPRE_IJParCSRMatrix.hxx bHYPRE_IJParCSRVector.hxx bHYPRE_IJVectorView.hxx \
  bHYPRE_IdentitySolver.hxx bHYPRE_MPICommunicator.hxx                        \
  bHYPRE_MatrixVectorView.hxx bHYPRE_Operator.hxx bHYPRE_PCG.hxx              \
  bHYPRE_ParCSRDiagScale.hxx bHYPRE_ParaSails.hxx bHYPRE_Pilut.hxx            \
  bHYPRE_PreconditionedSolver.hxx bHYPRE_ProblemDefinition.hxx                \
  bHYPRE_SStructDiagScale.hxx bHYPRE_SStructGraph.hxx bHYPRE_SStructGrid.hxx  \
  bHYPRE_SStructMatrix.hxx bHYPRE_SStructMatrixVectorView.hxx                 \
  bHYPRE_SStructMatrixView.hxx bHYPRE_SStructParCSRMatrix.hxx                 \
  bHYPRE_SStructParCSRVector.hxx bHYPRE_SStructSplit.hxx                      \
  bHYPRE_SStructStencil.hxx bHYPRE_SStructVariable.hxx                        \
  bHYPRE_SStructVector.hxx bHYPRE_SStructVectorView.hxx bHYPRE_Schwarz.hxx    \
  bHYPRE_Solver.hxx bHYPRE_StructDiagScale.hxx bHYPRE_StructGrid.hxx          \
  bHYPRE_StructJacobi.hxx bHYPRE_StructMatrix.hxx bHYPRE_StructMatrixView.hxx \
  bHYPRE_StructPFMG.hxx bHYPRE_StructSMG.hxx bHYPRE_StructStencil.hxx         \
  bHYPRE_StructVector.hxx bHYPRE_StructVectorView.hxx bHYPRE_Vector.hxx       \
  sidl.hxx sidl_BaseClass.hxx sidl_BaseException.hxx sidl_BaseInterface.hxx   \
  sidl_CastException.hxx sidl_ClassInfo.hxx sidl_ClassInfoI.hxx               \
  sidl_DFinder.hxx sidl_DLL.hxx sidl_Finder.hxx sidl_InvViolation.hxx         \
  sidl_LangSpecificException.hxx sidl_Loader.hxx                              \
  sidl_MemoryAllocationException.hxx sidl_NotImplementedException.hxx         \
  sidl_PostViolation.hxx sidl_PreViolation.hxx sidl_Resolve.hxx               \
  sidl_RuntimeException.hxx sidl_SIDLException.hxx sidl_Scope.hxx sidl_io.hxx \
  sidl_io_Deserializer.hxx sidl_io_IOException.hxx sidl_io_Serializable.hxx   \
  sidl_io_Serializer.hxx sidl_rmi.hxx sidl_rmi_BindException.hxx              \
  sidl_rmi_Call.hxx sidl_rmi_ConnectException.hxx                             \
  sidl_rmi_ConnectRegistry.hxx sidl_rmi_InstanceHandle.hxx                    \
  sidl_rmi_InstanceRegistry.hxx sidl_rmi_Invocation.hxx                       \
  sidl_rmi_MalformedURLException.hxx sidl_rmi_NetworkException.hxx            \
  sidl_rmi_NoRouteToHostException.hxx sidl_rmi_NoServerException.hxx          \
  sidl_rmi_ObjectDoesNotExistException.hxx sidl_rmi_ProtocolException.hxx     \
  sidl_rmi_ProtocolFactory.hxx sidl_rmi_Response.hxx sidl_rmi_Return.hxx      \
  sidl_rmi_ServerInfo.hxx sidl_rmi_ServerRegistry.hxx sidl_rmi_Ticket.hxx     \
  sidl_rmi_TicketBook.hxx sidl_rmi_TimeOutException.hxx                       \
  sidl_rmi_UnexpectedCloseException.hxx sidl_rmi_UnknownHostException.hxx
STUBSRCS = bHYPRE_BiCGSTAB.cxx bHYPRE_BoomerAMG.cxx bHYPRE_CGNR.cxx           \
  bHYPRE_CoefficientAccess.cxx bHYPRE_ErrorHandler.cxx bHYPRE_Euclid.cxx      \
  bHYPRE_GMRES.cxx bHYPRE_HGMRES.cxx bHYPRE_HPCG.cxx bHYPRE_Hybrid.cxx        \
  bHYPRE_IJMatrixView.cxx bHYPRE_IJParCSRMatrix.cxx bHYPRE_IJParCSRVector.cxx \
  bHYPRE_IJVectorView.cxx bHYPRE_IdentitySolver.cxx                           \
  bHYPRE_MPICommunicator.cxx bHYPRE_MatrixVectorView.cxx bHYPRE_Operator.cxx  \
  bHYPRE_PCG.cxx bHYPRE_ParCSRDiagScale.cxx bHYPRE_ParaSails.cxx              \
  bHYPRE_Pilut.cxx bHYPRE_PreconditionedSolver.cxx                            \
  bHYPRE_ProblemDefinition.cxx bHYPRE_SStructDiagScale.cxx                    \
  bHYPRE_SStructGraph.cxx bHYPRE_SStructGrid.cxx bHYPRE_SStructMatrix.cxx     \
  bHYPRE_SStructMatrixVectorView.cxx bHYPRE_SStructMatrixView.cxx             \
  bHYPRE_SStructParCSRMatrix.cxx bHYPRE_SStructParCSRVector.cxx               \
  bHYPRE_SStructSplit.cxx bHYPRE_SStructStencil.cxx bHYPRE_SStructVector.cxx  \
  bHYPRE_SStructVectorView.cxx bHYPRE_Schwarz.cxx bHYPRE_Solver.cxx           \
  bHYPRE_StructDiagScale.cxx bHYPRE_StructGrid.cxx bHYPRE_StructJacobi.cxx    \
  bHYPRE_StructMatrix.cxx bHYPRE_StructMatrixView.cxx bHYPRE_StructPFMG.cxx   \
  bHYPRE_StructSMG.cxx bHYPRE_StructStencil.cxx bHYPRE_StructVector.cxx       \
  bHYPRE_StructVectorView.cxx bHYPRE_Vector.cxx sidl_BaseClass.cxx            \
  sidl_BaseException.cxx sidl_BaseInterface.cxx sidl_CastException.cxx        \
  sidl_ClassInfo.cxx sidl_ClassInfoI.cxx sidl_DFinder.cxx sidl_DLL.cxx        \
  sidl_Finder.cxx sidl_InvViolation.cxx sidl_LangSpecificException.cxx        \
  sidl_Loader.cxx sidl_MemoryAllocationException.cxx                          \
  sidl_NotImplementedException.cxx sidl_PostViolation.cxx                     \
  sidl_PreViolation.cxx sidl_RuntimeException.cxx sidl_SIDLException.cxx      \
  sidl_io_Deserializer.cxx sidl_io_IOException.cxx sidl_io_Serializable.cxx   \
  sidl_io_Serializer.cxx sidl_rmi_BindException.cxx sidl_rmi_Call.cxx         \
  sidl_rmi_ConnectException.cxx sidl_rmi_ConnectRegistry.cxx                  \
  sidl_rmi_InstanceHandle.cxx sidl_rmi_InstanceRegistry.cxx                   \
  sidl_rmi_Invocation.cxx sidl_rmi_MalformedURLException.cxx                  \
  sidl_rmi_NetworkException.cxx sidl_rmi_NoRouteToHostException.cxx           \
  sidl_rmi_NoServerException.cxx sidl_rmi_ObjectDoesNotExistException.cxx     \
  sidl_rmi_ProtocolException.cxx sidl_rmi_ProtocolFactory.cxx                 \
  sidl_rmi_Response.cxx sidl_rmi_Return.cxx sidl_rmi_ServerInfo.cxx           \
  sidl_rmi_ServerRegistry.cxx sidl_rmi_Ticket.cxx sidl_rmi_TicketBook.cxx     \
  sidl_rmi_TimeOutException.cxx sidl_rmi_UnexpectedCloseException.cxx         \
  sidl_rmi_UnknownHostException.cxx
