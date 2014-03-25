IMPLHDRS = sidl_BaseClass_Impl.h sidl_CastException_Impl.h                    \
  sidl_ClassInfoI_Impl.h sidl_DFinder_Impl.h sidl_DLL_Impl.h                  \
  sidl_InvViolation_Impl.h sidl_LangSpecificException_Impl.h                  \
  sidl_Loader_Impl.h sidl_MemoryAllocationException_Impl.h                    \
  sidl_NotImplementedException_Impl.h sidl_PostViolation_Impl.h               \
  sidl_PreViolation_Impl.h sidl_SIDLException_Impl.h                          \
  sidl_io_IOException_Impl.h sidl_rmi_BindException_Impl.h                    \
  sidl_rmi_ConnectException_Impl.h sidl_rmi_ConnectRegistry_Impl.h            \
  sidl_rmi_InstanceRegistry_Impl.h sidl_rmi_MalformedURLException_Impl.h      \
  sidl_rmi_NetworkException_Impl.h sidl_rmi_NoRouteToHostException_Impl.h     \
  sidl_rmi_NoServerException_Impl.h                                           \
  sidl_rmi_ObjectDoesNotExistException_Impl.h                                 \
  sidl_rmi_ProtocolException_Impl.h sidl_rmi_ProtocolFactory_Impl.h           \
  sidl_rmi_ServerRegistry_Impl.h sidl_rmi_TimeOutException_Impl.h             \
  sidl_rmi_UnexpectedCloseException_Impl.h                                    \
  sidl_rmi_UnknownHostException_Impl.h
IMPLSRCS = sidl_BaseClass_Impl.c sidl_CastException_Impl.c                    \
  sidl_ClassInfoI_Impl.c sidl_DFinder_Impl.c sidl_DLL_Impl.c                  \
  sidl_InvViolation_Impl.c sidl_LangSpecificException_Impl.c                  \
  sidl_Loader_Impl.c sidl_MemoryAllocationException_Impl.c                    \
  sidl_NotImplementedException_Impl.c sidl_PostViolation_Impl.c               \
  sidl_PreViolation_Impl.c sidl_SIDLException_Impl.c                          \
  sidl_io_IOException_Impl.c sidl_rmi_BindException_Impl.c                    \
  sidl_rmi_ConnectException_Impl.c sidl_rmi_ConnectRegistry_Impl.c            \
  sidl_rmi_InstanceRegistry_Impl.c sidl_rmi_MalformedURLException_Impl.c      \
  sidl_rmi_NetworkException_Impl.c sidl_rmi_NoRouteToHostException_Impl.c     \
  sidl_rmi_NoServerException_Impl.c                                           \
  sidl_rmi_ObjectDoesNotExistException_Impl.c                                 \
  sidl_rmi_ProtocolException_Impl.c sidl_rmi_ProtocolFactory_Impl.c           \
  sidl_rmi_ServerRegistry_Impl.c sidl_rmi_TimeOutException_Impl.c             \
  sidl_rmi_UnexpectedCloseException_Impl.c                                    \
  sidl_rmi_UnknownHostException_Impl.c
IORHDRS = sidl_BaseClass_IOR.h sidl_BaseException_IOR.h                       \
  sidl_BaseInterface_IOR.h sidl_CastException_IOR.h sidl_ClassInfoI_IOR.h     \
  sidl_ClassInfo_IOR.h sidl_DFinder_IOR.h sidl_DLL_IOR.h sidl_Finder_IOR.h    \
  sidl_IOR.h sidl_InvViolation_IOR.h sidl_LangSpecificException_IOR.h         \
  sidl_Loader_IOR.h sidl_MemoryAllocationException_IOR.h                      \
  sidl_NotImplementedException_IOR.h sidl_PostViolation_IOR.h                 \
  sidl_PreViolation_IOR.h sidl_Resolve_IOR.h sidl_RuntimeException_IOR.h      \
  sidl_SIDLException_IOR.h sidl_Scope_IOR.h sidl_io_Deserializer_IOR.h        \
  sidl_io_IOException_IOR.h sidl_io_IOR.h sidl_io_Serializable_IOR.h          \
  sidl_io_Serializer_IOR.h sidl_rmi_BindException_IOR.h sidl_rmi_Call_IOR.h   \
  sidl_rmi_ConnectException_IOR.h sidl_rmi_ConnectRegistry_IOR.h              \
  sidl_rmi_IOR.h sidl_rmi_InstanceHandle_IOR.h                                \
  sidl_rmi_InstanceRegistry_IOR.h sidl_rmi_Invocation_IOR.h                   \
  sidl_rmi_MalformedURLException_IOR.h sidl_rmi_NetworkException_IOR.h        \
  sidl_rmi_NoRouteToHostException_IOR.h sidl_rmi_NoServerException_IOR.h      \
  sidl_rmi_ObjectDoesNotExistException_IOR.h sidl_rmi_ProtocolException_IOR.h \
  sidl_rmi_ProtocolFactory_IOR.h sidl_rmi_Response_IOR.h                      \
  sidl_rmi_Return_IOR.h sidl_rmi_ServerInfo_IOR.h                             \
  sidl_rmi_ServerRegistry_IOR.h sidl_rmi_TicketBook_IOR.h                     \
  sidl_rmi_Ticket_IOR.h sidl_rmi_TimeOutException_IOR.h                       \
  sidl_rmi_UnexpectedCloseException_IOR.h sidl_rmi_UnknownHostException_IOR.h
IORSRCS = sidl_BaseClass_IOR.c sidl_CastException_IOR.c sidl_ClassInfoI_IOR.c \
  sidl_DFinder_IOR.c sidl_DLL_IOR.c sidl_InvViolation_IOR.c                   \
  sidl_LangSpecificException_IOR.c sidl_Loader_IOR.c                          \
  sidl_MemoryAllocationException_IOR.c sidl_NotImplementedException_IOR.c     \
  sidl_PostViolation_IOR.c sidl_PreViolation_IOR.c sidl_SIDLException_IOR.c   \
  sidl_io_IOException_IOR.c sidl_rmi_BindException_IOR.c                      \
  sidl_rmi_ConnectException_IOR.c sidl_rmi_ConnectRegistry_IOR.c              \
  sidl_rmi_InstanceRegistry_IOR.c sidl_rmi_MalformedURLException_IOR.c        \
  sidl_rmi_NetworkException_IOR.c sidl_rmi_NoRouteToHostException_IOR.c       \
  sidl_rmi_NoServerException_IOR.c sidl_rmi_ObjectDoesNotExistException_IOR.c \
  sidl_rmi_ProtocolException_IOR.c sidl_rmi_ProtocolFactory_IOR.c             \
  sidl_rmi_ServerRegistry_IOR.c sidl_rmi_TimeOutException_IOR.c               \
  sidl_rmi_UnexpectedCloseException_IOR.c sidl_rmi_UnknownHostException_IOR.c
SKELSRCS = sidl_BaseClass_Skel.c sidl_CastException_Skel.c                    \
  sidl_ClassInfoI_Skel.c sidl_DFinder_Skel.c sidl_DLL_Skel.c                  \
  sidl_InvViolation_Skel.c sidl_LangSpecificException_Skel.c                  \
  sidl_Loader_Skel.c sidl_MemoryAllocationException_Skel.c                    \
  sidl_NotImplementedException_Skel.c sidl_PostViolation_Skel.c               \
  sidl_PreViolation_Skel.c sidl_SIDLException_Skel.c                          \
  sidl_io_IOException_Skel.c sidl_rmi_BindException_Skel.c                    \
  sidl_rmi_ConnectException_Skel.c sidl_rmi_ConnectRegistry_Skel.c            \
  sidl_rmi_InstanceRegistry_Skel.c sidl_rmi_MalformedURLException_Skel.c      \
  sidl_rmi_NetworkException_Skel.c sidl_rmi_NoRouteToHostException_Skel.c     \
  sidl_rmi_NoServerException_Skel.c                                           \
  sidl_rmi_ObjectDoesNotExistException_Skel.c                                 \
  sidl_rmi_ProtocolException_Skel.c sidl_rmi_ProtocolFactory_Skel.c           \
  sidl_rmi_ServerRegistry_Skel.c sidl_rmi_TimeOutException_Skel.c             \
  sidl_rmi_UnexpectedCloseException_Skel.c                                    \
  sidl_rmi_UnknownHostException_Skel.c
STUBHDRS = sidl.h sidl_BaseClass.h sidl_BaseException.h sidl_BaseInterface.h  \
  sidl_CastException.h sidl_ClassInfo.h sidl_ClassInfoI.h sidl_DFinder.h      \
  sidl_DLL.h sidl_Finder.h sidl_InvViolation.h sidl_LangSpecificException.h   \
  sidl_Loader.h sidl_MemoryAllocationException.h                              \
  sidl_NotImplementedException.h sidl_PostViolation.h sidl_PreViolation.h     \
  sidl_Resolve.h sidl_RuntimeException.h sidl_SIDLException.h sidl_Scope.h    \
  sidl_io.h sidl_io_Deserializer.h sidl_io_IOException.h                      \
  sidl_io_Serializable.h sidl_io_Serializer.h sidl_rmi.h                      \
  sidl_rmi_BindException.h sidl_rmi_Call.h sidl_rmi_ConnectException.h        \
  sidl_rmi_ConnectRegistry.h sidl_rmi_InstanceHandle.h                        \
  sidl_rmi_InstanceRegistry.h sidl_rmi_Invocation.h                           \
  sidl_rmi_MalformedURLException.h sidl_rmi_NetworkException.h                \
  sidl_rmi_NoRouteToHostException.h sidl_rmi_NoServerException.h              \
  sidl_rmi_ObjectDoesNotExistException.h sidl_rmi_ProtocolException.h         \
  sidl_rmi_ProtocolFactory.h sidl_rmi_Response.h sidl_rmi_Return.h            \
  sidl_rmi_ServerInfo.h sidl_rmi_ServerRegistry.h sidl_rmi_Ticket.h           \
  sidl_rmi_TicketBook.h sidl_rmi_TimeOutException.h                           \
  sidl_rmi_UnexpectedCloseException.h sidl_rmi_UnknownHostException.h
STUBSRCS = sidl_BaseClass_Stub.c sidl_BaseException_Stub.c                    \
  sidl_BaseInterface_Stub.c sidl_CastException_Stub.c sidl_ClassInfoI_Stub.c  \
  sidl_ClassInfo_Stub.c sidl_DFinder_Stub.c sidl_DLL_Stub.c                   \
  sidl_Finder_Stub.c sidl_InvViolation_Stub.c                                 \
  sidl_LangSpecificException_Stub.c sidl_Loader_Stub.c                        \
  sidl_MemoryAllocationException_Stub.c sidl_NotImplementedException_Stub.c   \
  sidl_PostViolation_Stub.c sidl_PreViolation_Stub.c sidl_Resolve_Stub.c      \
  sidl_RuntimeException_Stub.c sidl_SIDLException_Stub.c sidl_Scope_Stub.c    \
  sidl_io_Deserializer_Stub.c sidl_io_IOException_Stub.c                      \
  sidl_io_Serializable_Stub.c sidl_io_Serializer_Stub.c                       \
  sidl_rmi_BindException_Stub.c sidl_rmi_Call_Stub.c                          \
  sidl_rmi_ConnectException_Stub.c sidl_rmi_ConnectRegistry_Stub.c            \
  sidl_rmi_InstanceHandle_Stub.c sidl_rmi_InstanceRegistry_Stub.c             \
  sidl_rmi_Invocation_Stub.c sidl_rmi_MalformedURLException_Stub.c            \
  sidl_rmi_NetworkException_Stub.c sidl_rmi_NoRouteToHostException_Stub.c     \
  sidl_rmi_NoServerException_Stub.c                                           \
  sidl_rmi_ObjectDoesNotExistException_Stub.c                                 \
  sidl_rmi_ProtocolException_Stub.c sidl_rmi_ProtocolFactory_Stub.c           \
  sidl_rmi_Response_Stub.c sidl_rmi_Return_Stub.c sidl_rmi_ServerInfo_Stub.c  \
  sidl_rmi_ServerRegistry_Stub.c sidl_rmi_TicketBook_Stub.c                   \
  sidl_rmi_Ticket_Stub.c sidl_rmi_TimeOutException_Stub.c                     \
  sidl_rmi_UnexpectedCloseException_Stub.c                                    \
  sidl_rmi_UnknownHostException_Stub.c
