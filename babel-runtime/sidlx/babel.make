IMPLHDRS = sidlx_io_IOException_Impl.h sidlx_io_TxtIOStream_Impl.h            \
  sidlx_io_TxtIStream_Impl.h sidlx_io_TxtOStream_Impl.h                       \
  sidlx_rmi_ChildSocket_Impl.h sidlx_rmi_ClientSocket_Impl.h                  \
  sidlx_rmi_Common_Impl.h sidlx_rmi_GenNetworkException_Impl.h                \
  sidlx_rmi_IPv4Socket_Impl.h sidlx_rmi_JimEchoServer_Impl.h                  \
  sidlx_rmi_ServerSocket_Impl.h sidlx_rmi_SimCall_Impl.h                      \
  sidlx_rmi_SimHandle_Impl.h sidlx_rmi_SimReturn_Impl.h                       \
  sidlx_rmi_SimpleOrb_Impl.h sidlx_rmi_SimpleServer_Impl.h                    \
  sidlx_rmi_Simsponse_Impl.h sidlx_rmi_Simvocation_Impl.h
IMPLSRCS = sidlx_io_IOException_Impl.c sidlx_io_TxtIOStream_Impl.c            \
  sidlx_io_TxtIStream_Impl.c sidlx_io_TxtOStream_Impl.c                       \
  sidlx_rmi_ChildSocket_Impl.c sidlx_rmi_ClientSocket_Impl.c                  \
  sidlx_rmi_Common_Impl.c sidlx_rmi_GenNetworkException_Impl.c                \
  sidlx_rmi_IPv4Socket_Impl.c sidlx_rmi_JimEchoServer_Impl.c                  \
  sidlx_rmi_ServerSocket_Impl.c sidlx_rmi_SimCall_Impl.c                      \
  sidlx_rmi_SimHandle_Impl.c sidlx_rmi_SimReturn_Impl.c                       \
  sidlx_rmi_SimpleOrb_Impl.c sidlx_rmi_SimpleServer_Impl.c                    \
  sidlx_rmi_Simsponse_Impl.c sidlx_rmi_Simvocation_Impl.c
IORHDRS = sidlx_IOR.h sidlx_io_IOException_IOR.h sidlx_io_IOR.h               \
  sidlx_io_IOStream_IOR.h sidlx_io_IStream_IOR.h sidlx_io_OStream_IOR.h       \
  sidlx_io_Serializable_IOR.h sidlx_io_TxtIOStream_IOR.h                      \
  sidlx_io_TxtIStream_IOR.h sidlx_io_TxtOStream_IOR.h                         \
  sidlx_rmi_CallType_IOR.h sidlx_rmi_ChildSocket_IOR.h                        \
  sidlx_rmi_ClientSocket_IOR.h sidlx_rmi_Common_IOR.h                         \
  sidlx_rmi_GenNetworkException_IOR.h sidlx_rmi_IOR.h                         \
  sidlx_rmi_IPv4Socket_IOR.h sidlx_rmi_JimEchoServer_IOR.h                    \
  sidlx_rmi_ServerSocket_IOR.h sidlx_rmi_SimCall_IOR.h                        \
  sidlx_rmi_SimHandle_IOR.h sidlx_rmi_SimReturn_IOR.h                         \
  sidlx_rmi_SimpleOrb_IOR.h sidlx_rmi_SimpleServer_IOR.h                      \
  sidlx_rmi_Simsponse_IOR.h sidlx_rmi_Simvocation_IOR.h sidlx_rmi_Socket_IOR.h
IORSRCS = sidlx_io_IOException_IOR.c sidlx_io_TxtIOStream_IOR.c               \
  sidlx_io_TxtIStream_IOR.c sidlx_io_TxtOStream_IOR.c                         \
  sidlx_rmi_ChildSocket_IOR.c sidlx_rmi_ClientSocket_IOR.c                    \
  sidlx_rmi_Common_IOR.c sidlx_rmi_GenNetworkException_IOR.c                  \
  sidlx_rmi_IPv4Socket_IOR.c sidlx_rmi_JimEchoServer_IOR.c                    \
  sidlx_rmi_ServerSocket_IOR.c sidlx_rmi_SimCall_IOR.c                        \
  sidlx_rmi_SimHandle_IOR.c sidlx_rmi_SimReturn_IOR.c                         \
  sidlx_rmi_SimpleOrb_IOR.c sidlx_rmi_SimpleServer_IOR.c                      \
  sidlx_rmi_Simsponse_IOR.c sidlx_rmi_Simvocation_IOR.c
SKELSRCS = sidlx_io_IOException_Skel.c sidlx_io_TxtIOStream_Skel.c            \
  sidlx_io_TxtIStream_Skel.c sidlx_io_TxtOStream_Skel.c                       \
  sidlx_rmi_ChildSocket_Skel.c sidlx_rmi_ClientSocket_Skel.c                  \
  sidlx_rmi_Common_Skel.c sidlx_rmi_GenNetworkException_Skel.c                \
  sidlx_rmi_IPv4Socket_Skel.c sidlx_rmi_JimEchoServer_Skel.c                  \
  sidlx_rmi_ServerSocket_Skel.c sidlx_rmi_SimCall_Skel.c                      \
  sidlx_rmi_SimHandle_Skel.c sidlx_rmi_SimReturn_Skel.c                       \
  sidlx_rmi_SimpleOrb_Skel.c sidlx_rmi_SimpleServer_Skel.c                    \
  sidlx_rmi_Simsponse_Skel.c sidlx_rmi_Simvocation_Skel.c
STUBHDRS = sidlx.h sidlx_io.h sidlx_io_IOException.h sidlx_io_IOStream.h      \
  sidlx_io_IStream.h sidlx_io_OStream.h sidlx_io_Serializable.h               \
  sidlx_io_TxtIOStream.h sidlx_io_TxtIStream.h sidlx_io_TxtOStream.h          \
  sidlx_rmi.h sidlx_rmi_CallType.h sidlx_rmi_ChildSocket.h                    \
  sidlx_rmi_ClientSocket.h sidlx_rmi_Common.h sidlx_rmi_GenNetworkException.h \
  sidlx_rmi_IPv4Socket.h sidlx_rmi_JimEchoServer.h sidlx_rmi_ServerSocket.h   \
  sidlx_rmi_SimCall.h sidlx_rmi_SimHandle.h sidlx_rmi_SimReturn.h             \
  sidlx_rmi_SimpleOrb.h sidlx_rmi_SimpleServer.h sidlx_rmi_Simsponse.h        \
  sidlx_rmi_Simvocation.h sidlx_rmi_Socket.h
STUBSRCS = sidlx_io_IOException_Stub.c sidlx_io_IOStream_Stub.c               \
  sidlx_io_IStream_Stub.c sidlx_io_OStream_Stub.c                             \
  sidlx_io_Serializable_Stub.c sidlx_io_TxtIOStream_Stub.c                    \
  sidlx_io_TxtIStream_Stub.c sidlx_io_TxtOStream_Stub.c                       \
  sidlx_rmi_CallType_Stub.c sidlx_rmi_ChildSocket_Stub.c                      \
  sidlx_rmi_ClientSocket_Stub.c sidlx_rmi_Common_Stub.c                       \
  sidlx_rmi_GenNetworkException_Stub.c sidlx_rmi_IPv4Socket_Stub.c            \
  sidlx_rmi_JimEchoServer_Stub.c sidlx_rmi_ServerSocket_Stub.c                \
  sidlx_rmi_SimCall_Stub.c sidlx_rmi_SimHandle_Stub.c                         \
  sidlx_rmi_SimReturn_Stub.c sidlx_rmi_SimpleOrb_Stub.c                       \
  sidlx_rmi_SimpleServer_Stub.c sidlx_rmi_Simsponse_Stub.c                    \
  sidlx_rmi_Simvocation_Stub.c sidlx_rmi_Socket_Stub.c
