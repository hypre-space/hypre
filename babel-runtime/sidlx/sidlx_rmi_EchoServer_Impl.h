/*
 * File:          sidlx_rmi_EchoServer_Impl.h
 * Symbol:        sidlx.rmi.EchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.rmi.EchoServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_rmi_EchoServer_Impl_h
#define included_sidlx_rmi_EchoServer_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_rmi_EchoServer_h
#include "sidlx_rmi_EchoServer.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.EchoServer._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.EchoServer._includes) */

/*
 * Private data for class sidlx.rmi.EchoServer
 */

struct sidlx_rmi_EchoServer__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.EchoServer._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.EchoServer._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_EchoServer__data*
sidlx_rmi_EchoServer__get_data(
  sidlx_rmi_EchoServer);

extern void
sidlx_rmi_EchoServer__set_data(
  sidlx_rmi_EchoServer,
  struct sidlx_rmi_EchoServer__data*);

extern void
impl_sidlx_rmi_EchoServer__ctor(
  sidlx_rmi_EchoServer);

extern void
impl_sidlx_rmi_EchoServer__dtor(
  sidlx_rmi_EchoServer);

/*
 * User-defined object methods
 */

extern void
impl_sidlx_rmi_EchoServer_serviceRequest(
  sidlx_rmi_EchoServer,
  int32_t,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
