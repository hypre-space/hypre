/*
 * File:          sidlx_rmi_SimpleServer_Impl.h
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.rmi.SimpleServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_rmi_SimpleServer_Impl_h
#define included_sidlx_rmi_SimpleServer_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_rmi_SimpleServer_h
#include "sidlx_rmi_SimpleServer.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._includes) */
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._includes) */

/*
 * Private data for class sidlx.rmi.SimpleServer
 */

struct sidlx_rmi_SimpleServer__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._data) */
  int d_listen_socket;
  struct sockaddr_in d_serv_addr;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimpleServer__data*
sidlx_rmi_SimpleServer__get_data(
  sidlx_rmi_SimpleServer);

extern void
sidlx_rmi_SimpleServer__set_data(
  sidlx_rmi_SimpleServer,
  struct sidlx_rmi_SimpleServer__data*);

extern void
impl_sidlx_rmi_SimpleServer__ctor(
  sidlx_rmi_SimpleServer);

extern void
impl_sidlx_rmi_SimpleServer__dtor(
  sidlx_rmi_SimpleServer);

/*
 * User-defined object methods
 */

extern void
impl_sidlx_rmi_SimpleServer_setPort(
  sidlx_rmi_SimpleServer,
  int32_t);

extern void
impl_sidlx_rmi_SimpleServer_run(
  sidlx_rmi_SimpleServer,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
