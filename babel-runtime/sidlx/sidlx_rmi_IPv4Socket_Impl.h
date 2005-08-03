/*
 * File:          sidlx_rmi_IPv4Socket_Impl.h
 * Symbol:        sidlx.rmi.IPv4Socket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for sidlx.rmi.IPv4Socket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_rmi_IPv4Socket_Impl_h
#define included_sidlx_rmi_IPv4Socket_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidlx_rmi_IPv4Socket_h
#include "sidlx_rmi_IPv4Socket.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 38 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket._includes) */
/* insert implementation here: sidlx.rmi.IPv4Socket._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket._includes) */
#line 42 "sidlx_rmi_IPv4Socket_Impl.h"

/*
 * Private data for class sidlx.rmi.IPv4Socket
 */

struct sidlx_rmi_IPv4Socket__data {
#line 47 "../../../babel/runtime/sidlx/sidlx_rmi_IPv4Socket_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.IPv4Socket._data) */
  /* insert implementation here: sidlx.rmi.IPv4Socket._data (private data members) */
  int fd; /* File descriptor (Socket) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.IPv4Socket._data) */
#line 54 "sidlx_rmi_IPv4Socket_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_IPv4Socket__data*
sidlx_rmi_IPv4Socket__get_data(
  sidlx_rmi_IPv4Socket);

extern void
sidlx_rmi_IPv4Socket__set_data(
  sidlx_rmi_IPv4Socket,
  struct sidlx_rmi_IPv4Socket__data*);

extern
void
impl_sidlx_rmi_IPv4Socket__load(
  void);

extern
void
impl_sidlx_rmi_IPv4Socket__ctor(
  /* in */ sidlx_rmi_IPv4Socket self);

extern
void
impl_sidlx_rmi_IPv4Socket__dtor(
  /* in */ sidlx_rmi_IPv4Socket self);

/*
 * User-defined object methods
 */

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_sidlx_rmi_IPv4Socket_getsockname(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_getpeername(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_close(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readn(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readline(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readstring(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readstring_alloc(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readint(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_writen(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_writestring(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_writeint(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_IPv4Socket_setFileDescriptor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t fd,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_getFileDescriptor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
