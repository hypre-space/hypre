/*
 * File:          sidlx_rmi_SimpleOrb_Impl.h
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.SimpleOrb
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_rmi_SimpleOrb_Impl_h
#define included_sidlx_rmi_SimpleOrb_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_SimpleOrb_h
#include "sidlx_rmi_SimpleOrb.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidlx_rmi_SimpleServer_h
#include "sidlx_rmi_SimpleServer.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 41 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._includes) */
#line 45 "sidlx_rmi_SimpleOrb_Impl.h"

/*
 * Private data for class sidlx.rmi.SimpleOrb
 */

struct sidlx_rmi_SimpleOrb__data {
#line 50 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._data) */
#line 57 "sidlx_rmi_SimpleOrb_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimpleOrb__data*
sidlx_rmi_SimpleOrb__get_data(
  sidlx_rmi_SimpleOrb);

extern void
sidlx_rmi_SimpleOrb__set_data(
  sidlx_rmi_SimpleOrb,
  struct sidlx_rmi_SimpleOrb__data*);

extern
void
impl_sidlx_rmi_SimpleOrb__load(
  void);

extern
void
impl_sidlx_rmi_SimpleOrb__ctor(
  /* in */ sidlx_rmi_SimpleOrb self);

extern
void
impl_sidlx_rmi_SimpleOrb__dtor(
  /* in */ sidlx_rmi_SimpleOrb self);

/*
 * User-defined object methods
 */

extern struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleOrb(struct 
  sidlx_rmi_SimpleOrb__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_SimpleOrb_serviceRequest(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleOrb(struct 
  sidlx_rmi_SimpleOrb__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
