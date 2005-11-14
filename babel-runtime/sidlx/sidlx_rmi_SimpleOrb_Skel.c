/*
 * File:          sidlx_rmi_SimpleOrb_Skel.c
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for sidlx.rmi.SimpleOrb
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_rmi_SimpleOrb_IOR.h"
#include "sidlx_rmi_SimpleOrb.h"
#include <stddef.h>

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
extern "C" {
#endif

void
sidlx_rmi_SimpleOrb__set_epv(struct sidlx_rmi_SimpleOrb__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimpleOrb__ctor;
  epv->f__dtor = impl_sidlx_rmi_SimpleOrb__dtor;
  epv->f_serviceRequest = impl_sidlx_rmi_SimpleOrb_serviceRequest;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimpleOrb__call_load(void) { 
  impl_sidlx_rmi_SimpleOrb__load();
}
struct sidlx_rmi_SimpleOrb__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleOrb(struct 
  sidlx_rmi_SimpleOrb__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleOrb(obj);
}

struct sidl_SIDLException__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_SIDLException(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_SIDLException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseInterface(obj);
}

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleServer(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_SimpleOrb__data*
sidlx_rmi_SimpleOrb__get_data(sidlx_rmi_SimpleOrb self)
{
  return (struct sidlx_rmi_SimpleOrb__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimpleOrb__set_data(
  sidlx_rmi_SimpleOrb self,
  struct sidlx_rmi_SimpleOrb__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
