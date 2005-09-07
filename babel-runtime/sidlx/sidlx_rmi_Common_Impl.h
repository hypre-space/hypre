/*
 * File:          sidlx_rmi_Common_Impl.h
 * Symbol:        sidlx.rmi.Common-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.Common
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_rmi_Common_Impl_h
#define included_sidlx_rmi_Common_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_Common_h
#include "sidlx_rmi_Common.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 35 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._includes) */
/* insert implementation here: sidlx.rmi.Common._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._includes) */
#line 39 "sidlx_rmi_Common_Impl.h"

/*
 * Private data for class sidlx.rmi.Common
 */

struct sidlx_rmi_Common__data {
#line 44 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._data) */
  /* insert implementation here: sidlx.rmi.Common._data (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._data) */
#line 51 "sidlx_rmi_Common_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_Common__data*
sidlx_rmi_Common__get_data(
  sidlx_rmi_Common);

extern void
sidlx_rmi_Common__set_data(
  sidlx_rmi_Common,
  struct sidlx_rmi_Common__data*);

extern
void
impl_sidlx_rmi_Common__load(
  void);

extern
void
impl_sidlx_rmi_Common__ctor(
  /* in */ sidlx_rmi_Common self);

extern
void
impl_sidlx_rmi_Common__dtor(
  /* in */ sidlx_rmi_Common self);

/*
 * User-defined object methods
 */

extern
int32_t
impl_sidlx_rmi_Common_fork(
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_Common_gethostbyname(
  /* in */ const char* hostname,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
