/*
 * File:          sidlx_rmi_GenNetworkException_Impl.h
 * Symbol:        sidlx.rmi.GenNetworkException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.GenNetworkException
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_GenNetworkException_Impl_h
#define included_sidlx_rmi_GenNetworkException_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_GenNetworkException_h
#include "sidlx_rmi_GenNetworkException.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_io_IOException_h
#include "sidl_io_IOException.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 44 "../../../babel/runtime/sidlx/sidlx_rmi_GenNetworkException_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._includes) */
/* insert implementation here: sidlx.rmi.GenNetworkException._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._includes) */
#line 48 "sidlx_rmi_GenNetworkException_Impl.h"

/*
 * Private data for class sidlx.rmi.GenNetworkException
 */

struct sidlx_rmi_GenNetworkException__data {
#line 53 "../../../babel/runtime/sidlx/sidlx_rmi_GenNetworkException_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._data) */
  /* insert implementation here: sidlx.rmi.GenNetworkException._data (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._data) */
#line 60 "sidlx_rmi_GenNetworkException_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_GenNetworkException__data*
sidlx_rmi_GenNetworkException__get_data(
  sidlx_rmi_GenNetworkException);

extern void
sidlx_rmi_GenNetworkException__set_data(
  sidlx_rmi_GenNetworkException,
  struct sidlx_rmi_GenNetworkException__data*);

extern
void
impl_sidlx_rmi_GenNetworkException__load(
  void);

extern
void
impl_sidlx_rmi_GenNetworkException__ctor(
  /* in */ sidlx_rmi_GenNetworkException self);

extern
void
impl_sidlx_rmi_GenNetworkException__dtor(
  /* in */ sidlx_rmi_GenNetworkException self);

/*
 * User-defined object methods
 */

extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
