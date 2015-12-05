/*
 * File:          bHYPRE_ErrorHandler_Impl.h
 * Symbol:        bHYPRE.ErrorHandler-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.ErrorHandler
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_ErrorHandler_Impl_h
#define included_bHYPRE_ErrorHandler_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_ErrorHandler_h
#include "bHYPRE_ErrorHandler.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._includes) */
/* Insert-Code-Here {bHYPRE.ErrorHandler._includes} (include files) */
/* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._includes) */

/*
 * Private data for class bHYPRE.ErrorHandler
 */

struct bHYPRE_ErrorHandler__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._data) */
  /* Insert-Code-Here {bHYPRE.ErrorHandler._data} (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_ErrorHandler__data*
bHYPRE_ErrorHandler__get_data(
  bHYPRE_ErrorHandler);

extern void
bHYPRE_ErrorHandler__set_data(
  bHYPRE_ErrorHandler,
  struct bHYPRE_ErrorHandler__data*);

extern
void
impl_bHYPRE_ErrorHandler__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ErrorHandler__ctor(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ErrorHandler__ctor2(
  /* in */ bHYPRE_ErrorHandler self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ErrorHandler__dtor(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
int32_t
impl_bHYPRE_ErrorHandler_Check(
  /* in */ int32_t ierr,
  /* in */ enum bHYPRE_ErrorCode__enum error_code,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_ErrorHandler_Describe(
  /* in */ int32_t ierr,
  /* out */ char** message,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_ErrorHandler__object* 
  impl_bHYPRE_ErrorHandler_fconnect_bHYPRE_ErrorHandler(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ErrorHandler__object* 
  impl_bHYPRE_ErrorHandler_fcast_bHYPRE_ErrorHandler(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_ErrorHandler__object* 
  impl_bHYPRE_ErrorHandler_fconnect_bHYPRE_ErrorHandler(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_ErrorHandler__object* 
  impl_bHYPRE_ErrorHandler_fcast_bHYPRE_ErrorHandler(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
