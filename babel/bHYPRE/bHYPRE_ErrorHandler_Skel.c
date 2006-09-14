/*
 * File:          bHYPRE_ErrorHandler_Skel.c
 * Symbol:        bHYPRE.ErrorHandler-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.ErrorHandler
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_ErrorHandler_IOR.h"
#include "bHYPRE_ErrorHandler.h"
#include <stddef.h>

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
extern "C" {
#endif

void
bHYPRE_ErrorHandler__set_epv(struct bHYPRE_ErrorHandler__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_ErrorHandler__ctor;
  epv->f__ctor2 = impl_bHYPRE_ErrorHandler__ctor2;
  epv->f__dtor = impl_bHYPRE_ErrorHandler__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_ErrorHandler__set_sepv(struct bHYPRE_ErrorHandler__sepv *sepv)
{
  sepv->f_Check = impl_bHYPRE_ErrorHandler_Check;
  sepv->f_Describe = impl_bHYPRE_ErrorHandler_Describe;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_ErrorHandler__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_ErrorHandler__load(&_throwaway_exception);
}
struct bHYPRE_ErrorHandler__object* 
  skel_bHYPRE_ErrorHandler_fconnect_bHYPRE_ErrorHandler(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fconnect_bHYPRE_ErrorHandler(url, ar, _ex);
}

struct bHYPRE_ErrorHandler__object* 
  skel_bHYPRE_ErrorHandler_fcast_bHYPRE_ErrorHandler(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fcast_bHYPRE_ErrorHandler(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_ErrorHandler_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_ErrorHandler_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_ErrorHandler_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_ErrorHandler_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_ErrorHandler_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_ErrorHandler_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_ErrorHandler_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_ErrorHandler_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_ErrorHandler_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_ErrorHandler__data*
bHYPRE_ErrorHandler__get_data(bHYPRE_ErrorHandler self)
{
  return (struct bHYPRE_ErrorHandler__data*)(self ? self->d_data : NULL);
}

void bHYPRE_ErrorHandler__set_data(
  bHYPRE_ErrorHandler self,
  struct bHYPRE_ErrorHandler__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
