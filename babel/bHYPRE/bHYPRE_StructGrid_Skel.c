/*
 * File:          bHYPRE_StructGrid_Skel.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_StructGrid_IOR.h"
#include "bHYPRE_StructGrid.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructGrid__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructGrid__ctor(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructGrid__ctor2(
  /* in */ bHYPRE_StructGrid self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructGrid__dtor(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_StructGrid
impl_bHYPRE_StructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  /* in */ bHYPRE_StructGrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructGrid_Destroy(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructGrid_SetDimension(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructGrid_Assemble(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* ilower_proxy = sidl_int__array_ensure(ilower, 1,
    sidl_column_major_order);
  int32_t* ilower_tmp = ilower_proxy->d_firstElement;
  struct sidl_int__array* iupper_proxy = sidl_int__array_ensure(iupper, 1,
    sidl_column_major_order);
  int32_t* iupper_tmp = iupper_proxy->d_firstElement;
  int32_t dim = sidlLength(iupper_proxy,0);
  _return =
    impl_bHYPRE_StructGrid_SetExtents(
      self,
      ilower_tmp,
      iupper_tmp,
      dim,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ struct sidl_int__array* periodic,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* periodic_proxy = sidl_int__array_ensure(periodic, 1,
    sidl_column_major_order);
  int32_t* periodic_tmp = periodic_proxy->d_firstElement;
  int32_t dim = sidlLength(periodic_proxy,0);
  _return =
    impl_bHYPRE_StructGrid_SetPeriodic(
      self,
      periodic_tmp,
      dim,
      _ex);
  return _return;
}

static int32_t
skel_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* num_ghost_proxy = sidl_int__array_ensure(num_ghost, 1,
    sidl_column_major_order);
  int32_t* num_ghost_tmp = num_ghost_proxy->d_firstElement;
  int32_t dim2 = sidlLength(num_ghost_proxy,0);
  _return =
    impl_bHYPRE_StructGrid_SetNumGhost(
      self,
      num_ghost_tmp,
      dim2,
      _ex);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructGrid__set_epv(struct bHYPRE_StructGrid__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructGrid__ctor;
  epv->f__ctor2 = impl_bHYPRE_StructGrid__ctor2;
  epv->f__dtor = impl_bHYPRE_StructGrid__dtor;
  epv->f_SetCommunicator = impl_bHYPRE_StructGrid_SetCommunicator;
  epv->f_Destroy = impl_bHYPRE_StructGrid_Destroy;
  epv->f_SetDimension = impl_bHYPRE_StructGrid_SetDimension;
  epv->f_SetExtents = skel_bHYPRE_StructGrid_SetExtents;
  epv->f_SetPeriodic = skel_bHYPRE_StructGrid_SetPeriodic;
  epv->f_SetNumGhost = skel_bHYPRE_StructGrid_SetNumGhost;
  epv->f_Assemble = impl_bHYPRE_StructGrid_Assemble;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructGrid__set_sepv(struct bHYPRE_StructGrid__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructGrid_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructGrid__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_StructGrid__load(&_throwaway_exception);
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(url, ar, _ex);
}

struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_StructGrid_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fcast_bHYPRE_MPICommunicator(bi, _ex);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(url, ar, _ex);
}

struct bHYPRE_StructGrid__object* 
  skel_bHYPRE_StructGrid_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fcast_bHYPRE_StructGrid(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructGrid_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructGrid_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructGrid_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructGrid_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructGrid_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructGrid_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_StructGrid__data*
bHYPRE_StructGrid__get_data(bHYPRE_StructGrid self)
{
  return (struct bHYPRE_StructGrid__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructGrid__set_data(
  bHYPRE_StructGrid self,
  struct bHYPRE_StructGrid__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
